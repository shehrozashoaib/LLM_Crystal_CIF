"""Shared weight access + edit rules for the weight-space-programming harness.
Port of the kernel helpers in the blog's Exp. 8 (load / load_original / save_edit / drop_edit / edited)."""
import os, json, hashlib, shutil
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

RUN = os.environ["WSP_RUN"]
CFG = json.load(open(os.path.join(RUN, "config.json")))
KIND = CFG["kind"]                       # "base" | "adapter"
BASE_DIR = CFG["base_model"]
ADAPTER_DIR = CFG.get("adapter")
WORK = os.path.join(RUN, "work")
EDIT_DIR = os.path.join(WORK, "edited")
CKPT = os.path.join(RUN, "ckpt")
os.makedirs(EDIT_DIR, exist_ok=True)

FORBIDDEN = {"model.norm.weight", "model.embed_tokens.weight", "lm_head.weight"}
BASE_MAP = {k: os.path.join(BASE_DIR, v) for k, v in
            json.load(open(os.path.join(BASE_DIR, "model.safetensors.index.json")))["weight_map"].items()}
BASE_NAMES = sorted(BASE_MAP)
if KIND == "adapter":
    AD_FILE = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")
    with safe_open(AD_FILE, "pt") as _f:
        NAMES = sorted(_f.keys())
else:
    AD_FILE = None
    NAMES = BASE_NAMES
_NAMESET = set(NAMES)


def _read(file, name):
    with safe_open(file, "pt") as f:
        return f.get_tensor(name)


def load_base(name):
    """Frozen base-model tensor (float32)."""
    return _read(BASE_MAP[name], name).float()


def load_original(name):
    """Original (unedited) value of an editable tensor, float32."""
    if name not in _NAMESET:
        raise KeyError(f"{name} is not an editable tensor of this run (see NAMES)")
    return (_read(AD_FILE, name) if KIND == "adapter" else _read(BASE_MAP[name], name)).float()


def _epath(name):
    return os.path.join(EDIT_DIR, name + ".npy")


def edited():
    """Names of the tensors that currently have an edit."""
    return sorted(n[:-4] for n in os.listdir(EDIT_DIR) if n.endswith(".npy"))


def load(name):
    """Current value: your edit if there is one, else the original (float32)."""
    p = _epath(name)
    if os.path.exists(p):
        return torch.from_numpy(np.load(p)).float()
    return load_original(name)


def drop_edit(name):
    p = _epath(name)
    if os.path.exists(p):
        os.remove(p)
        return True
    return False


def save_edit(name, tensor):
    """Replace a tensor by `tensor` (same name and shape). Refuses out-of-scope edits."""
    if name not in _NAMESET:
        raise ValueError(f"refused: {name} is not an editable tensor of this run")
    if name in FORBIDDEN:
        raise ValueError(f"refused: {name} is out of scope (final norm / embedding / output layer)")
    t = torch.as_tensor(np.asarray(tensor) if not torch.is_tensor(tensor) else tensor).detach().float().cpu()
    o = load_original(name)
    if tuple(t.shape) != tuple(o.shape):
        raise ValueError(f"refused: shape {tuple(t.shape)} != original {tuple(o.shape)}")
    if not torch.isfinite(t).all():
        raise ValueError("refused: tensor has non-finite values")
    if torch.equal(t, o):
        drop_edit(name)
        print(f"[save_edit] {name}: identical to the original -> no edit kept")
        return
    c = float((t.double() * o.double()).sum() / (o.double() * o.double()).sum().clamp_min(1e-30))
    if float((t.double() - c * o.double()).norm()) <= 1e-5 * float(t.double().norm().clamp_min(1e-30)):
        raise ValueError(f"refused: this edit only multiplies the whole tensor by a constant ({c:.6g})")
    tmp = _epath(name) + ".tmp.npy"
    np.save(tmp, t.numpy())
    os.replace(tmp, _epath(name))
    d = (t - o)
    print(f"[save_edit] {name}: changed entries {int((d != 0).sum())}/{d.numel()}, "
          f"rel. change ||dW||/||W|| = {float(d.norm() / o.norm().clamp_min(1e-30)):.4g}")


def edit_manifest():
    out = {}
    for n in edited():
        b = open(_epath(n), "rb").read()
        out[n] = hashlib.sha256(b).hexdigest()[:16]
    return out


def build_ckpt(edit_dir=None, out=None):
    """Materialise the edited model for the forward-pass tool / evaluation.
    base run   -> a full model dir (edited shards rewritten in the original dtype, the rest symlinked)
    adapter run-> an adapter dir (adapter_config.json + edited adapter_model.safetensors)
    Edits are cast to the original tensor dtype, as in the blog's build_ckpt."""
    edit_dir = edit_dir or EDIT_DIR
    out = out or os.path.join(CKPT, "edited")
    names = sorted(n[:-4] for n in os.listdir(edit_dir) if n.endswith(".npy"))
    sig = hashlib.sha256(json.dumps(
        [(n, hashlib.sha256(open(os.path.join(edit_dir, n + ".npy"), "rb").read()).hexdigest()) for n in names]
    ).encode()).hexdigest()
    sigf = os.path.join(out, "EDIT_SIG")
    if os.path.exists(sigf) and open(sigf).read() == sig:
        return out, names
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    get = lambda n, like: torch.from_numpy(np.load(os.path.join(edit_dir, n + ".npy"))).to(like.dtype)
    if KIND == "adapter":
        d = load_file(AD_FILE)
        for n in names:
            assert d[n].shape == get(n, d[n]).shape
            d[n] = get(n, d[n])
        save_file(d, os.path.join(out, "adapter_model.safetensors"), metadata={"format": "pt"})
        shutil.copy(os.path.join(ADAPTER_DIR, "adapter_config.json"), out)
    else:
        by_shard = {}
        for n in names:
            by_shard.setdefault(BASE_MAP[n], []).append(n)
        for fn in os.listdir(BASE_DIR):
            src = os.path.join(BASE_DIR, fn)
            if not os.path.isfile(src):
                continue
            if src in by_shard:
                d = load_file(src)
                for n in by_shard[src]:
                    assert d[n].shape == get(n, d[n]).shape
                    d[n] = get(n, d[n])
                save_file(d, os.path.join(out, fn), metadata={"format": "pt"})
            else:
                os.symlink(src, os.path.join(out, fn))
    open(sigf, "w").write(sig)
    return out, names
