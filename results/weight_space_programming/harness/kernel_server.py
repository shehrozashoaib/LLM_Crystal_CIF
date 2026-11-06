"""Persistent CPU-only Python kernel (one per run). Variables survive between code blocks."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # no GPU in the kernel
import socket, struct, io, ast, traceback, contextlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RUN = os.environ["WSP_RUN"]
G = {"__name__": "__main__"}
exec(r'''
import os, sys, json, math, numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(48)
from wsp_common import (NAMES, BASE_NAMES, KIND, load, load_original, load_base, save_edit, drop_edit,
                        edited, WORK, RUN)
from transformers import AutoTokenizer as _AT
tok = _AT.from_pretrained(os.path.join(RUN, "tokenizer"))
MODEL = os.path.join(RUN, "model"); PRACTICE = os.path.join(RUN, "practice")
''', G)
os.chdir(os.path.join(RUN, "work"))
path = os.path.join(RUN, "kernel.sock")
if os.path.exists(path):
    os.remove(path)
srv = socket.socket(socket.AF_UNIX); srv.bind(path); srv.listen(1)
open(os.path.join(RUN, "kernel.pid"), "w").write(str(os.getpid()))


def recvn(c, n):
    b = b""
    while len(b) < n:
        x = c.recv(min(1 << 20, n - len(b)))
        if not x:
            raise EOFError
        b += x
    return b


while True:
    conn, _ = srv.accept()
    try:
        n = struct.unpack("!Q", recvn(conn, 8))[0]
        code = recvn(conn, n).decode()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                tree = ast.parse(code)
                last = None
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last = ast.Expression(tree.body.pop().value)
                exec(compile(tree, "<cell>", "exec"), G)
                if last is not None:
                    v = eval(compile(last, "<cell>", "eval"), G)
                    if v is not None:
                        print(repr(v))
            except BaseException:
                traceback.print_exc()
        out = buf.getvalue().encode()
        conn.sendall(struct.pack("!Q", len(out)) + out)
    except Exception:
        pass
    finally:
        conn.close()
