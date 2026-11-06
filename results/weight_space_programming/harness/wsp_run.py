#!/venv/py312/bin/python
"""Client of the persistent kernel:  run <file.py> | run -c "code" | run -   (stdin)"""
import os, sys, json, time, socket, struct, subprocess, signal, fcntl
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.environ["WSP_RUN"]
MAX_RUNS, TIME_LIMIT, OUT_CUT = 300, 900, 10000
sock_path, pid_path, state_path = [os.path.join(RUN, x) for x in ("kernel.sock", "kernel.pid", "state.json")]


def state_update(fn):
    with open(state_path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        s = json.load(f); r = fn(s)
        f.seek(0); f.truncate(); json.dump(s, f, indent=1)
        return r


def connect():
    c = socket.socket(socket.AF_UNIX); c.connect(sock_path); return c


def kill_kernel():
    try:
        os.kill(int(open(pid_path).read()), signal.SIGKILL)
    except Exception:
        pass
    for p in (sock_path, pid_path):
        if os.path.exists(p):
            os.remove(p)


def ensure():
    try:
        return connect(), False
    except Exception:
        kill_kernel()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false")
        log = open(os.path.join(RUN, "kernel.log"), "a")
        subprocess.Popen(["/venv/py312/bin/python", os.path.join(HERE, "kernel_server.py")], env=env,
                         stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        for _ in range(600):
            time.sleep(0.5)
            try:
                return connect(), True
            except Exception:
                pass
        sys.exit("[kernel failed to start: see kernel.log]")


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    code = a[1] if a[0] == "-c" else (sys.stdin.read() if a[0] == "-" else open(a[0]).read())
    used = state_update(lambda s: s["code_runs"])
    if used >= MAX_RUNS:
        sys.exit(f"[refused: all {MAX_RUNS} code runs are used]")
    conn, fresh = ensure()
    had = state_update(lambda s: (s.get("kernel_started", False), s.__setitem__("kernel_started", True))[0])
    note = "[kernel was (re)started: variables from earlier blocks are lost]\n" if (fresh and had) else ""
    t0 = time.time()
    data = code.encode()
    conn.sendall(struct.pack("!Q", len(data)) + data)
    conn.settimeout(TIME_LIMIT)
    try:
        hdr = b""
        while len(hdr) < 8:
            x = conn.recv(8 - len(hdr))
            if not x:
                raise EOFError("kernel died (memory limit?)")
            hdr += x
        n = struct.unpack("!Q", hdr)[0]; out = b""
        while len(out) < n:
            x = conn.recv(min(1 << 20, n - len(out)))
            if not x:
                break
            out += x
        out = out.decode(errors="replace")
    except (socket.timeout, EOFError) as e:
        kill_kernel()
        out = f"[{'time limit of %d s exceeded' % TIME_LIMIT if isinstance(e, socket.timeout) else e}: " \
              f"the kernel was restarted, its variables are lost]"
    if len(out) > OUT_CUT:
        out = out[:OUT_CUT] + f"\n[output cut at {OUT_CUT} characters]"
    k = state_update(lambda s: (s.__setitem__("code_runs", s["code_runs"] + 1), s["code_runs"])[1])
    with open(os.path.join(RUN, "tool_calls.jsonl"), "a") as f:
        f.write(json.dumps({"kind": "python", "k": k, "t": time.time(), "seconds": round(time.time() - t0, 2),
                            "code": code, "output": out}) + "\n")
    print(note + out + f"\n[code run {k}/{MAX_RUNS}, {time.time() - t0:.1f} s]")


main()
