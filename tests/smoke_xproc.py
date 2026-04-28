"""跨进程 Python 烟雾测试。"""
import sys, os, time, threading
_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_PY = os.path.normpath(os.path.join(_HERE, "..", "build", "python"))
sys.path.insert(0, _BUILD_PY)

import numpy as np
import multiprocessing as mp
import fastipc


def server_proc(prefix, total, ready, done):
    pools = [(256*1024, 64), (4*1024*1024, 32), (16*1024*1024, 8)]
    srv = fastipc.Server.create(shm_prefix=prefix, max_clients=1, num_workers=1,
                                ring_capacity=64, resp_capacity=64, pools=pools)
    srv.start()
    ready.set()
    n = 0
    while n < total:
        req = srv.pull(timeout_ms=1000)
        if req is None:
            continue
        srv.ack(req["dp_client_id"], req["task_id"], 0, None)
        n += 1
    srv.stop()
    done.set()


def client_proc(prefix, ready):
    ready.wait()
    pools = [(256*1024, 64), (4*1024*1024, 32), (16*1024*1024, 8)]
    cli = fastipc.Client.create(prefix, 0, pools)
    print("=== small (1K) ===", flush=True)
    for i in range(5):
        tid = cli.push_put(np.arange(1024, dtype=np.int64),
                           np.arange(1024, dtype=np.int64))
        r = cli.pull(timeout_ms=2000)
        assert r is not None and r["task_id"] == tid, f"mismatch: {r}"
    print("  small ok", flush=True)

    print("=== large (512K) x 10 ===", flush=True)
    big_ti = np.arange(512*1024, dtype=np.int64)
    big_sm = np.arange(512*1024, dtype=np.int64)
    for i in range(10):
        t0 = time.perf_counter()
        tid = cli.push_put(big_ti, big_sm)
        r = cli.pull(timeout_ms=5000)
        dt = (time.perf_counter() - t0) * 1e6
        assert r is not None and r["task_id"] == tid
        print(f"  iter {i} RTT={dt:.1f} us", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    ready = mp.Event()
    done = mp.Event()
    total = 5 + 10
    sp = mp.Process(target=server_proc, args=("smoke_xp", total, ready, done))
    sp.start()
    cp = mp.Process(target=client_proc, args=("smoke_xp", ready))
    cp.start()
    cp.join(timeout=30)
    sp.join(timeout=15)
    print("PY XPROC SMOKE: DONE")
