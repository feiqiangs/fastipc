"""并发 smoke：8 clients 并发 + 1 server with N python workers."""
import sys, os, time, threading
_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_PY = os.path.normpath(os.path.join(_HERE, "..", "build", "python"))
sys.path.insert(0, _BUILD_PY)

import numpy as np
import multiprocessing as mp
import fastipc

POOLS = [(256*1024, 256), (4*1024*1024, 256), (16*1024*1024, 64)]


def server_proc(prefix, max_clients, total, ready, done):
    srv = fastipc.Server.create(shm_prefix=prefix, max_clients=max_clients,
                                num_workers=8, ring_capacity=1024,
                                resp_capacity=1024, pools=POOLS)
    srv.start()
    ready.set()

    counter = [0]
    per_client = [0] * max_clients
    lk = threading.Lock()

    def consume(idx):
        while True:
            with lk:
                if counter[0] >= total: return
            req = srv.pull(timeout_ms=500)
            if req is None: continue
            cid = req["dp_client_id"]
            srv.ack(cid, req["task_id"], 0, None)
            with lk:
                counter[0] += 1
                per_client[cid] += 1

    threads = [threading.Thread(target=consume, args=(i,)) for i in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"[server] processed total={counter[0]}, per_client={per_client}", flush=True)
    srv.stop()
    done.set()


def client_proc(prefix, cid, num_reqs, token_len, ready, result_q):
    ready.wait()
    cli = fastipc.Client.create(prefix, cid, POOLS)
    rng = np.random.default_rng(cid)
    reqs = [(rng.integers(0, 1<<20, size=token_len, dtype=np.int64),
             np.arange(token_len, dtype=np.int64)) for _ in range(num_reqs)]

    INFLIGHT = 8
    pushed = acked = 0
    t0 = time.perf_counter()
    while acked < num_reqs:
        while pushed < num_reqs and (pushed - acked) < INFLIGHT:
            cli.push_put(reqs[pushed][0], reqs[pushed][1], None, -1, 0)
            pushed += 1
        r = cli.pull(timeout_ms=10000)
        if r is None:
            print(f"  client {cid} pull TIMEOUT at acked={acked}/{num_reqs} pushed={pushed}", flush=True)
            break
        acked += 1
    t1 = time.perf_counter()
    result_q.put((cid, acked, t1 - t0))


if __name__ == "__main__":
    mp.set_start_method("spawn")
    NC = 8
    RPC = 100
    LEN = 1024
    total = NC * RPC

    ready = mp.Event(); done = mp.Event(); rq = mp.Queue()
    sp = mp.Process(target=server_proc, args=("smoke_cc", NC, total, ready, done))
    sp.start()

    clients = [mp.Process(target=client_proc, args=("smoke_cc", c, RPC, LEN, ready, rq)) for c in range(NC)]
    for c in clients: c.start()

    t0 = time.perf_counter()
    for c in clients: c.join(timeout=30)
    t_end = time.perf_counter()

    print(f"[main] wall={t_end-t0:.2f}s")
    while not rq.empty():
        cid, acked, dt = rq.get()
        print(f"  client {cid}: acked={acked}/{RPC}  dt={dt:.2f}s")
    sp.join(timeout=10)
    print(f"QPS = {total/(t_end-t0):.0f}")
    print("CONCURRENT SMOKE: DONE")
