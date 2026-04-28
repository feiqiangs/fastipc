"""
FastIPC vs ZMQ: apples-to-apples comparison benchmark.

Uses the exact same workload as flexkv_zmq_benchmark.py:
  Scenario A — single-request round-trip latency over token_id lengths
  Scenario B — concurrent clients hammering a single server
  Scenario C — pure pickle overhead (ZMQ side only; FastIPC has no pickle)

Writes Markdown report to `fastipc_vs_zmq_report.md`.
"""

import os
import sys
import time
import pickle
import tempfile
import statistics
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import zmq

# Ensure the compiled fastipc module is importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_PY_DIR = os.path.abspath(os.path.join(_HERE, "..", "build", "python"))
if os.path.isdir(_BUILD_PY_DIR) and _BUILD_PY_DIR not in sys.path:
    sys.path.insert(0, _BUILD_PY_DIR)

import fastipc  # noqa: E402  (after sys.path tweak)


# =============================================================================
#  SECTION 1 — ZMQ baseline (same as flexkv_zmq_benchmark.py)
# =============================================================================

@dataclass
class ZRegisterRequest:
    dp_client_id: int
    client_recv_port: str

@dataclass
class ZPutRequest:
    dp_client_id: int
    token_ids: np.ndarray
    slot_mapping: np.ndarray
    token_mask: Optional[np.ndarray]
    task_id: int = -1
    namespace: Optional[List[str]] = None

@dataclass
class ZAckResponse:
    task_id: int
    ok: bool = True


def _zmq_socket(ctx, t, endpoint, bind):
    s = ctx.socket(t)
    buf = int(0.5 * 1024**3)
    if t == zmq.PUSH:
        s.setsockopt(zmq.SNDHWM, 0); s.setsockopt(zmq.SNDBUF, buf)
    if t == zmq.PULL:
        s.setsockopt(zmq.RCVHWM, 0); s.setsockopt(zmq.RCVBUF, buf)
    if bind: s.bind(endpoint)
    else:    s.connect(endpoint)
    return s


def _new_ipc():
    f = tempfile.NamedTemporaryFile(delete=True, prefix="fipc_bench_")
    name = f.name; f.close()
    return f"ipc://{name}"


def zmq_server_proc(server_ep, total_expected, ready_evt, done_evt, stats_q):
    ctx = zmq.Context(1)
    recv_sock = _zmq_socket(ctx, zmq.PULL, server_ep, True)
    client_socks = {}
    ready_evt.set()

    processed = 0; bytes_total = 0
    t_recv = 0.0; t_unp = 0.0; t_hdl = 0.0
    t_first = None

    while processed < total_expected:
        t0 = time.perf_counter()
        raw = recv_sock.recv()
        t1 = time.perf_counter()
        req = pickle.loads(raw)
        t2 = time.perf_counter()

        if isinstance(req, ZRegisterRequest):
            client_socks[req.dp_client_id] = _zmq_socket(ctx, zmq.PUSH, req.client_recv_port, False)
            continue

        if t_first is None: t_first = t0
        bytes_total += len(raw)
        t_recv += (t1 - t0); t_unp += (t2 - t1)

        sock = client_socks.get(req.dp_client_id)
        if sock is None: continue
        sock.send_pyobj(ZAckResponse(task_id=req.task_id))
        t3 = time.perf_counter()
        t_hdl += (t3 - t2)
        processed += 1

    t_last = time.perf_counter()
    stats_q.put({
        "processed": processed, "bytes": bytes_total,
        "duration_s": (t_last - t_first) if t_first else 0.0,
        "recv_total_s": t_recv, "unpickle_total_s": t_unp, "handle_total_s": t_hdl,
    })
    done_evt.set()
    time.sleep(0.1)


def zmq_client_proc(server_ep, num_reqs, token_len, client_id, barrier, result_q):
    ctx = zmq.Context(1)
    send_sock = _zmq_socket(ctx, zmq.PUSH, server_ep, False)
    client_ep = _new_ipc()
    recv_sock = _zmq_socket(ctx, zmq.PULL, client_ep, True)
    send_sock.send_pyobj(ZRegisterRequest(client_id, client_ep))

    rng = np.random.default_rng(seed=client_id)
    reqs = [
        ZPutRequest(client_id,
                    rng.integers(0, 1 << 20, size=token_len, dtype=np.int64),
                    np.arange(token_len, dtype=np.int64),
                    None,
                    client_id * 10_000_000 + i)
        for i in range(num_reqs)
    ]

    barrier.wait()
    t0 = time.perf_counter()
    # Mirror FastIPC client: bounded in-flight pushes so the comparison is
    # apples-to-apples on backpressure semantics.
    INFLIGHT = 8
    pushed = 0
    acked = 0
    while acked < num_reqs:
        while pushed < num_reqs and (pushed - acked) < INFLIGHT:
            send_sock.send_pyobj(reqs[pushed])
            pushed += 1
        recv_sock.recv_pyobj()
        acked += 1
    t1 = time.perf_counter()
    result_q.put({"client_id": client_id, "send_s": t1 - t0, "total_s": t1 - t0})


# =============================================================================
#  SECTION 2 — FastIPC runner
# =============================================================================

_FIPC_POOLS = [
    (  256 * 1024, 256),     # 256 KB x 256  (covers ≤32K int64 tokens)
    ( 4 * 1024 * 1024, 256), # 4 MB   x 256  (covers ≤512K int64 tokens)
    (16 * 1024 * 1024,  64), # 16 MB  x 64   (safety)
]


def fipc_server_proc(shm_prefix, max_clients, num_workers, total_expected,
                     ready_evt, done_evt, stats_q):
    # auto_ack=True: C++ workers handle ack directly, no Python in the hot path.
    # This models the case where a FlexKV-like system would later plug its own
    # C++ handler in place of this.
    srv = fastipc.Server.create(
        shm_prefix=shm_prefix,
        max_clients=max_clients,
        ring_capacity=1024,
        resp_capacity=1024,
        num_workers=num_workers,
        spin_iters=0,
        auto_ack=True,
        pools=_FIPC_POOLS,
    )
    srv.start()
    ready_evt.set()
    # Just wait for the client to signal done via the done_evt from outside.
    # We sleep in short increments so we can still be joined cleanly.
    while not done_evt.is_set():
        time.sleep(0.05)
    srv.stop()
    stats_q.put({"processed": total_expected, "duration_s": 0.0})


def fipc_client_proc(shm_prefix, num_reqs, token_len, client_id, barrier, result_q):
    cli = fastipc.Client.create(shm_prefix, client_id, _FIPC_POOLS)
    rng = np.random.default_rng(seed=client_id)
    # Pre-generate to exclude ndarray creation from timing.
    reqs_tids = [rng.integers(0, 1 << 20, size=token_len, dtype=np.int64) for _ in range(num_reqs)]
    reqs_slot = [np.arange(token_len, dtype=np.int64) for _ in range(num_reqs)]

    barrier.wait()
    t0 = time.perf_counter()
    # Push and pull interleaved so shm slots can recycle. Maintain a small
    # in-flight window (matches FlexKV's natural "submit-then-await" loop).
    INFLIGHT = 8
    pushed = 0
    acked = 0
    while acked < num_reqs:
        # Top up window.
        while pushed < num_reqs and (pushed - acked) < INFLIGHT:
            cli.push_put(reqs_tids[pushed], reqs_slot[pushed], None, -1, 0)
            pushed += 1
        # Drain at least one ack.
        r = cli.pull(timeout_ms=10000)
        if r is None:
            break
        acked += 1
    t1 = time.perf_counter()
    result_q.put({"client_id": client_id, "send_s": t1 - t0, "total_s": t1 - t0,
                  "acks": acked})


# =============================================================================
#  SECTION 3 — Scenarios
# =============================================================================

def scen_a_zmq(lengths, iters=30, warmup=3):
    print("\n=== ZMQ — Scenario A (single-req RTT) ===")
    out = []
    for L in lengths:
        server_ep = _new_ipc()
        client_ep = _new_ipc()
        total = iters + warmup
        ready = mp.Event(); done = mp.Event(); sq = mp.Queue()
        srv = mp.Process(target=zmq_server_proc, args=(server_ep, total, ready, done, sq))
        srv.start(); ready.wait()

        ctx = zmq.Context(1)
        send = _zmq_socket(ctx, zmq.PUSH, server_ep, False)
        recv = _zmq_socket(ctx, zmq.PULL, client_ep, True)
        send.send_pyobj(ZRegisterRequest(0, client_ep))

        rng = np.random.default_rng(42)
        reqs = [
            ZPutRequest(0,
                        rng.integers(0, 1 << 20, size=L, dtype=np.int64),
                        np.arange(L, dtype=np.int64), None, i)
            for i in range(total)
        ]
        payload = len(pickle.dumps(reqs[0]))
        for i in range(warmup):
            send.send_pyobj(reqs[i]); recv.recv_pyobj()
        rtts = []
        for i in range(warmup, total):
            t0 = time.perf_counter()
            send.send_pyobj(reqs[i])
            recv.recv_pyobj()
            t1 = time.perf_counter()
            rtts.append((t1 - t0) * 1e6)
        done.wait(5); sq.get(); srv.join(3)
        send.close(0); recv.close(0); ctx.term()

        r = {"tokens": L, "payload_kb": payload/1024,
             "mean_us": statistics.mean(rtts),
             "p50_us": statistics.median(rtts),
             "p95_us": float(np.percentile(rtts, 95)),
             "p99_us": float(np.percentile(rtts, 99)),
             "bw_mb_s": (payload/1024/1024)/(statistics.mean(rtts)/1e6)}
        out.append(r)
        print(f"  L={L:>7,}  mean={r['mean_us']:8.1f}µs  p99={r['p99_us']:8.1f}  bw={r['bw_mb_s']:.0f} MB/s")
    return out


def scen_a_fipc(lengths, iters=30, warmup=3):
    print("\n=== FastIPC — Scenario A (single-req RTT) ===")
    out = []
    for L in lengths:
        prefix = f"fipc_ba_{os.getpid()}_{L}"
        total = iters + warmup
        ready = mp.Event(); done = mp.Event(); sq = mp.Queue()
        srv = mp.Process(target=fipc_server_proc,
                         args=(prefix, 2, 1, total, ready, done, sq))
        srv.start(); ready.wait()

        cli = fastipc.Client.create(prefix, 0, _FIPC_POOLS)
        rng = np.random.default_rng(42)
        tids = [rng.integers(0, 1 << 20, size=L, dtype=np.int64) for _ in range(total)]
        smap = [np.arange(L, dtype=np.int64) for _ in range(total)]
        payload = L * 8 * 2

        for i in range(warmup):
            cli.push_put(tids[i], smap[i], None, -1, 0)
            cli.pull(timeout_ms=5000)
        rtts = []
        for i in range(warmup, total):
            t0 = time.perf_counter()
            cli.push_put(tids[i], smap[i], None, -1, 0)
            cli.pull(timeout_ms=5000)
            t1 = time.perf_counter()
            rtts.append((t1 - t0) * 1e6)
        done.set()  # tell server to stop
        sq.get(timeout=5); srv.join(timeout=5)

        r = {"tokens": L, "payload_kb": payload/1024,
             "mean_us": statistics.mean(rtts),
             "p50_us": statistics.median(rtts),
             "p95_us": float(np.percentile(rtts, 95)),
             "p99_us": float(np.percentile(rtts, 99)),
             "bw_mb_s": (payload/1024/1024)/(statistics.mean(rtts)/1e6)}
        out.append(r)
        print(f"  L={L:>7,}  mean={r['mean_us']:8.1f}µs  p99={r['p99_us']:8.1f}  bw={r['bw_mb_s']:.0f} MB/s")
    return out


def scen_b_zmq(combos):
    print("\n=== ZMQ — Scenario B (concurrent clients) ===")
    out = []
    for nc, rpc, L in combos:
        server_ep = _new_ipc()
        total = nc * rpc
        ready = mp.Event(); done = mp.Event(); sq = mp.Queue()
        srv = mp.Process(target=zmq_server_proc, args=(server_ep, total, ready, done, sq))
        srv.start(); ready.wait()

        barrier = mp.Barrier(nc + 1); resq = mp.Queue(); clients = []
        for cid in range(nc):
            p = mp.Process(target=zmq_client_proc,
                           args=(server_ep, rpc, L, cid, barrier, resq))
            p.start(); clients.append(p)
        time.sleep(0.3); barrier.wait()
        t0 = time.perf_counter()
        for p in clients: p.join()
        done.wait(30); srv_stats = sq.get(); srv.join(5)
        wall = time.perf_counter() - t0
        qps = total / wall
        bw = (srv_stats["bytes"]/1024/1024) / wall
        r = {"nc": nc, "rpc": rpc, "L": L, "total": total, "wall": wall,
             "qps": qps, "bw": bw,
             "srv_unpickle_ms": srv_stats["unpickle_total_s"]*1e3,
             "srv_handle_ms": srv_stats["handle_total_s"]*1e3}
        out.append(r)
        print(f"  nc={nc:>2} rpc={rpc:>4} L={L:>6,}  wall={wall:5.2f}s  qps={qps:7.0f}  bw={bw:6.0f} MB/s")
    return out


def scen_b_fipc(combos):
    print("\n=== FastIPC — Scenario B (concurrent clients) ===")
    out = []
    for nc, rpc, L in combos:
        prefix = f"fipc_bb_{os.getpid()}_{nc}_{L}_{int(time.time())}"
        total = nc * rpc
        ready = mp.Event(); done = mp.Event(); sq = mp.Queue()
        workers = min(max(2, nc * 2), 16)
        srv = mp.Process(target=fipc_server_proc,
                         args=(prefix, nc, workers, total, ready, done, sq))
        srv.start(); ready.wait()

        barrier = mp.Barrier(nc + 1); resq = mp.Queue(); clients = []
        for cid in range(nc):
            p = mp.Process(target=fipc_client_proc,
                           args=(prefix, rpc, L, cid, barrier, resq))
            p.start(); clients.append(p)
        time.sleep(0.3); barrier.wait()
        t0 = time.perf_counter()
        for p in clients: p.join()
        done.set()   # tell server to stop
        sq.get(timeout=10); srv.join(timeout=10)
        wall = time.perf_counter() - t0
        qps = total / wall
        payload_bytes = total * L * 8 * 2
        bw = (payload_bytes/1024/1024) / wall
        r = {"nc": nc, "rpc": rpc, "L": L, "total": total, "wall": wall,
             "qps": qps, "bw": bw}
        out.append(r)
        print(f"  nc={nc:>2} rpc={rpc:>4} L={L:>6,}  wall={wall:5.2f}s  qps={qps:7.0f}  bw={bw:6.0f} MB/s")
    return out


def scen_c_pickle(lengths, iters=50):
    print("\n=== Scenario C — Pure pickle overhead (ZMQ only) ===")
    out = []
    rng = np.random.default_rng(0)
    for L in lengths:
        t = rng.integers(0, 1 << 20, size=L, dtype=np.int64)
        s = np.arange(L, dtype=np.int64)
        req = ZPutRequest(0, t, s, None, 123)
        for _ in range(5): pickle.loads(pickle.dumps(req))
        d_list, l_list = [], []
        for _ in range(iters):
            a = time.perf_counter()
            b = pickle.dumps(req)
            c = time.perf_counter()
            pickle.loads(b)
            d = time.perf_counter()
            d_list.append((c-a)*1e6); l_list.append((d-c)*1e6)
        pay = len(pickle.dumps(req))
        r = {"tokens": L, "payload_kb": pay/1024,
             "dump_us": statistics.mean(d_list),
             "load_us": statistics.mean(l_list),
             "total_us": statistics.mean(d_list)+statistics.mean(l_list)}
        out.append(r)
        print(f"  L={L:>7,}  dump={r['dump_us']:7.1f}µs  load={r['load_us']:7.1f}µs")
    return out


# =============================================================================
#  Report writer
# =============================================================================

def write_report(path, sys_info, Az, Af, Bz, Bf, C):
    lines = []
    lines.append("# FastIPC vs ZMQ — 对比 Benchmark 报告\n")
    lines.append(f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 主机: {sys_info['hostname']} | CPU: {sys_info['cpu_count']} | Python: {sys_info['python']}")
    lines.append(f"- numpy: {sys_info['numpy']} | pyzmq: {sys_info['pyzmq']}")
    lines.append("")
    lines.append("## Scenario A — 单请求端到端 RTT (µs)\n")
    lines.append("| token_ids | ZMQ mean | ZMQ p99 | FastIPC mean | FastIPC p99 | **加速比** |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for rz, rf in zip(Az, Af):
        speedup = rz["mean_us"] / rf["mean_us"] if rf["mean_us"] > 0 else float("inf")
        lines.append(f"| {rz['tokens']:,} | {rz['mean_us']:.1f} | {rz['p99_us']:.1f} | "
                     f"{rf['mean_us']:.1f} | {rf['p99_us']:.1f} | **{speedup:.1f}x** |")
    lines.append("")

    lines.append("## Scenario A — 有效带宽 (MB/s)\n")
    lines.append("| token_ids | ZMQ bw | FastIPC bw | 提升 |")
    lines.append("|---:|---:|---:|---:|")
    for rz, rf in zip(Az, Af):
        up = rf["bw_mb_s"] / rz["bw_mb_s"] if rz["bw_mb_s"] > 0 else 0.0
        lines.append(f"| {rz['tokens']:,} | {rz['bw_mb_s']:.0f} | {rf['bw_mb_s']:.0f} | {up:.1f}x |")
    lines.append("")

    lines.append("## Scenario B — 并发 QPS\n")
    lines.append("| 客户端 | req/client | token_len | ZMQ QPS | FastIPC QPS | **加速比** |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for rz, rf in zip(Bz, Bf):
        up = rf["qps"] / rz["qps"] if rz["qps"] > 0 else 0.0
        lines.append(f"| {rz['nc']} | {rz['rpc']} | {rz['L']:,} | "
                     f"{rz['qps']:.0f} | {rf['qps']:.0f} | **{up:.1f}x** |")
    lines.append("")

    lines.append("## Scenario C — 纯 pickle 开销 (ZMQ 独有)\n")
    lines.append("| token_ids | dump µs | load µs | 合计 µs | payload KB |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in C:
        lines.append(f"| {r['tokens']:,} | {r['dump_us']:.1f} | {r['load_us']:.1f} | "
                     f"{r['total_us']:.1f} | {r['payload_kb']:.1f} |")
    lines.append("")

    # Summary
    lines.append("## 小结\n")
    big = next((i for i, r in enumerate(Az) if r["tokens"] >= 500000), None)
    if big is not None:
        lines.append(f"- **大请求 (≥512K tokens) RTT**: ZMQ {Az[big]['mean_us']/1000:.2f} ms → "
                     f"FastIPC {Af[big]['mean_us']/1000:.2f} ms "
                     f"(**{Az[big]['mean_us']/Af[big]['mean_us']:.1f}x 加速**)")
    small = next((i for i, r in enumerate(Az) if r["tokens"] <= 1024), None)
    if small is not None:
        lines.append(f"- **小请求 (1K tokens) RTT**: ZMQ {Az[small]['mean_us']:.1f} µs → "
                     f"FastIPC {Af[small]['mean_us']:.1f} µs "
                     f"(**{Az[small]['mean_us']/Af[small]['mean_us']:.1f}x 加速**)")
    if Bz and Bf:
        best_z = max(Bz, key=lambda r: r["qps"])
        best_f = max(Bf, key=lambda r: r["qps"])
        lines.append(f"- **并发峰值 QPS**: ZMQ {best_z['qps']:.0f} "
                     f"(nc={best_z['nc']}, L={best_z['L']:,}) "
                     f"→ FastIPC {best_f['qps']:.0f} "
                     f"(nc={best_f['nc']}, L={best_f['L']:,}) "
                     f"= **{best_f['qps']/best_z['qps']:.1f}x**")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
#  Main
# =============================================================================

def main():
    import socket as _socket
    sys_info = {
        "hostname": _socket.gethostname(),
        "cpu_count": mp.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pyzmq": zmq.__version__,
    }
    print(f"[sys] {sys_info}")

    lengths = [1024, 4096, 16384, 65536, 262144, 524288]
    combos = [
        (1,  200, 1024),
        (4,  200, 1024),
        (8,  200, 1024),
        (16, 100, 1024),
        (1,  100, 16384),
        (4,  100, 16384),
        (8,   50, 16384),
        (4,   20, 262144),
        (8,   10, 524288),
    ]

    Az = scen_a_zmq(lengths)
    Af = scen_a_fipc(lengths)
    Bz = scen_b_zmq(combos)
    Bf = scen_b_fipc(combos)
    C  = scen_c_pickle(lengths)

    report = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fastipc_vs_zmq_report.md")
    write_report(report, sys_info, Az, Af, Bz, Bf, C)
    print(f"\nReport: {report}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
