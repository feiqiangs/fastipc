# FastIPC

Zero-copy, lock-free IPC for Linux, designed as a drop-in replacement for
FlexKV's ZMQ+pickle client-server control plane.

## Design highlights

- **POSIX shm slab pools** for `token_ids` / `slot_mapping` / `token_mask`
- **Fixed-size POD messages** (104 B) — no pickle, no serialization
- **Per-client SPSC rings** in shm, multi-worker CAS-on-tail for MPMC consumption
- **epoll_wait + eventfd edge-triggered wakeup** — no sleep, no busy-loop
- **C++17 core, pthread workers** — no Python GIL contention on the hot path
- **pybind11 bindings** expose `push_put(ndarray)` / `pull() -> dict`

See `../fastipc_design.md` for the full design document.

## Build

```bash
pip install pybind11 numpy
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
# Outputs:
#   build/libfastipc_core.a
#   build/fastipc_selftest             # C++ smoke test
#   build/python/fastipc/_fastipc.so   # Python extension
```

## Quick test

```bash
# C++ smoke
./build/fastipc_selftest

# Python smoke
cd build/python && python3 -c "
import fastipc, numpy as np, threading
srv = fastipc.Server.create('demo', max_clients=1, num_workers=1,
                             pools=[(256*1024, 32), (16*1024*1024, 4)])
srv.start()

def consume():
    while True:
        req = srv.pull(timeout_ms=1000)
        if req is None: break
        srv.ack(req['dp_client_id'], req['task_id'], 0, None)

threading.Thread(target=consume, daemon=True).start()

cli = fastipc.Client.create('demo', 0, [(256*1024, 32), (16*1024*1024, 4)])
tid = cli.push_put(np.arange(1000, dtype=np.int64),
                   np.arange(1000, dtype=np.int64))
print('pushed', tid, '->', cli.pull(timeout_ms=2000))
srv.stop()
"
```

## Benchmark vs ZMQ

```bash
python3 bench/bench_compare.py
# Writes fastipc_vs_zmq_report.md
```
