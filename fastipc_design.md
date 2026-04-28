# FastIPC：FlexKV Client-Server 零拷贝通信层 设计文档

> 版本：**v0.2**（锁定实现方向）
> 日期：2026-04-27
> 作者：phaedonsun
> 目标：为 FlexKV `server_client_mode` 下的控制面通信提供一套 **C++ 核 + Python 绑定** 的替代实现，基于 **POSIX shm + SPSC ring + epoll/eventfd + pthread worker**，在 Linux 平台跑出可量化收益后再评估合入 FlexKV。

---

## 1. 背景与动机

### 1.1 基线实测瓶颈（来自 `flexkv_zmq_benchmark_report.md`）

对齐 FlexKV 现有 ZMQ PUSH/PULL + pickle 方案，在本机测得：

| 指标 | 实测值 | 观察 |
| :--- | ---: | :--- |
| 1K tokens 单请求 RTT | 117 µs | 小请求基线 |
| **512K tokens 单请求 RTT** | **16.7 ms** | payload 8 MB，有效带宽 ~480 MB/s |
| **单线程 Server 峰值 QPS** | **9,349** | clients≥16 后 QPS 反降，单线程 PULL 饱和 |
| pickle 开销 (512K) | 1.4 ms (~9% RTT) | 不是主角 |

### 1.2 根因拆解

```
client 用户态                内核                 server 用户态（单线程）
─────────────                ────                  ──────────────────
PutRequest(token_ids)
   │
   │ pickle.dumps        ── [memcpy #1] ndarray → pickle bytes
   │
   │ zmq.send → UDS      ── [memcpy #2] user → kernel sk_buff
   │                     ── [memcpy #3] kernel → user
   │                                          │
   │                                          ▼ zmq.recv
   │                                          │ pickle.loads  ── [memcpy #4] bytes → ndarray
   │                                          │ handler(req)  ── 串行
```

**三个可独立攻击的开销**：
1. **4 次 memcpy**
2. **pickle 序列化/反序列化的 CPU + 对象重建**
3. **Server 主循环单线程串行**（`server.py:286-320`）

### 1.3 目标（量化）

| 目标项 | 基线 | 目标 |
| :--- | ---: | ---: |
| 1K tokens RTT | 117 µs | **< 10 µs** |
| 512K tokens RTT | 16.7 ms | **< 1.5 ms** |
| 多 client 并发 QPS (1K, 8 clients) | 9,349 | **> 50,000** |
| 端到端 memcpy 次数 | 4 | **1** |

---

## 2. 实现方向决策（v0.2 锁定）

| 维度 | 决策 | 理由 |
| :--- | :--- | :--- |
| **平台支持** | **只支持 Linux**（先做对） | POSIX shm / eventfd / epoll / futex 都是 Linux 原生，跨平台会严重分散精力 |
| **实现语言** | **核心 C++ (C++17)，Python 通过 pybind11 绑定** | 跳开 Python GIL；原子/内存序/cache-line 对齐在 C++ 里天然；方便未来直接融进 FlexKV 的 C++ 组件（参考 FlexKV 已有 `csrc/` 目录） |
| **Server Worker** | **pthread** (C++)，**完全不在 Python 侧跑** | GIL 会彻底毁掉 MPMC 并行；worker 全程只碰 shm + POD，不调 Python |
| **唤醒机制** | **epoll_wait + eventfd**（不用 sleep，也不用纯 busy-loop） | 低延迟 + 零 CPU 浪费；eventfd 是 Linux 原生的进程间事件信号；epoll 支持多 fd 多路复用（多 ring 对应多 eventfd） |
| **Python 绑定** | **pybind11**，接口 `push(np.ndarray...) / pull() -> np.ndarray`，**只处理 ndarray** | 对 Python 调用方（FlexKV）最友好；ndarray 内部 buffer 直接 memcpy 进 shm，无 pickle、无对象重建 |
| **消息格式** | **固定 96 B POD `PutRequestPOD`**（见 §4） | 替代 pickle，零序列化开销 |
| **Payload 存储** | **POSIX shm + 定长 slab allocator** | 避开变长堆碎片；配合引用计数做安全回收 |
| **队列形态** | **每 client 一个 SPSC ring** + **多 worker MPMC 消费** (fetch-and-add 抢占) | SPSC 原语简单、正确性容易证明；多 ring 被多 worker 消费天然是 MPMC |

---

## 3. 系统架构

```
┌─────────────────────────── Python 世界（FlexKV） ──────────────────────────┐
│                                                                            │
│  KVDPClient 0         KVDPClient 1        ...       KVDPClient N           │
│    │                      │                             │                  │
│    │ fastipc.Client       │                             │                  │
│    │  .push_put(...)      │                             │                  │
│    │  .pull_response()    │                             │                  │
│    ▼                      ▼                             ▼                  │
└────┼──────────────────────┼─────────────────────────────┼──────────────────┘
     │ pybind11 调用        │                             │
     │ (GIL 释放，ndarray zero-copy view)                                    │
     ▼                      ▼                             ▼
┌────────────────────────── C++ 核 (libfastipc.so) ───────────────────────────┐
│                                                                             │
│   ┌──────────────────────── Producer 侧 ────────────────────────────┐       │
│   │                                                                   │       │
│   │  Client 0                Client 1        ...       Client N       │       │
│   │    │                        │                         │           │       │
│   │    │  [1] slot = shm_pool.alloc()                                  │       │
│   │    │  [2] memcpy(slot.data, token_ids.buf)  ← 唯一的用户态拷贝       │       │
│   │    │  [3] req_pod.tids = {slot.id, ...}                            │       │
│   │    │  [4] ring[c].push(req_pod)   ← 96B，无锁 release store          │       │
│   │    │  [5] eventfd_write(ring[c].ev_fd, 1)                          │       │
│   │    ▼                        ▼                         ▼           │       │
│   │  ┌─────┐                 ┌─────┐                 ┌─────┐            │       │
│   │  │Ring │ eventfd#0       │Ring │ eventfd#1       │Ring │ eventfd#N    │       │
│   │  │ 0   │───┐             │ 1   │───┐             │ N   │───┐          │       │
│   │  └─────┘   │             └─────┘   │             └─────┘   │          │       │
│   │            ▼                       ▼                       ▼          │       │
│   └────────────┼───────────────────────┼───────────────────────┼──────────┘       │
│                │                       │                       │                  │
│                └───────┬───────────────┴─────────────┬─────────┘                  │
│                        ▼                             ▼                            │
│                ┌─────────── epoll_fd ──────────────┐                               │
│                │  (一个 epoll 监听所有 ring 的 eventfd)│                             │
│                └──────────────────────────────────┘                               │
│                        │                                                          │
│             worker 阻塞在 epoll_wait(timeout=-1)                                  │
│                        │                                                          │
│   ┌────────────── Server Worker Pool (pthread) ──────────────┐                    │
│   │  Worker 0       Worker 1       ...    Worker M            │                    │
│   │    │               │                     │                │                    │
│   │    │  [1] epoll_wait 唤醒                                  │                    │
│   │    │  [2] 遍历 ready ring，fetch_and_add 抢 slot           │                    │
│   │    │  [3] pop req_pod (无拷贝，直接 from ring slot)         │                    │
│   │    │  [4] 拿 shm slot 指针，回调 handler(shm_ptr, len, ...)  │                    │
│   │    │  [5] 填 response_pod 到 response_ring[c]               │                    │
│   │    │  [6] eventfd_write(resp_ring[c].ev_fd, 1)              │                    │
│   │    ▼               ▼                     ▼                │                    │
│   │                                                          │                    │
│   └────────────────────────────────────────────────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**设计要点**：
- Python 线程调用 `push_put()` 时，pybind11 函数里 **主动 `py::gil_scoped_release`**，C++ 侧 memcpy + ring push + eventfd_write 全程不持 GIL
- Server 的 worker 是 **pure C++ pthread**，从创建到退出完全不接触 Python 解释器 → 没有 GIL 争用
- 回调 handler 有两种形态（可选其一）：
  - **v0.1**：handler 由 C++ 实现（例如直接转发到 FlexKV 的 `kv_task_engine`，如果后续合入）
  - **v0.1 Python 兼容形态**：Python 侧 consumer 线程调 `pull()` 拿 POD + shm view（`py::array` zero-copy），此时**只有 Python 侧消费时需要 GIL**，但 C++ worker 还是能并行地把请求从 ring 搬到 Python 可见的 "deliver queue"

v0.1 为了先跑通，**走 Python 兼容形态**——C++ 做无锁搬运和聚合，Python 侧 `pull()` 返回零拷贝 numpy view。这样既不改 FlexKV 上层逻辑，又能测出通信层收益。

---

## 4. POD 消息格式

```cpp
// 所有字段 little-endian，#pragma pack(1)
struct PutRequestPOD {                     // total 96 B
    uint32_t magic;             //  0  0x46415354 "FAST"
    uint16_t req_type;          //  4  0=Put 1=Get 2=PutMatch ...
    uint16_t flags;             //  6  bit0=has_mask, bit1=as_batch
    uint32_t dp_client_id;      //  8
    uint32_t _pad0;             // 12

    int64_t  task_id;           // 16
    int32_t  layer_granularity; // 24
    int32_t  namespace_id;      // 28  (预注册 namespace → id)

    // token_ids 位置
    uint32_t tids_pool_id;      // 32
    uint32_t tids_slot_id;      // 36
    uint32_t tids_offset;       // 40  (slot 内偏移，字节)
    uint32_t tids_nbytes;       // 44
    uint16_t tids_dtype;        // 48  (numpy dtype code)
    uint16_t tids_ndim;         // 50
    uint32_t tids_shape0;       // 52

    // slot_mapping 位置
    uint32_t smap_pool_id;      // 56
    uint32_t smap_slot_id;      // 60
    uint32_t smap_offset;       // 64
    uint32_t smap_nbytes;       // 68
    uint16_t smap_dtype;        // 72
    uint16_t _pad1;             // 74

    // token_mask 位置 (可选，flags bit0)
    uint32_t mask_pool_id;      // 76
    uint32_t mask_slot_id;      // 80
    uint32_t mask_offset;       // 84
    uint32_t mask_nbytes;       // 88
    uint16_t mask_dtype;        // 92
    uint16_t _pad2;             // 94
};
static_assert(sizeof(PutRequestPOD) == 96);

struct ResponsePOD {                       // total 64 B
    uint32_t magic;
    uint16_t resp_type;
    uint16_t status;
    uint32_t dp_client_id;
    uint32_t _pad;
    int64_t  task_id;
    // 可选 mask slot
    uint32_t mask_pool_id;
    uint32_t mask_slot_id;
    uint32_t mask_offset;
    uint32_t mask_nbytes;
    uint64_t _rsvd[3];
};
```

**与 ZMQ+pickle 的差异**：

| 项 | ZMQ+pickle | FastIPC |
| :--- | :--- | :--- |
| 消息大小 | 16 KB ~ 8 MB 随 payload 变 | **固定 96 B** |
| 序列化 | `pickle.dumps` | **0**（结构体直接写 ring slot） |
| 反序列化 | `pickle.loads` | **0**（`reinterpret_cast<PutRequestPOD*>(slot)`） |

---

## 5. 共享内存 Slab Pool

```cpp
// 在共享内存里放一个池控制块 + 若干定长 slot
struct ShmPoolHeader {
    uint64_t magic;           // 版本/校验
    uint32_t slot_size;       // 每个 slot 字节数（如 512 KB）
    uint32_t num_slots;       // 总 slot 数
    uint32_t pool_id;         // 用于多池共存
    uint32_t _pad;
    // free-list 头（lock-free stack, 2-word CAS 避免 ABA）
    std::atomic<uint64_t> free_head;  // 高 32 位 = aba_counter, 低 32 位 = slot_id
    // 每个 slot 的引用计数
    std::atomic<int32_t> refcount[/* num_slots */];
    // 后面紧跟 slot 数据区，slot i 起点 = &header + data_offset + i*slot_size
};

class ShmPool {
public:
    // 打开（或创建）一个命名共享内存池
    static std::shared_ptr<ShmPool> create_or_open(
        const std::string& name, uint32_t slot_size, uint32_t num_slots);

    // 分配一个 slot（lock-free pop）；失败返回 -1
    int32_t alloc_slot();

    // 归还一个 slot（lock-free push）
    void free_slot(int32_t slot_id);

    // 拿到 slot 数据起点（已经映射到当前进程地址空间）
    void* slot_ptr(int32_t slot_id);

    uint32_t slot_size() const;

    // 引用计数控制（多 consumer 场景）
    void ref(int32_t slot_id);
    void unref(int32_t slot_id);  // 降到 0 时自动 free_slot
};
```

**多池策略**：
- Pool #0：**小 slot**，512 KB，覆盖 ≤ 64K int64 tokens（~95% 推理请求）
- Pool #1：**中 slot**，4 MB，覆盖 ≤ 512K tokens
- Pool #2：**大 slot**，16 MB，兜底超长 prompt

alloc_slot 时根据 `nbytes` 选最小适配的池。

**生命周期与引用计数**：
- Client alloc slot → refcount=1
- Push 到 ring 后 client 不再碰数据 → 不变
- Server worker pop 出来，ref+1（refcount=2）
- Handler 用完 ndarray view → unref（refcount=1）
- Server 发 ACK 后 unref（refcount=0 → 自动 free）

Client 只要 alloc 完 push 就不再碰 slot，**生命周期完全交给 Server**。

---

## 6. SPSC Ring（每 client 一个）

```cpp
// 放在共享内存里。每个 ring 独占若干 cache line。
template <typename T, size_t N>  // N 必须是 2 的幂
struct SPSCRing {
    static_assert((N & (N-1)) == 0, "N must be power of 2");

    // Producer 侧（单写者）
    alignas(64) std::atomic<uint64_t> head{0};
    char _pad0[64 - sizeof(std::atomic<uint64_t>)];

    // Consumer 侧（注意：虽然全局是 MPMC，但从单个 ring 的视角只有一个 worker 通过
    // fetch_and_add 拿走 slot，所以对 ring 自身仍是 SPSC 语义）
    alignas(64) std::atomic<uint64_t> tail{0};
    char _pad1[64 - sizeof(std::atomic<uint64_t>)];

    // 唤醒用
    int ev_fd;   // eventfd，初始化时创建并 mmap 到 ring 头
    char _pad2[60];

    T slots[N];
};

// 典型 N = 1024；T = PutRequestPOD（96B）
// 则 ring 大小 ≈ 96 KB，放 shm 里很便宜
```

**push (client 侧)**：
```cpp
bool push(const PutRequestPOD& req) {
    uint64_t h = head.load(std::memory_order_relaxed);
    uint64_t t = tail.load(std::memory_order_acquire);
    if (h - t >= N) return false;              // full
    slots[h & (N-1)] = req;                     // 写入
    head.store(h + 1, std::memory_order_release);
    // 边沿唤醒：只有从空变非空时才写 eventfd，减少系统调用
    if (h == t) {
        uint64_t one = 1;
        ::write(ev_fd, &one, 8);
    }
    return true;
}
```

**pop (worker 侧，带 MPMC 抢占)**：
```cpp
bool try_pop(PutRequestPOD& out) {
    // 用 fetch_and_add 在 tail 上抢占，保证多 worker 安全
    uint64_t h = head.load(std::memory_order_acquire);
    for (;;) {
        uint64_t t = tail.load(std::memory_order_relaxed);
        if (t >= h) return false;               // empty
        // 尝试把 tail 抢占到 t+1
        if (tail.compare_exchange_weak(t, t + 1,
                std::memory_order_acq_rel)) {
            out = slots[t & (N-1)];
            return true;
        }
    }
}
```

> ⚠️ 因为多 worker 抢同一个 ring，严格来说这是 **SP-MC**（单生产多消费）。CAS on tail 在 worker 间是有争用的。如果发现争用严重，退化成 **每个 ring 只绑一个 worker**（纯 SPSC），然后 worker 总数不少于 client 数。v0.1 先走 SP-MC，benchmark 看争用情况再定。

---

## 7. 唤醒：epoll + eventfd（核心机制）

**原则**：
- 不用 sleep，不用 busy-loop
- 用 `epoll_wait(timeout=-1)` 阻塞等事件
- 每个 request ring 绑一个 eventfd，producer push 后**仅在从空变非空**时 `eventfd_write(1)`
- 可选：短 spin（比如 100 次）再 fallback 到 epoll_wait，极致降低单请求 RTT（已测）

### 7.1 Server worker 事件循环

```cpp
void worker_main(WorkerCtx* ctx) {
    int epfd = ctx->epoll_fd;
    epoll_event events[64];

    while (!ctx->stop.load(std::memory_order_relaxed)) {
        // 先做短 spin，低延迟取胜（可配置，默认 0）
        int popped = 0;
        if (ctx->spin_iters > 0) {
            for (int i = 0; i < ctx->spin_iters; ++i) {
                if (try_drain_all_rings(ctx) > 0) { popped = 1; break; }
                cpu_relax();  // __builtin_ia32_pause()
            }
        }
        if (popped) continue;

        // 阻塞在 epoll，零 CPU 占用
        int n = epoll_wait(epfd, events, 64, -1);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("epoll_wait"); break;
        }

        for (int i = 0; i < n; ++i) {
            int fd = events[i].data.fd;
            uint64_t cnt;
            ::read(fd, &cnt, 8);              // 消费 eventfd 计数
            // 找到对应的 ring（fd→ring_idx 映射）并尽量多 pop
            drain_ring(ctx->fd_to_ring[fd], ctx);
        }
    }
}
```

### 7.2 Client 拉响应

对称设计：每个 client 有自己的 response ring + eventfd。
```cpp
ResponsePOD FastIPCClient::pull(int timeout_ms) {
    // 先无阻塞尝试
    if (response_ring->try_pop(out)) return out;
    // 阻塞等
    epoll_event ev;
    int n = epoll_wait(my_epfd, &ev, 1, timeout_ms);
    if (n <= 0) throw timeout;
    uint64_t cnt; ::read(response_evfd, &cnt, 8);
    response_ring->try_pop(out);
    return out;
}
```

---

## 8. Python 绑定（pybind11）

**设计原则**：接口最小化，**输入输出全是 `numpy.ndarray` 或基本类型**，无需 Python 调用者理解 POD/shm/ring。

### 8.1 Python 可见的 API

```python
import numpy as np
import fastipc

# ─── Server 侧 ───
srv = fastipc.Server(
    shm_name="flexkv-ctrl",
    max_clients=16,
    ring_size=1024,
    num_workers=4,
    spin_iters=0,
)
srv.start()
# handler 有两种方式：
# (a) 纯 C++ 回调（需要编译时注入，最快）
# (b) Python 拉模式：worker 搬到 delivery queue，Python 主动 pull
while True:
    req = srv.pull(timeout_ms=1000)   # 返回 dict 或 None
    # req = {
    #   "task_id": 123,
    #   "dp_client_id": 0,
    #   "req_type": "put",
    #   "token_ids":    np.ndarray (zero-copy view into shm),
    #   "slot_mapping": np.ndarray (zero-copy view into shm),
    #   "token_mask":   np.ndarray or None,
    #   "_slot_handles": <opaque, 用于 server 回收 slot>,
    # }
    # ... 处理完后 ack ...
    srv.ack(req, status=0, mask=None_or_np_ndarray)

# ─── Client 侧 ───
cli = fastipc.Client(shm_name="flexkv-ctrl", dp_client_id=0)

task_id = cli.push_put(
    token_ids=np.array([...], dtype=np.int64),
    slot_mapping=np.array([...], dtype=np.int64),
    token_mask=None,
    layer_granularity=-1,
    namespace_id=0,
)

resp = cli.pull(timeout_ms=1000)
# resp = {"task_id": 123, "status": 0, "mask": None}
```

### 8.2 pybind11 层的关键实现细节

```cpp
// push_put 热路径示意
int64_t Client::push_put(py::array token_ids,
                         py::array slot_mapping,
                         py::object token_mask,
                         int32_t layer_granularity,
                         int32_t namespace_id) {
    // 拿到 numpy buffer 的底层指针（不持 GIL）
    py::buffer_info ti = token_ids.request();
    py::buffer_info sm = slot_mapping.request();

    // 1. 申请 shm slot
    int32_t tid_slot = pool_->alloc_slot();
    int32_t sm_slot  = pool_->alloc_slot();
    if (tid_slot < 0 || sm_slot < 0) throw std::runtime_error("shm full");

    {
        py::gil_scoped_release nogil;  // ★ 释放 GIL
        // 2. memcpy 入 shm（唯一的用户态拷贝）
        std::memcpy(pool_->slot_ptr(tid_slot), ti.ptr, ti.size * ti.itemsize);
        std::memcpy(pool_->slot_ptr(sm_slot),  sm.ptr, sm.size * sm.itemsize);

        // 3. 构造 POD 直接写 ring
        PutRequestPOD req{};
        req.magic = FAST_MAGIC;
        req.req_type = REQ_PUT;
        req.dp_client_id = client_id_;
        req.task_id = next_task_id();
        // ... 填 slot 位置 ...
        while (!ring_->push(req)) cpu_relax();   // 或用 epoll 反压

        // 4. 边沿唤醒
        if (ring_->was_empty_before_push()) {
            uint64_t one = 1;
            ::write(ring_->ev_fd, &one, 8);
        }
    }
    return req.task_id;
}
```

```cpp
// server.pull 返回 zero-copy numpy view
py::object Server::pull(int timeout_ms) {
    PutRequestPOD req;
    if (!delivery_queue_.pop_wait(req, timeout_ms)) return py::none();

    auto mk_view = [&](uint32_t pool_id, uint32_t slot_id,
                        uint32_t nbytes, uint16_t dtype, uint16_t ndim) {
        void* ptr = pools_[pool_id]->slot_ptr(slot_id);
        // 用 capsule 把 slot 的生命周期挂到 ndarray 上
        auto capsule = py::capsule(ptr,
            [slot_id, pool_id, this](void*){
                pools_[pool_id]->unref(slot_id);
            });
        return py::array(py::dtype::of_code(dtype),
                         {nbytes / dtype_size(dtype)},
                         {dtype_size(dtype)},
                         ptr,
                         capsule);  // ★ zero-copy，capsule 负责释放
    };

    py::dict d;
    d["task_id"] = req.task_id;
    d["dp_client_id"] = req.dp_client_id;
    d["token_ids"]    = mk_view(req.tids_pool_id, req.tids_slot_id, req.tids_nbytes,
                                 req.tids_dtype, req.tids_ndim);
    d["slot_mapping"] = mk_view(req.smap_pool_id, req.smap_slot_id, req.smap_nbytes,
                                 req.smap_dtype, 1);
    if (req.flags & HAS_MASK) {
        d["token_mask"] = mk_view(req.mask_pool_id, req.mask_slot_id, req.mask_nbytes,
                                  req.mask_dtype, 1);
    } else {
        d["token_mask"] = py::none();
    }
    return d;
}
```

**关键点**：
- `push` 侧：Python 把 ndarray 递给 C++，C++ 立刻 `gil_scoped_release`，剩下的 memcpy + ring push 全是 no-GIL 的 C++ 操作
- `pull` 侧：C++ 构造 `py::array`，data pointer 直指 shm slot，**零拷贝**；通过 capsule 钩子保证 numpy view 销毁时自动归还 slot

---

## 9. 端到端数据路径对比

```
【ZMQ+pickle 基线】

client 线程(GIL)
   │ pickle.dumps        → memcpy #1 (ndarray→bytes)
   │ zmq.send(UDS)       → memcpy #2 (user→kernel)
                          → memcpy #3 (kernel→user)
server 主循环(单线程)
   │ zmq.recv
   │ pickle.loads        → memcpy #4 (bytes→ndarray)
   │ handler             → 串行

【FastIPC】

client 线程
   │ py::gil_scoped_release
   │ memcpy(ndarray→shm slot)  → memcpy #1（唯一）
   │ ring.push(POD)            → 96 B cache-line 写
   │ eventfd_write(1)          → 一次系统调用（仅空→非空时）
                                      │
                               ┌──────▼─────────────────────────┐
                               │ epoll_wait 唤醒某个 pthread worker │
                               │ try_pop POD（CAS）               │
                               │ handler(shm view)  ← 零拷贝     │
                               │ 多个 worker 并行处理             │
                               └────────────────────────────────┘
```

**预期**：
| tokens | 基线 RTT | FastIPC 预估 RTT | 收益 |
| ---: | ---: | ---: | ---: |
| 1K | 117 µs | ~5 µs | ~23x |
| 16K | 570 µs | ~20 µs | ~28x |
| 64K | 2,099 µs | ~150 µs (memcpy 主导) | ~14x |
| 512K | 16,713 µs | ~1,200 µs (memcpy 主导) | ~14x |

> 注：大 payload 下瓶颈变成 memcpy 本身的 DRAM 带宽（~10 GB/s 量级）。想继续降只能靠应用层预先把 ndarray 直接分配到 shm（省掉这唯一一次 memcpy），这个留给 v0.2。

---

## 10. 代码结构

```
fastipc/
├── CMakeLists.txt
├── include/fastipc/
│   ├── pod.hpp              # PutRequestPOD / ResponsePOD
│   ├── shm_pool.hpp         # ShmPool 接口
│   ├── spsc_ring.hpp        # SPSCRing 模板
│   ├── server.hpp           # Server 类（pthread pool + epoll）
│   └── client.hpp           # Client 类
├── src/
│   ├── shm_pool.cpp
│   ├── server.cpp
│   ├── client.cpp
│   └── pybind_module.cpp    # Python 绑定入口
├── python/fastipc/
│   └── __init__.py          # 暴露 from _fastipc import Server, Client
├── tests/
│   ├── test_shm_pool.cpp    # C++ gtest
│   ├── test_ring.cpp
│   └── test_python_api.py   # pytest，覆盖功能正确性
└── bench/
    ├── bench_cpp.cpp        # 纯 C++ benchmark (client/server 同进程)
    ├── bench_python.py      # 对齐 flexkv_zmq_benchmark.py 的 workload
    └── bench_compare.py     # ZMQ vs FastIPC 对照报告生成
```

### 10.1 构建

```bash
# 依赖：cmake ≥ 3.15, g++ ≥ 9, pybind11, numpy
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
# 产出：fastipc/_fastipc.cpython-3X-linux-gnu.so
pip install -e .
```

### 10.2 关键编译选项

- `-O3 -march=native`：给热路径开自动向量化 + 最新指令集
- `-fno-exceptions`（可选）：ring/shm 热路径不抛异常
- `-pthread`

---

## 11. 实施计划

| 阶段 | 交付物 | 时间估算 |
| :--- | :--- | :--- |
| **M1** | `ShmPool`（单池）+ `SPSCRing` + C++ gtest 通过 | 1 天 |
| **M2** | `Server`（pthread worker）+ epoll/eventfd 唤醒 + 单 client SP-MC 通路 | 1 天 |
| **M3** | pybind11 绑定，`push_put / pull` Python 可跑 | 0.5 天 |
| **M4** | 多 client / 多 worker 并发跑通，复用既有 benchmark workload | 0.5 天 |
| **M5** | 对比 benchmark 报告（vs ZMQ 基线） | 0.5 天 |
| **决策点** | 收益 ≥ 5x 进入 M6；否则归档 | — |
| **M6 (可选)** | 多池（小/中/大 slab）+ 引用计数 + 崩溃恢复 | 1 天 |
| **M7 (可选)** | 合入 FlexKV：在 `flexkv/server/` 新增 `fast_transport.py`，用 env var 切换 | 1 天 |

**M1-M5 总预算：3.5 天**，给出可量化对照报告。

---

## 12. 风险与未决问题

| 项 | 说明 | 当前决定 |
| :--- | :--- | :--- |
| **Python GIL** | push/pull 热路径在 C++ 层 `gil_scoped_release`，worker 是纯 pthread 不碰 Python | ✅ 已解决 |
| **shm 崩溃泄漏** | client 异常退出时申请的 slot 会泄漏 | v0.1 启动时 `shm_unlink` 再 create；v0.2 引入 heartbeat + epoch |
| **slot 生命周期 vs numpy view** | Python handler 可能在 view 还没销毁时就 `ack`，导致 slot 提前回收 | 用 pybind11 capsule 把 unref 绑在 ndarray 销毁上，handler 不需关心 |
| **SP-MC ring 的 CAS 争用** | 多 worker 抢同一个 ring 可能争用 | v0.1 先跑看；必要时降级到 "ring↔worker 一对一"（worker 数 ≥ client 数） |
| **namespace 是 `List[str]`** | POD 放不下变长字符串 | v0.1 只支持预注册 namespace（`register_namespace(name) → int id`），动态 namespace 留到 v0.2 |
| **FlexKV handler 回调开销** | bench 用空 handler 公平对比；真实 handler 走 `kv_task_engine` 的开销另算 | benchmark 对比只测通信层 |
| **macOS 本地开发** | 核心目标是 Linux 服务器，但 dev 机是 macOS | 开 CI 在 Linux 容器里跑测试；本地开发用 Linux docker |

---

## 13. 成功标准

以下任一条件达成即 "值得合入 FlexKV"：

1. **小请求高并发**：1K tokens、8 clients、200 reqs/client 总 QPS **≥ 50,000**（基线 9,349，提升 5.4x）
2. **大请求低延迟**：512K tokens 单请求 RTT **≤ 1.5 ms**（基线 16.7 ms，提升 11x）
3. **CPU 效率**：同 QPS 下 server 端 CPU 占用 **降低 ≥ 50%**

---

## 14. 参考

- FlexKV 现有实现：
  - `flexkv/server/client.py`, `flexkv/server/server.py`, `flexkv/server/request.py`
  - `flexkv/common/ring_buffer.py` — 现有 `SharedOpPool`，思路接近
  - `flexkv/csrc/` — FlexKV 已有 C++ 代码，后续合入时复用 CMake 流
- Benchmark 基线：`/Users/sunfeiqiang/WorkBuddy/Claw/flexkv_zmq_benchmark_report.md`
- 外部：
  - Dmitry Vyukov — *Bounded MPMC queue*
  - LMAX Disruptor — SPMC ring design
  - `eventfd(2)` / `epoll_wait(2)` / `shm_open(3)` / `mmap(2)` man pages
  - pybind11 docs: *Passing numpy arrays*, *Releasing the GIL*
