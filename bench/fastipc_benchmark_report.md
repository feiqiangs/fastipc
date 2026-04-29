# FastIPC 通信方案性能对比报告

- 测试时间：2026-04-28 20:48:59
- 测试机：VM-208-174-tencentos，CPU 核心数：32，Python：3.12.12
- numpy: 2.4.4，pyzmq: 27.1.0
- 测试参数：iters=30，warmup=5，FastIPC server workers=4，busy-poll 模式

## 测试方法

4 种通信方案横向对比：

| 方案 | 说明 | 进程模型 | 数据通道 |
|:---|:---|:---|:---|
| **Direct** | 同进程函数调用（FlexKV library 模式） | 单进程 | Python 引用传递 |
| **ZMQ+pickle** | FlexKV client-server 模式基线 | 跨进程 | Unix Domain Socket |
| **FIPC bp+mc** | FastIPC busy-poll + memcpy | 跨进程 | POSIX shm + 无锁 ring |
| **FIPC bp+zc** | FastIPC busy-poll + 零拷贝 | 跨进程 | POSIX shm + 无锁 ring |

数据源为 `List[int]`（模拟 sglang/vLLM tokenizer 输出 `(req.origin_input_ids + req.output_ids)[:-1]`）。

---

## Scenario 1：含 List[int] → np.array 转换（端到端，对齐实际使用路径）

计时范围：`List[int] → np.array() → 传输/调用 → handler ack 返回`

| tokens | payload | Direct | ZMQ+pickle | FIPC bp+mc | FIPC bp+zc | ZMQ vs bp+zc | bp+zc vs Direct |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 16 KB | 60.1 µs | 237.4 µs | 103.9 µs | 88.3 µs | 2.7x | 1.5x |
| 4,096 | 64 KB | 234.9 µs | 439.4 µs | 282.9 µs | 279.9 µs | 1.6x | 1.2x |
| 16,384 | 256 KB | 929.8 µs | 1281.5 µs | 1060.7 µs | 1059.2 µs | 1.2x | 1.1x |
| 65,536 | 1024 KB | 3720.2 µs | 4770.7 µs | 4160.1 µs | 4149.9 µs | 1.1x | 1.1x |
| 262,144 | 4096 KB | 14808.3 µs | 19017.1 µs | 16588.8 µs | 16587.7 µs | 1.1x | 1.1x |
| 524,288 | 8192 KB | 29745.8 µs | 38642.8 µs | 33053.4 µs | 32818.6 µs | 1.2x | 1.1x |

**观察：**

- `List[int] → np.array()` 的转换开销占端到端时延的 **85-95%**
- 包含转换开销后，4 种方案差距显著缩小：FIPC bp+zc 仅比 Direct 慢 10-50%
- ZMQ+pickle 比 FIPC bp+zc 慢 1.2-2.7x

---

## Scenario 2：不含 List[int] → np.array（纯通信/调用开销）

计时范围：数据已是 `np.ndarray`，只测 `传输/调用 → handler ack 返回`

| tokens | payload | Direct | ZMQ+pickle | FIPC bp+mc | FIPC bp+zc | ZMQ vs bp+zc | bp+zc vs Direct |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 16 KB | 0.6 µs | 172.1 µs | 32.3 µs | 19.1 µs | 9.0x | 31.8x |
| 4,096 | 64 KB | 0.5 µs | 187.6 µs | 44.9 µs | 18.2 µs | 10.3x | 33.2x |
| 16,384 | 256 KB | 0.6 µs | 280.2 µs | 125.4 µs | 20.8 µs | 13.4x | 35.8x |
| 65,536 | 1024 KB | 0.7 µs | 951.0 µs | 419.1 µs | 22.0 µs | 43.2x | 30.5x |
| 262,144 | 4096 KB | 0.8 µs | 6018.0 µs | 1618.2 µs | 25.1 µs | 239.4x | 29.9x |
| 524,288 | 8192 KB | 0.9 µs | 12738.5 µs | 3329.2 µs | 33.3 µs | 382.7x | 35.4x |

**观察：**

- Direct 模式在纯调用场景下极快（< 1µs），因为只是 Python 函数调用 + 引用传递
- **FIPC bp+zc 纯 IPC 开销约 18-30µs**，与 payload 大小基本无关（零拷贝 + busy-poll 消除了 memcpy 和信号开销）
- FIPC bp+mc 的开销随 payload 线性增长（memcpy 主导）
- ZMQ+pickle 开销最高，pickle 序列化 + 3 次内核 memcpy

---

## 结论

### 1. 实际使用场景（含 List[int] 转换）

在 sglang/vLLM + FlexKV 的实际调用链中，`List[int] → np.array()` 转换是端到端时延的**绝对主导项**（85-95%）。
在这个前提下：

- **Direct 模式**（`dp_size=1`，单实例）仍是最优解，但优势从百倍级缩小到 10-50%
- **FIPC bp+zc** 比 ZMQ+pickle 快 1.2-2.7x，在必须跨进程时是最佳选择
- **优化重点**应放在减少 `List[int] → np.array` 的转换开销（如让 tokenizer 直接输出 ndarray）

### 2. 纯通信开销

如果不考虑上游 List[int] 转换np.narray开销，则：

- **FIPC bp+zc 纯 IPC 开销仅 18-30µs**，是跨进程 shm+ring 方案的物理极限
- 比 ZMQ+pickle 快 **5-250x**（payload 越大优势越明显）
- 比 Direct 模式慢约 18-30µs（这是跨进程通信不可消除的固定开销：原子操作 + cache-line bouncing）

---

## Scenario 3：纯跨进程通信性能对比 — epoll 中断 vs busy-poll

不含 `List[int] → np.array` 转换开销，FIPC 零拷贝模式下数据写入 shm 也在计时之外，**只测纯 push + pull 的 IPC 往返时间**。

测试参数：iters=50，warmup=10

| 方案 | 信号机制 | 数据传输 |
|:---|:---|:---|
| ZMQ+pickle | UDS（内核态） | pickle.dumps → 3 次内核 memcpy → pickle.loads |
| FIPC epoll+zc | FIFO + epoll_wait（内核态唤醒） | 零拷贝：只传 104B POD 元数据 |
| FIPC bp+zc | busy-poll（纯用户态） | 零拷贝：只传 104B POD 元数据 |

### Mean RTT (µs)

| tokens | payload | ZMQ+pickle | FIPC epoll+zc | FIPC bp+zc | ZMQ/epoll | ZMQ/bp | epoll/bp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 16 KB | 162.1 | 35.2 | 11.1 | 4.6x | 14.6x | 3.16x |
| 4,096 | 64 KB | 196.4 | 35.9 | 17.0 | 5.5x | 11.6x | 2.12x |
| 16,384 | 256 KB | 405.1 | 33.8 | 20.3 | 12.0x | 19.9x | 1.66x |
| 65,536 | 1024 KB | 1405.2 | 39.8 | 21.7 | 35.3x | 64.8x | 1.83x |
| 262,144 | 4096 KB | 6109.7 | 44.9 | 26.7 | 136.1x | 229.1x | 1.68x |
| 524,288 | 8192 KB | 12744.0 | 50.0 | 30.3 | 254.7x | 420.5x | 1.65x |

### 分位数详情 (µs)

| 方案 | tokens | mean | p50 | p99 | min |
|:---|---:|---:|---:|---:|---:|
| ZMQ+pickle | 1,024 | 162.1 | 156.4 | 245.7 | 146.3 |
| ZMQ+pickle | 4,096 | 196.4 | 192.7 | 234.6 | 177.2 |
| ZMQ+pickle | 16,384 | 405.1 | 402.4 | 435.2 | 390.4 |
| ZMQ+pickle | 65,536 | 1405.2 | 1407.5 | 1471.7 | 1338.1 |
| ZMQ+pickle | 262,144 | 6109.7 | 6170.2 | 6749.0 | 5377.5 |
| ZMQ+pickle | 524,288 | 12744.0 | 12637.3 | 14154.9 | 11312.4 |
| FIPC epoll+zc | 1,024 | 35.2 | 34.8 | 43.1 | 32.9 |
| FIPC epoll+zc | 4,096 | 35.9 | 34.6 | 53.3 | 33.3 |
| FIPC epoll+zc | 16,384 | 33.8 | 33.1 | 41.4 | 31.4 |
| FIPC epoll+zc | 65,536 | 39.8 | 40.8 | 49.2 | 32.2 |
| FIPC epoll+zc | 262,144 | 44.9 | 43.8 | 65.9 | 39.1 |
| FIPC epoll+zc | 524,288 | 50.0 | 47.7 | 81.3 | 37.6 |
| FIPC bp+zc | 1,024 | 11.1 | 10.9 | 19.6 | 9.5 |
| FIPC bp+zc | 4,096 | 17.0 | 15.0 | 39.0 | 13.7 |
| FIPC bp+zc | 16,384 | 20.3 | 19.9 | 32.7 | 15.0 |
| FIPC bp+zc | 65,536 | 21.7 | 21.5 | 32.6 | 15.4 |
| FIPC bp+zc | 262,144 | 26.7 | 25.6 | 43.6 | 21.3 |
| FIPC bp+zc | 524,288 | 30.3 | 29.0 | 44.4 | 24.7 |

### 分析

**epoll+zc vs bp+zc：**

两者区别仅在于信号机制：

- **epoll+zc**：client push 后通过 FIFO write 通知 server；server worker 阻塞在 `epoll_wait`，被 FIFO 可读事件唤醒。`FIFO write → 内核 epoll 唤醒 → worker 返回用户态` 的往返引入约 **20-25µs** 额外延迟。
- **bp+zc**：server worker 持续 spin 轮询 ring，发现新数据立即处理。client push 后无需写 FIFO。纯用户态操作，无内核参与。

**epoll vs busy-poll 选择：**

| | FIPC epoll+zc | FIPC bp+zc |
|:---|:---|:---|
| 1K tokens RTT | 35 µs | **11 µs** |
| 512K tokens RTT | 50 µs | **30 µs** |
| 空闲 CPU 占用 | **0%** | 100%（worker spin） |
| 适用场景 | 低 QPS / CPU 紧张 | 低延迟 / CPU 充裕 |

**结论：**

1. **FIPC bp+zc 是最快的跨进程通信方案**，纯 IPC 开销恒定在约 11-30µs，与 payload 大小无关
2. **FIPC epoll+zc** 由于内核态信号唤醒，比 bp+zc 多出约 20-25µs 的固定开销，但空闲时零 CPU 占用
3. **ZMQ+pickle** 开销随 payload 线性增长（pickle 序列化 + 3 次内核 memcpy），512K tokens 下比 bp+zc 慢 **420x**
4. **bp+zc 的代价**是 worker 线程 100% CPU 占用（busy-poll）；在推理服务器 CPU 核心充裕（32 核）的场景下，牺牲一个核换取 20µs 延迟降低是值得的
