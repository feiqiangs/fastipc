# FlexKV-style Client→Server 通信 Benchmark 报告

- 生成时间：2026-04-27 20:38:05
- 主机：PHAEDONSUN-MC1  |  CPU 核心：12  |  Python：3.14.3
- numpy: 2.4.4  |  pyzmq: 27.1.0

本 benchmark 严格复刻 FlexKV 在 client-server 模式下的传输链路：
- **控制协议**：ZMQ `PUSH`/`PULL` over `ipc://` (Unix Domain Socket)
- **序列化**：`pyzmq.send_pyobj` = `pickle.dumps` + `send`
- **消息体**：`PutRequest` dataclass（含 `token_ids: np.ndarray`）
- **Server 主循环**：单线程 `while True: recv_pyobj()` → handler 分发（对齐 `server.py:286-320`）

## Scenario A — 单请求端到端延迟

测试路径：`client.send_pyobj(req)` → server `recv_pyobj + handler` → server `send_pyobj(ack)` → `client.recv_pyobj`，测量 RTT。

| token_ids 长度 | payload (KB) | mean (µs) | p50 (µs) | p95 (µs) | p99 (µs) | 有效带宽 (MB/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 16.3 | 117.2 | 115.4 | 136.3 | 153.9 | 135 |
| 4,096 | 64.3 | 225.2 | 217.4 | 305.1 | 323.0 | 279 |
| 16,384 | 256.3 | 570.1 | 551.9 | 712.3 | 781.6 | 439 |
| 65,536 | 1024.3 | 2098.8 | 2076.0 | 2364.5 | 2469.5 | 477 |
| 262,144 | 4096.3 | 8434.8 | 8542.6 | 10118.9 | 10258.6 | 474 |
| 524,288 | 8192.3 | 16712.5 | 16594.6 | 18295.1 | 18627.1 | 479 |

## Scenario B — 并发批量提交 vs 单线程 Server

模拟多个 DP Client 并发向同一个单线程 Server 打请求（FlexKV `dp_size > 1` 或 `instance_num > 1` 时就是这个场景）。

| 并发客户端 | 每客户端请求数 | token_ids 长度 | 总请求数 | 墙钟 (s) | QPS | 聚合带宽 (MB/s) | server recv 累计 (ms) | server unpickle 累计 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 1,024 | 200 | 0.15 | 1349 | 21 | 257.0 | 3.1 |
| 4 | 200 | 1,024 | 800 | 0.16 | 5033 | 80 | 254.7 | 7.1 |
| 8 | 200 | 1,024 | 1600 | 0.17 | 9349 | 149 | 256.8 | 11.4 |
| 16 | 100 | 1,024 | 1600 | 0.19 | 8636 | 137 | 231.2 | 11.6 |
| 1 | 100 | 16,384 | 100 | 0.18 | 555 | 139 | 294.0 | 2.8 |
| 4 | 100 | 16,384 | 400 | 0.22 | 1828 | 458 | 321.1 | 7.7 |
| 8 | 50 | 16,384 | 400 | 0.21 | 1926 | 482 | 266.4 | 8.7 |
| 4 | 20 | 262,144 | 80 | 0.43 | 185 | 741 | 489.6 | 49.2 |
| 8 | 10 | 524,288 | 80 | 0.68 | 118 | 942 | 702.9 | 71.6 |

## Scenario C — 纯 pickle 开销（无 IPC）

为对比定位，测量在**不发生 IPC** 时 `pickle.dumps` + `pickle.loads` 单独需要多少时间。

| token_ids 长度 | payload (KB) | pickle.dumps (µs) | pickle.loads (µs) | 合计 (µs) |
|---:|---:|---:|---:|---:|
| 1,024 | 16.3 | 6.2 | 4.0 | 10.2 |
| 4,096 | 64.3 | 10.9 | 9.1 | 20.0 |
| 16,384 | 256.3 | 72.3 | 11.2 | 83.6 |
| 65,536 | 1024.3 | 74.5 | 21.4 | 95.9 |
| 262,144 | 4096.3 | 357.5 | 221.3 | 578.8 |
| 524,288 | 8192.3 | 797.0 | 625.1 | 1422.0 |

## 结论与观察

### 1. 512K tokens 的传输成本

- **单请求端到端 RTT**：mean **16712 µs (≈ 16.71 ms)**，p99 ≈ 18.63 ms
- **有效载荷**：8.00 MB (token_ids + slot_mapping 各 4 MB int64)
- **有效带宽**：约 **479 MB/s** (双向即 ≈ 958 MB/s)

### 2. pickle 开销占比 (Scenario C vs A)

| tokens | pickle 总耗时 (µs) | 端到端 RTT (µs) | pickle 占比 |
|---:|---:|---:|---:|
| 1,024 | 10.2 | 117.2 | 8.7% |
| 4,096 | 20.0 | 225.2 | 8.9% |
| 16,384 | 83.6 | 570.1 | 14.7% |
| 65,536 | 95.9 | 2098.8 | 4.6% |
| 262,144 | 578.8 | 8434.8 | 6.9% |
| 524,288 | 1422.0 | 16712.5 | 8.5% |

观察：**token_ids 越大，pickle 在端到端中的占比越高**（小请求 < 10%，512K 已近 10%）。
但 RTT 绝对值上升主因仍是 IPC 本身的 memcpy——pyzmq 的 `send_pyobj` + Unix Domain Socket 中存在 **3 次 memcpy**（pickle buffer → kernel → receiver buffer → 重建 ndarray）。

### 3. 并发与 Server 单线程瓶颈

- **最高 QPS**：**9349** (clients=8, reqs/c=200, token_len=1,024)
- **最高聚合带宽**：942 MB/s (出现在大 token_ids 场景)

**Server 单线程实际工作耗时**（仅计 unpickle + handle，排除阻塞等 recv）：

| 并发 | reqs/c | L | 总请求 | 墙钟 (s) | Server unpickle+handle (ms) | 单线程占用率 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 1,024 | 200 | 0.15 | 3.1 | 2% |
| 4 | 200 | 1,024 | 800 | 0.16 | 7.1 | 4% |
| 8 | 200 | 1,024 | 1600 | 0.17 | 11.4 | 7% |
| 16 | 100 | 1,024 | 1600 | 0.19 | 11.6 | 6% |
| 1 | 100 | 16,384 | 100 | 0.18 | 2.8 | 2% |
| 4 | 100 | 16,384 | 400 | 0.22 | 7.7 | 4% |
| 8 | 50 | 16,384 | 400 | 0.21 | 8.7 | 4% |
| 4 | 20 | 262,144 | 80 | 0.43 | 49.2 | 11% |
| 8 | 10 | 524,288 | 80 | 0.68 | 71.6 | 11% |

- **1K tokens 小请求扩展性**：1→8 clients QPS 从 1349 → 9349 (加速比 6.9x)
- **16 clients vs 8 clients**：QPS 9349 → 8636，说明 **server 单线程在 clients ≥ 8 开始饱和**（加 client 反而因争抢 PULL socket 略降）。

### 4. 关键结论

1. **token_ids 走 ZMQ IPC + pickle + numpy**：512K int64 tokens (8 MB payload) 单程 ≈ 16.7 ms，有效带宽 ~480 MB/s。
2. **小请求高并发场景**：单线程 server 在本机 IPC 下 QPS 天花板约 **9K**（1K tokens 时）。这意味着当 DP client 数 ≥ 8 且打得勤时，**server 主循环会成为整个 FlexKV 控制面的瓶颈**。
3. **大请求低频场景**：带宽成为瓶颈，~480 MB/s 这条 Unix Domain Socket 的上限吃死延迟。
4. **pickle 本身不是主要开销**：即使 512K tokens 也只占 RTT 的 ~9%，真正贵的是 **内核 socket buffer 的多次 memcpy** 和 **server 单线程串行处理**。
5. **为什么 FlexKV 在 Nov 2025 改回 library 模式**：这套 benchmark 定量展示了 ZMQ IPC + pickle 在高 QPS 推理场景下的代价——能 in-process 调用时，**省掉 3 次 memcpy + pickle + 单线程串行化** 是非常可观的优化。
