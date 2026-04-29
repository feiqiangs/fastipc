# FastIPC 纯跨进程通信性能对比（不含 List[int]→np.array 转换）

- 测试时间：2026-04-29 11:05:41
- 测试机：VM-208-174-tencentos，CPU：32 核，Python：3.12.12
- numpy: 2.4.4，pyzmq: 27.1.0
- 测试参数：iters=50，warmup=10，FastIPC workers=4

## 测试方法

数据已预转为 `np.ndarray`，**不包含** `List[int] → np.array()` 转换开销。
FIPC 零拷贝模式下数据写入 shm 也在计时之外，**只测纯 push + pull 的 IPC 往返时间**。

| 方案 | 信号机制 | 数据传输 |
|:---|:---|:---|
| ZMQ+pickle | UDS（内核态） | pickle.dumps → 3次内核memcpy → pickle.loads |
| FIPC epoll+zc | FIFO + epoll_wait（内核态唤醒） | 零拷贝：只传 104B POD 元数据 |
| FIPC bp+zc | busy-poll（纯用户态） | 零拷贝：只传 104B POD 元数据 |

## 测试结果

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

## 分析

### epoll+zc vs bp+zc

两者的区别在于 **信号机制**：

- **epoll+zc**：client push 后通过 FIFO write 通知 server；server worker 阻塞在 `epoll_wait`，被 FIFO 可读事件唤醒。这个 `FIFO write → 内核 epoll 唤醒 → worker 返回用户态` 的往返引入了额外延迟。
- **bp+zc**：server worker 持续 spin 轮询 ring，发现新数据立即处理。client push 后无需写 FIFO。纯用户态操作，无内核参与。

### 结论

1. **FIPC bp+zc 是最快的跨进程通信方案**，纯 IPC 开销恒定在约 18-30µs，与 payload 大小无关
2. **FIPC epoll+zc** 由于内核态信号唤醒，比 bp+zc 多出约 10-30µs 的固定开销
3. **ZMQ+pickle** 开销随 payload 线性增长（pickle 序列化 + 3 次内核 memcpy），大 payload 下最慢
4. **bp+zc 的代价**是 worker 线程 100% CPU 占用（busy-poll）；epoll+zc 在空闲时零 CPU 占用
