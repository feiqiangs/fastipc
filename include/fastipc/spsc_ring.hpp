// SPDX-License-Identifier: Apache-2.0
// Bounded SPSC ring in shared memory; consumer side uses CAS on tail to
// support multiple workers sharing the same ring (SP-MC).
//
// Wakeup is done via a separately-managed FIFO (named pipe) — its path is
// stable so client/server can both open it from any process. The ring header
// no longer carries a file descriptor.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <new>

namespace fastipc {

constexpr size_t CACHELINE = 64;

struct RingHeader {
    alignas(CACHELINE) std::atomic<uint64_t> head;
    char _pad0[CACHELINE - sizeof(std::atomic<uint64_t>)];

    alignas(CACHELINE) std::atomic<uint64_t> tail;
    char _pad1[CACHELINE - sizeof(std::atomic<uint64_t>)];

    uint32_t capacity;        // must be power of two
    uint32_t slot_size;       // bytes per slot (POD sized)
    uint32_t _rsvd0;
    uint32_t _rsvd1;
    uint64_t _rsvd[4];

    // slots follow, each `slot_size` bytes, `capacity` of them
};

inline size_t ring_bytes(uint32_t capacity, uint32_t slot_size) {
    return sizeof(RingHeader) + static_cast<size_t>(capacity) * slot_size;
}

inline void ring_init(void* mem, uint32_t capacity, uint32_t slot_size) {
    auto* h = reinterpret_cast<RingHeader*>(mem);
    std::memset(h, 0, sizeof(RingHeader));
    new (&h->head) std::atomic<uint64_t>(0);
    new (&h->tail) std::atomic<uint64_t>(0);
    h->capacity = capacity;
    h->slot_size = slot_size;
}

template <typename POD>
inline bool ring_push(RingHeader* h, const POD& value, bool* notify_needed_out) {
    static_assert(std::is_trivially_copyable_v<POD>, "POD required");
    uint64_t head = h->head.load(std::memory_order_relaxed);
    uint64_t tail = h->tail.load(std::memory_order_acquire);
    if (head - tail >= h->capacity) {
        return false;
    }
    uint32_t mask = h->capacity - 1;
    auto* slots = reinterpret_cast<uint8_t*>(h + 1);
    std::memcpy(slots + static_cast<size_t>(head & mask) * h->slot_size,
                &value, sizeof(POD));
    h->head.store(head + 1, std::memory_order_release);
    if (notify_needed_out) {
        *notify_needed_out = (head == tail);
    }
    return true;
}

template <typename POD>
inline bool ring_try_pop(RingHeader* h, POD& out) {
    static_assert(std::is_trivially_copyable_v<POD>, "POD required");
    uint32_t mask = h->capacity - 1;
    auto* slots = reinterpret_cast<uint8_t*>(h + 1);
    for (;;) {
        uint64_t tail = h->tail.load(std::memory_order_relaxed);
        uint64_t head = h->head.load(std::memory_order_acquire);
        if (tail >= head) return false;
        POD tmp;
        std::memcpy(&tmp,
                    slots + static_cast<size_t>(tail & mask) * h->slot_size,
                    sizeof(POD));
        if (h->tail.compare_exchange_weak(
                tail, tail + 1,
                std::memory_order_acq_rel, std::memory_order_relaxed)) {
            out = tmp;
            return true;
        }
    }
}

inline uint64_t ring_size(RingHeader* h) {
    uint64_t t = h->tail.load(std::memory_order_acquire);
    uint64_t d = h->head.load(std::memory_order_acquire);
    return d - t;
}

} // namespace fastipc

