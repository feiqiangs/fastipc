// SPDX-License-Identifier: Apache-2.0
#include "fastipc/shm_pool.hpp"

#include <atomic>
#include <cstring>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>

namespace fastipc {

static constexpr uint64_t POOL_MAGIC = 0x46495043504F4F4Cull;
static constexpr uint32_t POOL_VERSION = 2;
static constexpr size_t   ALIGN = 4096;

static inline size_t align_up(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

// New layout (v2):
//   ShmPoolHeader
//   atomic<int32_t> refcount[N]   (per-slot reference count)
//   atomic<uint32_t> next_arr[N]  (Treiber stack "next" links, atomic to be safe)
//   align to 4096
//   slot[0..N-1]
size_t ShmPool::compute_total_bytes(uint32_t slot_size, uint32_t num_slots) {
    size_t hdr = sizeof(ShmPoolHeader);
    size_t rc  = sizeof(std::atomic<int32_t>) * static_cast<size_t>(num_slots);
    size_t nxt = sizeof(std::atomic<uint32_t>) * static_cast<size_t>(num_slots);
    size_t meta_end = align_up(hdr + rc + nxt, ALIGN);
    size_t data = static_cast<size_t>(slot_size) * static_cast<size_t>(num_slots);
    return meta_end + data;
}

static inline std::atomic<uint32_t>* nextarr_of(ShmPoolHeader* h,
                                                std::atomic<int32_t>* /*rc*/) {
    auto* base = reinterpret_cast<uint8_t*>(h) + sizeof(ShmPoolHeader);
    base += sizeof(std::atomic<int32_t>) * h->num_slots;
    return reinterpret_cast<std::atomic<uint32_t>*>(base);
}

std::shared_ptr<ShmPool> ShmPool::create(
    const std::string& name, uint32_t pool_id,
    uint32_t slot_size, uint32_t num_slots, bool unlink_first)
{
    if (slot_size == 0 || num_slots == 0)
        throw std::invalid_argument("slot_size and num_slots must be > 0");

    if (unlink_first) shm_unlink(name.c_str());

    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
        shm_unlink(name.c_str());
        fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd < 0)
            throw std::runtime_error(std::string("shm_open create failed: ") + strerror(errno));
    }

    size_t total = compute_total_bytes(slot_size, num_slots);
    if (ftruncate(fd, static_cast<off_t>(total)) != 0) {
        int e = errno; ::close(fd); shm_unlink(name.c_str());
        throw std::runtime_error(std::string("ftruncate failed: ") + strerror(e));
    }

    auto p = std::shared_ptr<ShmPool>(new ShmPool());
    p->name_ = name;
    p->fd_ = fd;
    p->mmap_region(fd, total);

    auto* h = reinterpret_cast<ShmPoolHeader*>(p->base_);
    std::memset(h, 0, sizeof(ShmPoolHeader));
    h->magic = POOL_MAGIC;
    h->version = POOL_VERSION;
    h->pool_id = pool_id;
    h->slot_size = slot_size;
    h->num_slots = num_slots;

    size_t rc_bytes  = sizeof(std::atomic<int32_t>) * num_slots;
    size_t nxt_bytes = sizeof(std::atomic<uint32_t>) * num_slots;
    h->data_offset = align_up(sizeof(ShmPoolHeader) + rc_bytes + nxt_bytes, ALIGN);
    h->total_bytes = total;
    new (&h->free_head) std::atomic<uint64_t>(0);

    p->header_ = h;
    p->build_refcount_table();
    auto* nx = nextarr_of(h, p->refcounts_);
    for (uint32_t i = 0; i < num_slots; ++i) {
        new (&p->refcounts_[i]) std::atomic<int32_t>(0);
        new (&nx[i]) std::atomic<uint32_t>(0);
    }
    p->data_ = reinterpret_cast<uint8_t*>(p->base_) + h->data_offset;

    // Build initial freelist: push slots in reverse order so alloc returns 0,1,2,...
    for (int32_t i = static_cast<int32_t>(num_slots) - 1; i >= 0; --i) {
        p->free_slot(i);
    }

    return p;
}

std::shared_ptr<ShmPool> ShmPool::attach(const std::string& name) {
    int fd = shm_open(name.c_str(), O_RDWR, 0600);
    if (fd < 0)
        throw std::runtime_error(std::string("shm_open attach failed: ") + strerror(errno));

    struct stat st;
    if (fstat(fd, &st) != 0) {
        int e = errno; ::close(fd);
        throw std::runtime_error(std::string("fstat failed: ") + strerror(e));
    }

    auto p = std::shared_ptr<ShmPool>(new ShmPool());
    p->name_ = name;
    p->fd_ = fd;
    p->mmap_region(fd, static_cast<size_t>(st.st_size));

    auto* h = reinterpret_cast<ShmPoolHeader*>(p->base_);
    if (h->magic != POOL_MAGIC || h->version != POOL_VERSION) {
        ::munmap(p->base_, p->size_); ::close(fd);
        throw std::runtime_error("shm pool magic/version mismatch");
    }
    p->header_ = h;
    p->build_refcount_table();
    p->data_ = reinterpret_cast<uint8_t*>(p->base_) + h->data_offset;
    return p;
}

void ShmPool::mmap_region(int fd, size_t len) {
    void* addr = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        int e = errno; ::close(fd);
        throw std::runtime_error(std::string("mmap failed: ") + strerror(e));
    }
    base_ = addr; size_ = len;
}

void ShmPool::build_refcount_table() {
    auto* p = reinterpret_cast<uint8_t*>(header_) + sizeof(ShmPoolHeader);
    refcounts_ = reinterpret_cast<std::atomic<int32_t>*>(p);
}

ShmPool::~ShmPool() {
    if (base_) ::munmap(base_, size_);
    if (fd_ >= 0) ::close(fd_);
}

void* ShmPool::slot_ptr(int32_t slot_id) {
    if (slot_id < 0 || static_cast<uint32_t>(slot_id) >= header_->num_slots) return nullptr;
    return data_ + static_cast<size_t>(slot_id) * header_->slot_size;
}

// Lock-free Treiber stack: head is encoded as (tag<<32)|(slot_id+1); 0 = empty.
// "next" links live in a parallel atomic array, NOT in the slot data, so they
// remain valid even while data is being written by a producer.
int32_t ShmPool::alloc_slot() {
    auto* nx = nextarr_of(header_, refcounts_);
    uint64_t old_head = header_->free_head.load(std::memory_order_acquire);
    for (;;) {
        uint32_t head_idx = static_cast<uint32_t>(old_head & 0xFFFFFFFFu);
        if (head_idx == 0) return -1;  // empty
        int32_t slot_id = static_cast<int32_t>(head_idx) - 1;
        uint32_t next_idx = nx[slot_id].load(std::memory_order_relaxed);
        uint64_t tag = (old_head >> 32) & 0xFFFFFFFFu;
        uint64_t new_head = ((tag + 1) << 32) | static_cast<uint64_t>(next_idx);
        if (header_->free_head.compare_exchange_weak(
                old_head, new_head,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            refcounts_[slot_id].store(1, std::memory_order_release);
            return slot_id;
        }
    }
}

void ShmPool::free_slot(int32_t slot_id) {
    if (slot_id < 0 || static_cast<uint32_t>(slot_id) >= header_->num_slots) return;
    auto* nx = nextarr_of(header_, refcounts_);
    uint64_t old_head = header_->free_head.load(std::memory_order_acquire);
    for (;;) {
        uint32_t head_idx = static_cast<uint32_t>(old_head & 0xFFFFFFFFu);
        nx[slot_id].store(head_idx, std::memory_order_relaxed);
        uint64_t tag = (old_head >> 32) & 0xFFFFFFFFu;
        uint64_t new_head = ((tag + 1) << 32) | static_cast<uint64_t>(slot_id + 1);
        if (header_->free_head.compare_exchange_weak(
                old_head, new_head,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            return;
        }
    }
}

void ShmPool::ref(int32_t slot_id) {
    if (slot_id < 0 || static_cast<uint32_t>(slot_id) >= header_->num_slots) return;
    refcounts_[slot_id].fetch_add(1, std::memory_order_relaxed);
}

int32_t ShmPool::unref(int32_t slot_id) {
    if (slot_id < 0 || static_cast<uint32_t>(slot_id) >= header_->num_slots) return -1;
    int32_t prev = refcounts_[slot_id].fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
        free_slot(slot_id);
        return 0;
    }
    return prev - 1;
}

} // namespace fastipc
