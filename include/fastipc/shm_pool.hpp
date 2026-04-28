// SPDX-License-Identifier: Apache-2.0
// POSIX shared-memory slab pool with lock-free freelist.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>

namespace fastipc {

// Layout in shared memory:
//   ShmPoolHeader (with atomic freelist head + per-slot refcount array)
//   padding to align data region to 4096
//   slot[0] | slot[1] | ... | slot[N-1]   (each 'slot_size' bytes)
//
// freelist uses a simple lock-free Treiber stack on an intrusive "next" word
// written to the first 4 bytes of each free slot. ABA is avoided by tagging
// the head with a 32-bit counter.

struct ShmPoolHeader {
    uint64_t magic;              // 0x46495043504F4F4Cu "FIPCPOOL"
    uint32_t version;
    uint32_t pool_id;
    uint32_t slot_size;
    uint32_t num_slots;
    uint64_t data_offset;        // from &header, must be 4096-aligned
    uint64_t total_bytes;        // entire region size
    // Treiber stack head: high 32b = tag, low 32b = head slot id (+1; 0 = empty)
    std::atomic<uint64_t> free_head;
    // reserved
    uint64_t _rsvd[4];
    // refcount[N] follows; then padding; then data region.
};

class ShmPool {
public:
    // Creates (or re-creates) a named POSIX shm region.
    // If `unlink_first` is true, any existing region with the same name is removed.
    static std::shared_ptr<ShmPool> create(
        const std::string& name, uint32_t pool_id,
        uint32_t slot_size, uint32_t num_slots, bool unlink_first = true);

    // Attaches to an already-created region.
    static std::shared_ptr<ShmPool> attach(const std::string& name);

    ~ShmPool();

    // Allocates one slot. Returns -1 if pool is exhausted.
    int32_t alloc_slot();

    // Frees one slot (refcount reset to 0).
    void free_slot(int32_t slot_id);

    // Reference counting (used when multiple consumers hold a slot).
    void ref(int32_t slot_id);
    // Decrements refcount; returns the new value. When it reaches 0, the slot
    // is returned to the freelist automatically.
    int32_t unref(int32_t slot_id);

    void* slot_ptr(int32_t slot_id);
    uint32_t slot_size() const { return header_->slot_size; }
    uint32_t num_slots() const { return header_->num_slots; }
    uint32_t pool_id()   const { return header_->pool_id; }

    const std::string& name() const { return name_; }

private:
    ShmPool() = default;
    static size_t compute_total_bytes(uint32_t slot_size, uint32_t num_slots);

    std::string  name_;
    int          fd_ = -1;
    void*        base_ = nullptr;
    size_t       size_ = 0;
    ShmPoolHeader* header_ = nullptr;
    std::atomic<int32_t>* refcounts_ = nullptr;
    uint8_t*     data_ = nullptr;

    void mmap_region(int fd, size_t len);
    void build_refcount_table();
};

} // namespace fastipc
