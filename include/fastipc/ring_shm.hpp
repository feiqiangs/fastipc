// SPDX-License-Identifier: Apache-2.0
// Named shared-memory region that hosts one RingHeader + slots.
#pragma once

#include <memory>
#include <string>
#include "fastipc/spsc_ring.hpp"

namespace fastipc {

class RingShm {
public:
    // Create a fresh ring-shm region. capacity must be power of 2.
    static std::shared_ptr<RingShm> create(
        const std::string& name, uint32_t capacity, uint32_t slot_size,
        bool unlink_first = true);

    // Attach to an existing ring-shm region.
    static std::shared_ptr<RingShm> attach(const std::string& name);

    ~RingShm();

    RingHeader* header() { return header_; }
    const std::string& name() const { return name_; }

private:
    RingShm() = default;
    std::string name_;
    int     fd_ = -1;
    void*   base_ = nullptr;
    size_t  size_ = 0;
    RingHeader* header_ = nullptr;
};

} // namespace fastipc
