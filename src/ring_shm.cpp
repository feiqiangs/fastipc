// SPDX-License-Identifier: Apache-2.0
#include "fastipc/ring_shm.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace fastipc {

std::shared_ptr<RingShm> RingShm::create(
    const std::string& name, uint32_t capacity, uint32_t slot_size,
    bool unlink_first)
{
    if ((capacity & (capacity - 1)) != 0)
        throw std::invalid_argument("ring capacity must be power of two");
    if (slot_size == 0)
        throw std::invalid_argument("slot_size must be > 0");

    if (unlink_first) {
        shm_unlink(name.c_str());
    }
    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
        shm_unlink(name.c_str());
        fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd < 0)
            throw std::runtime_error(std::string("ring shm_open failed: ") + strerror(errno));
    }

    size_t total = ring_bytes(capacity, slot_size);
    if (ftruncate(fd, static_cast<off_t>(total)) != 0) {
        int e = errno; ::close(fd); shm_unlink(name.c_str());
        throw std::runtime_error(std::string("ring ftruncate failed: ") + strerror(e));
    }
    void* addr = ::mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        int e = errno; ::close(fd); shm_unlink(name.c_str());
        throw std::runtime_error(std::string("ring mmap failed: ") + strerror(e));
    }

    ring_init(addr, capacity, slot_size);

    auto r = std::shared_ptr<RingShm>(new RingShm());
    r->name_ = name;
    r->fd_ = fd;
    r->base_ = addr;
    r->size_ = total;
    r->header_ = reinterpret_cast<RingHeader*>(addr);
    return r;
}

std::shared_ptr<RingShm> RingShm::attach(const std::string& name) {
    int fd = shm_open(name.c_str(), O_RDWR, 0600);
    if (fd < 0)
        throw std::runtime_error(std::string("ring shm_open attach failed: ") + strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0) {
        int e = errno; ::close(fd);
        throw std::runtime_error(std::string("fstat failed: ") + strerror(e));
    }
    void* addr = ::mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        int e = errno; ::close(fd);
        throw std::runtime_error(std::string("ring mmap attach failed: ") + strerror(errno));
    }
    auto r = std::shared_ptr<RingShm>(new RingShm());
    r->name_ = name;
    r->fd_ = fd;
    r->base_ = addr;
    r->size_ = static_cast<size_t>(st.st_size);
    r->header_ = reinterpret_cast<RingHeader*>(addr);
    return r;
}

RingShm::~RingShm() {
    if (base_) ::munmap(base_, size_);
    if (fd_ >= 0) ::close(fd_);
}

} // namespace fastipc
