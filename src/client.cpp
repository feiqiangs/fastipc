// SPDX-License-Identifier: Apache-2.0
#include "fastipc/client.hpp"

#include <sys/epoll.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace fastipc {

std::string Client::request_ring_name(uint32_t cid) const {
    return "/" + cfg_.shm_prefix + "_req_" + std::to_string(cid);
}
std::string Client::response_ring_name(uint32_t cid) const {
    return "/" + cfg_.shm_prefix + "_resp_" + std::to_string(cid);
}
std::string Client::pool_name(uint32_t pool_id) const {
    return "/" + cfg_.shm_prefix + "_pool_" + std::to_string(pool_id);
}
std::string Client::request_fifo_path(uint32_t cid) const {
    return cfg_.fifo_dir + "/" + cfg_.shm_prefix + "_reqfifo_" + std::to_string(cid);
}
std::string Client::response_fifo_path(uint32_t cid) const {
    return cfg_.fifo_dir + "/" + cfg_.shm_prefix + "_respfifo_" + std::to_string(cid);
}

std::shared_ptr<Client> Client::create(const ClientConfig& cfg) {
    auto c = std::shared_ptr<Client>(new Client());
    c->cfg_ = cfg;
    c->task_id_base_ = static_cast<int64_t>(cfg.dp_client_id) * 10'000'000LL;

    for (size_t i = 0; i < cfg.pools.size(); ++i) {
        c->pools_.push_back(ShmPool::attach(c->pool_name(static_cast<uint32_t>(i))));
    }
    c->req_ring_  = RingShm::attach(c->request_ring_name(cfg.dp_client_id));
    c->resp_ring_ = RingShm::attach(c->response_ring_name(cfg.dp_client_id));

    // Open FIFO ends. Server creates the FIFOs; client opens both ends RDWR
    // to be robust against ordering issues.
    c->req_fifo_write_fd_ = ::open(c->request_fifo_path(cfg.dp_client_id).c_str(),
                                    O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (c->req_fifo_write_fd_ < 0)
        throw std::runtime_error(std::string("client open req fifo: ") + strerror(errno));
    c->resp_fifo_read_fd_ = ::open(c->response_fifo_path(cfg.dp_client_id).c_str(),
                                    O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (c->resp_fifo_read_fd_ < 0)
        throw std::runtime_error(std::string("client open resp fifo: ") + strerror(errno));

    c->epoll_fd_ = ::epoll_create1(EPOLL_CLOEXEC);
    if (c->epoll_fd_ < 0)
        throw std::runtime_error(std::string("client epoll_create1: ") + strerror(errno));
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.u64 = 0;
    if (::epoll_ctl(c->epoll_fd_, EPOLL_CTL_ADD, c->resp_fifo_read_fd_, &ev) != 0)
        throw std::runtime_error(std::string("client epoll_ctl: ") + strerror(errno));
    return c;
}

Client::~Client() {
    if (epoll_fd_ >= 0) ::close(epoll_fd_);
    if (req_fifo_write_fd_ >= 0) ::close(req_fifo_write_fd_);
    if (resp_fifo_read_fd_ >= 0) ::close(resp_fifo_read_fd_);
}

std::shared_ptr<ShmPool> Client::pool_for_size(size_t nbytes) {
    for (auto& p : pools_) if (nbytes <= p->slot_size()) return p;
    return nullptr;
}

int64_t Client::push_put(
    const void* token_ids_ptr, size_t token_ids_nbytes, uint16_t tid_dtype,
    const void* slot_mapping_ptr, size_t slot_mapping_nbytes, uint16_t sm_dtype,
    const void* token_mask_ptr, size_t token_mask_nbytes, uint16_t mask_dtype,
    int32_t layer_granularity, int32_t namespace_id, uint16_t req_type)
{
    if (!token_ids_ptr || token_ids_nbytes == 0)
        throw std::invalid_argument("token_ids required");

    auto ti_pool = pool_for_size(token_ids_nbytes);
    if (!ti_pool) throw std::runtime_error("token_ids too large for any pool");
    int32_t ti_slot = ti_pool->alloc_slot();
    // Block-wait if pool is exhausted: spin then briefly sleep.
    int alloc_retry = 0;
    while (ti_slot < 0) {
        if (++alloc_retry > 100000) {
            throw std::runtime_error("token_ids slot pool exhausted (timeout)");
        }
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
        if (alloc_retry % 1000 == 0) {
            // Yield so the consumer can drain.
            ::usleep(100);
        }
        ti_slot = ti_pool->alloc_slot();
    }
    std::memcpy(ti_pool->slot_ptr(ti_slot), token_ids_ptr, token_ids_nbytes);

    int32_t sm_slot = -1;
    std::shared_ptr<ShmPool> sm_pool;
    if (slot_mapping_ptr && slot_mapping_nbytes > 0) {
        sm_pool = pool_for_size(slot_mapping_nbytes);
        if (!sm_pool) { ti_pool->unref(ti_slot); throw std::runtime_error("slot_mapping too large"); }
        sm_slot = sm_pool->alloc_slot();
        alloc_retry = 0;
        while (sm_slot < 0) {
            if (++alloc_retry > 100000) {
                ti_pool->unref(ti_slot);
                throw std::runtime_error("slot_mapping pool exhausted (timeout)");
            }
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
            if (alloc_retry % 1000 == 0) ::usleep(100);
            sm_slot = sm_pool->alloc_slot();
        }
        std::memcpy(sm_pool->slot_ptr(sm_slot), slot_mapping_ptr, slot_mapping_nbytes);
    }

    int32_t mask_slot = -1;
    std::shared_ptr<ShmPool> mask_pool;
    if (token_mask_ptr && token_mask_nbytes > 0) {
        mask_pool = pool_for_size(token_mask_nbytes);
        if (!mask_pool) {
            ti_pool->unref(ti_slot);
            if (sm_slot >= 0) sm_pool->unref(sm_slot);
            throw std::runtime_error("token_mask too large");
        }
        mask_slot = mask_pool->alloc_slot();
        alloc_retry = 0;
        while (mask_slot < 0) {
            if (++alloc_retry > 100000) {
                ti_pool->unref(ti_slot);
                if (sm_slot >= 0) sm_pool->unref(sm_slot);
                throw std::runtime_error("token_mask pool exhausted (timeout)");
            }
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
            if (alloc_retry % 1000 == 0) ::usleep(100);
            mask_slot = mask_pool->alloc_slot();
        }
        std::memcpy(mask_pool->slot_ptr(mask_slot), token_mask_ptr, token_mask_nbytes);
    }

    PutRequestPOD req{};
    req.magic = FAST_MAGIC;
    req.req_type = req_type;
    req.flags = (mask_slot >= 0) ? FLAG_HAS_MASK : 0;
    if (sm_slot >= 0) req.flags |= FLAG_HAS_SLOT_MAP;
    req.dp_client_id = cfg_.dp_client_id;
    req.namespace_id = namespace_id;
    req.layer_granularity = layer_granularity;
    int64_t tid = task_id_base_ + task_counter_.fetch_add(1, std::memory_order_relaxed);
    req.task_id = tid;

    req.token_ids.pool_id = ti_pool->pool_id();
    req.token_ids.slot_id = static_cast<uint32_t>(ti_slot);
    req.token_ids.offset  = 0;
    req.token_ids.nbytes  = static_cast<uint32_t>(token_ids_nbytes);
    req.token_ids.dtype   = tid_dtype;
    req.token_ids.ndim    = 1;
    req.token_ids.shape0  = static_cast<uint32_t>(token_ids_nbytes / (dtype_itemsize(tid_dtype) ? dtype_itemsize(tid_dtype) : 1));

    if (sm_slot >= 0) {
        req.slot_mapping.pool_id = sm_pool->pool_id();
        req.slot_mapping.slot_id = static_cast<uint32_t>(sm_slot);
        req.slot_mapping.offset  = 0;
        req.slot_mapping.nbytes  = static_cast<uint32_t>(slot_mapping_nbytes);
        req.slot_mapping.dtype   = sm_dtype;
        req.slot_mapping.ndim    = 1;
        req.slot_mapping.shape0  = static_cast<uint32_t>(slot_mapping_nbytes / (dtype_itemsize(sm_dtype) ? dtype_itemsize(sm_dtype) : 1));
    }
    if (mask_slot >= 0) {
        req.token_mask.pool_id = mask_pool->pool_id();
        req.token_mask.slot_id = static_cast<uint32_t>(mask_slot);
        req.token_mask.offset  = 0;
        req.token_mask.nbytes  = static_cast<uint32_t>(token_mask_nbytes);
        req.token_mask.dtype   = mask_dtype;
        req.token_mask.ndim    = 1;
        req.token_mask.shape0  = static_cast<uint32_t>(token_mask_nbytes / (dtype_itemsize(mask_dtype) ? dtype_itemsize(mask_dtype) : 1));
    }

    auto* h = req_ring_->header();
    bool need_notify = false;
    for (int i = 0; i < 1 << 20; ++i) {
        if (ring_push(h, req, &need_notify)) {
            if (need_notify) {
                char b = 1;
                ::write(req_fifo_write_fd_, &b, 1);
            }
            return tid;
        }
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
    }
    throw std::runtime_error("request ring full (backpressure timeout)");
}

bool Client::pull(PullResult& out, int timeout_ms) {
    auto* h = resp_ring_->header();
    if (ring_try_pop(h, out.pod)) {
        char buf[64];
        while (::read(resp_fifo_read_fd_, buf, sizeof(buf)) > 0) {}
    } else {
        epoll_event ev;
        int n = ::epoll_wait(epoll_fd_, &ev, 1, timeout_ms);
        if (n <= 0) return false;
        char buf[64];
        while (::read(resp_fifo_read_fd_, buf, sizeof(buf)) > 0) {}
        if (!ring_try_pop(h, out.pod)) return false;
    }

    if (out.pod.mask.nbytes > 0 && out.pod.mask.pool_id < pools_.size()) {
        auto& p = pools_[out.pod.mask.pool_id];
        out.mask_ptr = reinterpret_cast<uint8_t*>(p->slot_ptr(out.pod.mask.slot_id)) + out.pod.mask.offset;
        out.mask_nbytes = out.pod.mask.nbytes;
        out.mask_dtype  = out.pod.mask.dtype;
        out.mask_nitems = out.pod.mask.nbytes / (dtype_itemsize(out.pod.mask.dtype) ? dtype_itemsize(out.pod.mask.dtype) : 1);
    } else {
        out.mask_ptr = nullptr;
        out.mask_nbytes = 0;
    }
    return true;
}

void Client::release_pull_result(const PullResult& r) {
    if (r.pod.mask.nbytes > 0 && r.pod.mask.pool_id < pools_.size()) {
        pools_[r.pod.mask.pool_id]->unref(static_cast<int32_t>(r.pod.mask.slot_id));
    }
}

ShmBuffer Client::alloc_shm_buffer(size_t nbytes) {
    auto pool = pool_for_size(nbytes);
    if (!pool) throw std::runtime_error("no pool large enough for alloc_shm_buffer");
    int32_t slot = pool->alloc_slot();
    int alloc_retry = 0;
    while (slot < 0) {
        if (++alloc_retry > 100000) {
            throw std::runtime_error("alloc_shm_buffer: pool exhausted (timeout)");
        }
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
        if (alloc_retry % 1000 == 0) ::usleep(100);
        slot = pool->alloc_slot();
    }
    ShmBuffer buf;
    buf.pool = pool;
    buf.slot_id = slot;
    buf.ptr = pool->slot_ptr(slot);
    buf.capacity = pool->slot_size();
    buf.pool_id = pool->pool_id();
    return buf;
}

void Client::release_shm_buffer(ShmBuffer& buf) {
    if (buf.pool && buf.slot_id >= 0) {
        buf.pool->unref(buf.slot_id);
        buf.slot_id = -1;
        buf.ptr = nullptr;
    }
}

int64_t Client::push_put_prealloc(
    const ShmBuffer& token_ids_buf, size_t token_ids_nbytes, uint16_t tid_dtype,
    const ShmBuffer* slot_mapping_buf, size_t slot_mapping_nbytes, uint16_t sm_dtype,
    const ShmBuffer* token_mask_buf, size_t token_mask_nbytes, uint16_t mask_dtype,
    int32_t layer_granularity, int32_t namespace_id, uint16_t req_type)
{
    if (token_ids_buf.slot_id < 0 || !token_ids_buf.ptr)
        throw std::invalid_argument("token_ids ShmBuffer is invalid");

    PutRequestPOD req{};
    req.magic = FAST_MAGIC;
    req.req_type = req_type;
    req.flags = 0;
    req.dp_client_id = cfg_.dp_client_id;
    req.namespace_id = namespace_id;
    req.layer_granularity = layer_granularity;
    int64_t tid = task_id_base_ + task_counter_.fetch_add(1, std::memory_order_relaxed);
    req.task_id = tid;

    req.token_ids.pool_id = token_ids_buf.pool_id;
    req.token_ids.slot_id = static_cast<uint32_t>(token_ids_buf.slot_id);
    req.token_ids.offset  = 0;
    req.token_ids.nbytes  = static_cast<uint32_t>(token_ids_nbytes);
    req.token_ids.dtype   = tid_dtype;
    req.token_ids.ndim    = 1;
    req.token_ids.shape0  = static_cast<uint32_t>(token_ids_nbytes / (dtype_itemsize(tid_dtype) ? dtype_itemsize(tid_dtype) : 1));

    if (slot_mapping_buf && slot_mapping_buf->slot_id >= 0 && slot_mapping_nbytes > 0) {
        req.flags |= FLAG_HAS_SLOT_MAP;
        req.slot_mapping.pool_id = slot_mapping_buf->pool_id;
        req.slot_mapping.slot_id = static_cast<uint32_t>(slot_mapping_buf->slot_id);
        req.slot_mapping.offset  = 0;
        req.slot_mapping.nbytes  = static_cast<uint32_t>(slot_mapping_nbytes);
        req.slot_mapping.dtype   = sm_dtype;
        req.slot_mapping.ndim    = 1;
        req.slot_mapping.shape0  = static_cast<uint32_t>(slot_mapping_nbytes / (dtype_itemsize(sm_dtype) ? dtype_itemsize(sm_dtype) : 1));
    }

    if (token_mask_buf && token_mask_buf->slot_id >= 0 && token_mask_nbytes > 0) {
        req.flags |= FLAG_HAS_MASK;
        req.token_mask.pool_id = token_mask_buf->pool_id;
        req.token_mask.slot_id = static_cast<uint32_t>(token_mask_buf->slot_id);
        req.token_mask.offset  = 0;
        req.token_mask.nbytes  = static_cast<uint32_t>(token_mask_nbytes);
        req.token_mask.dtype   = mask_dtype;
        req.token_mask.ndim    = 1;
        req.token_mask.shape0  = static_cast<uint32_t>(token_mask_nbytes / (dtype_itemsize(mask_dtype) ? dtype_itemsize(mask_dtype) : 1));
    }

    auto* h = req_ring_->header();
    bool need_notify = false;
    for (int i = 0; i < 1 << 20; ++i) {
        if (ring_push(h, req, &need_notify)) {
            if (need_notify) {
                char b = 1;
                ::write(req_fifo_write_fd_, &b, 1);
            }
            return tid;
        }
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
    }
    throw std::runtime_error("request ring full (backpressure timeout)");
}

} // namespace fastipc
