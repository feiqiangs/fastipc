// SPDX-License-Identifier: Apache-2.0
#include "fastipc/server.hpp"

#include <sys/eventfd.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <stdexcept>

namespace fastipc {

std::string Server::request_ring_name(uint32_t client_id) const {
    return "/" + cfg_.shm_prefix + "_req_" + std::to_string(client_id);
}
std::string Server::response_ring_name(uint32_t client_id) const {
    return "/" + cfg_.shm_prefix + "_resp_" + std::to_string(client_id);
}
std::string Server::pool_name(uint32_t pool_id) const {
    return "/" + cfg_.shm_prefix + "_pool_" + std::to_string(pool_id);
}
std::string Server::request_fifo_path(uint32_t client_id) const {
    return cfg_.fifo_dir + "/" + cfg_.shm_prefix + "_reqfifo_" + std::to_string(client_id);
}
std::string Server::response_fifo_path(uint32_t client_id) const {
    return cfg_.fifo_dir + "/" + cfg_.shm_prefix + "_respfifo_" + std::to_string(client_id);
}

// Helper: ensure FIFO at given path exists with mode 0600. Removes any stale
// file/symlink first.
static void ensure_fifo(const std::string& path) {
    ::unlink(path.c_str());
    if (::mkfifo(path.c_str(), 0600) != 0 && errno != EEXIST) {
        throw std::runtime_error(std::string("mkfifo ") + path + ": " + strerror(errno));
    }
}

// Open one end of a FIFO non-blocking. For pure read end we want O_RDONLY|O_NONBLOCK,
// but that requires the write end to be open; otherwise we'd EOF immediately on read.
// Trick: open O_RDWR|O_NONBLOCK on a FIFO in Linux is allowed and avoids EOF.
static int open_fifo(const std::string& path, bool for_write) {
    int flags = O_NONBLOCK | O_CLOEXEC;
    flags |= for_write ? O_WRONLY : O_RDWR;  // O_RDWR avoids EOF when no writer
    int fd = ::open(path.c_str(), flags);
    if (fd < 0) {
        // for_write may fail with ENXIO if no reader yet; retry by opening O_RDWR.
        if (for_write && (errno == ENXIO)) {
            fd = ::open(path.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
        }
        if (fd < 0)
            throw std::runtime_error(std::string("open fifo ") + path + ": " + strerror(errno));
    }
    return fd;
}

std::shared_ptr<Server> Server::create(const ServerConfig& cfg) {
    auto s = std::shared_ptr<Server>(new Server());
    s->cfg_ = cfg;
    s->pool_configs_ = cfg.pools;

    s->pools_.reserve(cfg.pools.size());
    for (size_t i = 0; i < cfg.pools.size(); ++i) {
        auto p = ShmPool::create(s->pool_name(static_cast<uint32_t>(i)),
                                 static_cast<uint32_t>(i),
                                 cfg.pools[i].first, cfg.pools[i].second, true);
        s->pools_.push_back(p);
    }

    s->epoll_fd_ = ::epoll_create1(EPOLL_CLOEXEC);
    if (s->epoll_fd_ < 0)
        throw std::runtime_error(std::string("epoll_create1: ") + strerror(errno));
    s->stop_ev_fd_ = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK | EFD_SEMAPHORE);
    if (s->stop_ev_fd_ < 0)
        throw std::runtime_error(std::string("eventfd stop: ") + strerror(errno));

    {
        epoll_event ev{};
        ev.events = EPOLLIN;
        ev.data.u64 = 0xFFFFFFFFu;
        if (::epoll_ctl(s->epoll_fd_, EPOLL_CTL_ADD, s->stop_ev_fd_, &ev) != 0)
            throw std::runtime_error(std::string("epoll_ctl stop: ") + strerror(errno));
    }

    s->clients_.resize(cfg.max_clients);
    for (uint32_t c = 0; c < cfg.max_clients; ++c) {
        auto cs = std::unique_ptr<ClientSlot>(new ClientSlot());
        cs->req_ring = RingShm::create(s->request_ring_name(c),
                                       cfg.ring_capacity,
                                       sizeof(PutRequestPOD), true);
        cs->resp_ring = RingShm::create(s->response_ring_name(c),
                                        cfg.resp_capacity,
                                        sizeof(ResponsePOD), true);

        std::string req_fp = s->request_fifo_path(c);
        std::string resp_fp = s->response_fifo_path(c);
        ensure_fifo(req_fp);
        ensure_fifo(resp_fp);

        cs->req_fifo_read_fd = open_fifo(req_fp, /*for_write=*/false);
        cs->resp_fifo_write_fd = open_fifo(resp_fp, /*for_write=*/true);

        epoll_event ev{};
        ev.events = EPOLLIN;
        ev.data.u64 = static_cast<uint64_t>(c);
        if (::epoll_ctl(s->epoll_fd_, EPOLL_CTL_ADD, cs->req_fifo_read_fd, &ev) != 0)
            throw std::runtime_error("epoll_ctl add req_fifo");
        s->clients_[c] = std::move(cs);
    }

    return s;
}

Server::~Server() {
    stop();
    for (auto& cptr : clients_) {
        if (!cptr) continue;
        if (cptr->req_fifo_read_fd >= 0) ::close(cptr->req_fifo_read_fd);
        if (cptr->resp_fifo_write_fd >= 0) ::close(cptr->resp_fifo_write_fd);
    }
    if (epoll_fd_ >= 0) ::close(epoll_fd_);
    if (stop_ev_fd_ >= 0) ::close(stop_ev_fd_);

    for (uint32_t c = 0; c < clients_.size(); ++c) {
        ::shm_unlink(request_ring_name(c).c_str());
        ::shm_unlink(response_ring_name(c).c_str());
        ::unlink(request_fifo_path(c).c_str());
        ::unlink(response_fifo_path(c).c_str());
    }
    for (size_t i = 0; i < pools_.size(); ++i) {
        ::shm_unlink(pool_name(static_cast<uint32_t>(i)).c_str());
    }
}

void Server::start() {
    stop_.store(false);
    workers_.clear();
    for (uint32_t i = 0; i < cfg_.num_workers; ++i) {
        workers_.emplace_back([this, i]{ this->worker_main(i); });
    }
}

void Server::stop() {
    if (!stop_.exchange(true)) {
        uint64_t one = 1;
        if (stop_ev_fd_ >= 0) {
            for (uint32_t i = 0; i < cfg_.num_workers; ++i) {
                ::write(stop_ev_fd_, &one, sizeof(one));
            }
        }
        deliver_cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
        workers_.clear();
    }
}

void Server::drain_ring(uint32_t client_id) {
    char buf[64];
    while (::read(clients_[client_id]->req_fifo_read_fd, buf, sizeof(buf)) > 0) {
        // discard
    }

    auto* h = clients_[client_id]->req_ring->header();
    int popped = 0;
    while (true) {
        PutRequestPOD pod;
        if (!ring_try_pop(h, pod)) break;
        popped++;

        if (cfg_.auto_ack) {
            // Fast path: ack immediately and release slot refcounts.
            // We bypass deliver queue entirely. Note: no lock needed for
            // ring_push from a worker thread because only one worker handles
            // a given client's resp ring at a time (we still grab resp_mtx to
            // be safe with spin-iters path that can interleave clients).
            ResponsePOD resp{};
            resp.magic = FAST_MAGIC;
            resp.resp_type = 0;
            resp.status = 0;
            resp.dp_client_id = client_id;
            resp.task_id = pod.task_id;
            resp.mask.nbytes = 0;

            auto& cs = *clients_[client_id];
            {
                std::lock_guard<std::mutex> g(cs.resp_mtx);
                bool need_notify = false;
                auto* rh = cs.resp_ring->header();
                for (int i = 0; i < 1 << 20; ++i) {
                    if (ring_push(rh, resp, &need_notify)) {
                        if (need_notify) {
                            char b = 1;
                            ::write(cs.resp_fifo_write_fd, &b, 1);
                        }
                        break;
                    }
#if defined(__x86_64__)
                    __builtin_ia32_pause();
#endif
                }
            }
            // Release all slots referenced by this request.
            auto free_ref = [&](const ArrayRef& a) {
                if (a.nbytes == 0 || a.pool_id >= pools_.size()) return;
                pools_[a.pool_id]->unref(static_cast<int32_t>(a.slot_id));
            };
            free_ref(pod.token_ids);
            free_ref(pod.slot_mapping);
            if (pod.flags & FLAG_HAS_MASK) free_ref(pod.token_mask);
            continue;
        }

        DeliveredRequest d;
        d.pod = pod;

        auto fill = [&](const ArrayRef& a,
                        void*& p, size_t& n, uint16_t& dt, size_t& ni) {
            if (a.nbytes == 0 || a.pool_id >= pools_.size()) {
                p = nullptr; n = 0; dt = 0; ni = 0; return;
            }
            auto& pool = pools_[a.pool_id];
            pool->ref(static_cast<int32_t>(a.slot_id));
            p = reinterpret_cast<uint8_t*>(pool->slot_ptr(a.slot_id)) + a.offset;
            n = a.nbytes;
            dt = a.dtype;
            ni = a.nbytes / (dtype_itemsize(a.dtype) ? dtype_itemsize(a.dtype) : 1);
        };
        fill(pod.token_ids,    d.token_ids_ptr,    d.token_ids_nbytes,    d.token_ids_dtype,    d.token_ids_nitems);
        fill(pod.slot_mapping, d.slot_mapping_ptr, d.slot_mapping_nbytes, d.slot_mapping_dtype, d.slot_mapping_nitems);
        if (pod.flags & FLAG_HAS_MASK) {
            fill(pod.token_mask, d.token_mask_ptr, d.token_mask_nbytes, d.token_mask_dtype, d.token_mask_nitems);
        }

        {
            std::lock_guard<std::mutex> g(deliver_mtx_);
            deliver_q_.push(std::move(d));
        }
        deliver_cv_.notify_one();
    }
    (void)popped;
}

void Server::worker_main(int /*worker_id*/) {
    epoll_event events[64];
    while (!stop_.load(std::memory_order_relaxed)) {
        if (cfg_.spin_iters > 0) {
            int popped_any = 0;
            for (int it = 0; it < cfg_.spin_iters; ++it) {
                for (uint32_t c = 0; c < clients_.size(); ++c) {
                    auto* h = clients_[c]->req_ring->header();
                    if (ring_size(h) > 0) {
                        drain_ring(c);
                        popped_any = 1;
                        break;
                    }
                }
                if (popped_any) break;
#if defined(__x86_64__)
                __builtin_ia32_pause();
#endif
            }
            if (popped_any) continue;
        }

        int n = ::epoll_wait(epoll_fd_, events, 64, -1);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("epoll_wait");
            break;
        }
        bool saw_stop = false;
        for (int i = 0; i < n; ++i) {
            uint64_t tag = events[i].data.u64;
            if (tag == 0xFFFFFFFFu) {
                uint64_t v; ::read(stop_ev_fd_, &v, sizeof(v));
                saw_stop = true;
                continue;
            }
            uint32_t client_id = static_cast<uint32_t>(tag);
            drain_ring(client_id);
        }
        if (saw_stop || stop_.load(std::memory_order_relaxed)) return;
    }
}

bool Server::pull(DeliveredRequest& out, int timeout_ms) {
    std::unique_lock<std::mutex> lk(deliver_mtx_);
    if (deliver_q_.empty()) {
        if (timeout_ms < 0) {
            deliver_cv_.wait(lk, [this]{ return !deliver_q_.empty() || stop_.load(); });
        } else {
            deliver_cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                                 [this]{ return !deliver_q_.empty() || stop_.load(); });
        }
    }
    if (deliver_q_.empty()) return false;
    out = std::move(deliver_q_.front());
    deliver_q_.pop();
    return true;
}

void Server::release_request(const DeliveredRequest& req) {
    auto unref_if = [&](const ArrayRef& a) {
        if (a.nbytes == 0 || a.pool_id >= pools_.size()) return;
        pools_[a.pool_id]->unref(static_cast<int32_t>(a.slot_id));
    };
    unref_if(req.pod.token_ids);
    unref_if(req.pod.slot_mapping);
    if (req.pod.flags & FLAG_HAS_MASK) unref_if(req.pod.token_mask);
}

std::shared_ptr<ShmPool> Server::pool_for_size(size_t nbytes) {
    for (auto& p : pools_) {
        if (nbytes <= p->slot_size()) return p;
    }
    return nullptr;
}

bool Server::ack(uint32_t dp_client_id, int64_t task_id, int32_t status,
                 const void* mask_ptr, size_t mask_nbytes, uint16_t mask_dtype) {
    if (dp_client_id >= clients_.size()) return false;
    auto& cs = *clients_[dp_client_id];
    ResponsePOD resp{};
    resp.magic = FAST_MAGIC;
    resp.resp_type = 0;
    resp.status = static_cast<uint16_t>(status);
    resp.dp_client_id = dp_client_id;
    resp.task_id = task_id;
    resp.mask.nbytes = 0;
    if (mask_ptr && mask_nbytes > 0) {
        auto p = pool_for_size(mask_nbytes);
        if (!p) return false;
        int32_t slot = p->alloc_slot();
        if (slot < 0) return false;
        std::memcpy(p->slot_ptr(slot), mask_ptr, mask_nbytes);
        resp.mask.pool_id = p->pool_id();
        resp.mask.slot_id = static_cast<uint32_t>(slot);
        resp.mask.offset  = 0;
        resp.mask.nbytes  = static_cast<uint32_t>(mask_nbytes);
        resp.mask.dtype   = mask_dtype;
        resp.mask.ndim    = 1;
        resp.mask.shape0  = static_cast<uint32_t>(mask_nbytes / (dtype_itemsize(mask_dtype) ? dtype_itemsize(mask_dtype) : 1));
    }

    std::lock_guard<std::mutex> g(cs.resp_mtx);
    bool need_notify = false;
    auto* h = cs.resp_ring->header();
    for (int i = 0; i < 1 << 20; ++i) {
        if (ring_push(h, resp, &need_notify)) {
            if (need_notify) {
                char b = 1;
                ::write(cs.resp_fifo_write_fd, &b, 1);
            }
            return true;
        }
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
    }
    return false;
}

} // namespace fastipc
