// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <string>
#include <queue>
#include <functional>

#include "fastipc/pod.hpp"
#include "fastipc/ring_shm.hpp"
#include "fastipc/shm_pool.hpp"

namespace fastipc {

struct ServerConfig {
    std::string shm_prefix      = "fipc";
    std::string fifo_dir        = "/tmp";
    uint32_t    max_clients     = 16;
    uint32_t    ring_capacity   = 1024;
    uint32_t    resp_capacity   = 1024;
    uint32_t    num_workers     = 4;
    std::vector<std::pair<uint32_t,uint32_t>> pools = {
        {   512u * 1024u, 1024u },
        { 4u * 1024u * 1024u,  256u },
        { 16u * 1024u * 1024u,  32u },
    };
    int spin_iters = 0;
    // When true, worker threads auto-ack each pulled request and immediately
    // release the shm slots. Delivery queue is skipped entirely. This is
    // useful for benchmarking and simple echo handlers.
    bool auto_ack = false;
};

// A delivered request, with zero-copy pointers into shm slots.
struct DeliveredRequest {
    PutRequestPOD pod;
    void*  token_ids_ptr = nullptr;
    size_t token_ids_nbytes = 0;
    uint16_t token_ids_dtype = 0;
    size_t token_ids_nitems = 0;

    void*  slot_mapping_ptr = nullptr;
    size_t slot_mapping_nbytes = 0;
    uint16_t slot_mapping_dtype = 0;
    size_t slot_mapping_nitems = 0;

    void*  token_mask_ptr = nullptr;
    size_t token_mask_nbytes = 0;
    uint16_t token_mask_dtype = 0;
    size_t token_mask_nitems = 0;
};

class Server {
public:
    static std::shared_ptr<Server> create(const ServerConfig& cfg);
    ~Server();

    void start();
    void stop();

    bool pull(DeliveredRequest& out, int timeout_ms);
    void release_request(const DeliveredRequest& req);

    bool ack(uint32_t dp_client_id, int64_t task_id, int32_t status,
             const void* mask_ptr = nullptr, size_t mask_nbytes = 0,
             uint16_t mask_dtype = 0);

    const std::vector<std::pair<uint32_t,uint32_t>>& pool_configs() const { return pool_configs_; }

    // Name resolution helpers.
    std::string request_ring_name(uint32_t client_id) const;
    std::string response_ring_name(uint32_t client_id) const;
    std::string pool_name(uint32_t pool_id) const;
    std::string request_fifo_path(uint32_t client_id) const;
    std::string response_fifo_path(uint32_t client_id) const;

private:
    Server() = default;

    struct ClientSlot {
        std::shared_ptr<RingShm> req_ring;
        std::shared_ptr<RingShm> resp_ring;
        int req_fifo_read_fd = -1;
        int resp_fifo_write_fd = -1;
        std::mutex resp_mtx;
    };

    void worker_main(int worker_id);
    void drain_ring(uint32_t client_id);
    std::shared_ptr<ShmPool> pool_for_size(size_t nbytes);

    ServerConfig cfg_;
    std::vector<std::unique_ptr<ClientSlot>> clients_;
    std::vector<std::shared_ptr<ShmPool>> pools_;
    std::vector<std::pair<uint32_t,uint32_t>> pool_configs_;

    int epoll_fd_ = -1;
    int stop_ev_fd_ = -1;
    std::atomic<bool> stop_{false};
    std::vector<std::thread> workers_;

    std::mutex deliver_mtx_;
    std::condition_variable deliver_cv_;
    std::queue<DeliveredRequest> deliver_q_;
};

} // namespace fastipc
