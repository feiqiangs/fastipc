// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>

#include "fastipc/pod.hpp"
#include "fastipc/shm_pool.hpp"
#include "fastipc/ring_shm.hpp"

namespace fastipc {

struct ClientConfig {
    std::string shm_prefix;
    std::string fifo_dir = "/tmp";
    uint32_t    dp_client_id;
    std::vector<std::pair<uint32_t,uint32_t>> pools;
};

struct PullResult {
    ResponsePOD pod;
    void*    mask_ptr = nullptr;
    size_t   mask_nbytes = 0;
    uint16_t mask_dtype = 0;
    size_t   mask_nitems = 0;
};

// A handle to a pre-allocated shm slot for zero-copy writes.
struct ShmBuffer {
    std::shared_ptr<ShmPool> pool;
    int32_t  slot_id = -1;
    void*    ptr = nullptr;
    size_t   capacity = 0;
    uint32_t pool_id = 0;
};

class Client {
public:
    static std::shared_ptr<Client> create(const ClientConfig& cfg);
    ~Client();

    int64_t push_put(
        const void* token_ids_ptr, size_t token_ids_nbytes, uint16_t tid_dtype,
        const void* slot_mapping_ptr, size_t slot_mapping_nbytes, uint16_t sm_dtype,
        const void* token_mask_ptr, size_t token_mask_nbytes, uint16_t mask_dtype,
        int32_t layer_granularity, int32_t namespace_id, uint16_t req_type = REQ_PUT);

    // Zero-copy: allocate a buffer directly in shm. The caller writes data into
    // the returned pointer, then passes the ShmBuffer to push_put_prealloc.
    ShmBuffer alloc_shm_buffer(size_t nbytes);

    // Zero-copy push: the caller has already written data into ShmBuffers
    // obtained from alloc_shm_buffer. No memcpy occurs.
    int64_t push_put_prealloc(
        const ShmBuffer& token_ids_buf, size_t token_ids_nbytes, uint16_t tid_dtype,
        const ShmBuffer* slot_mapping_buf, size_t slot_mapping_nbytes, uint16_t sm_dtype,
        const ShmBuffer* token_mask_buf, size_t token_mask_nbytes, uint16_t mask_dtype,
        int32_t layer_granularity, int32_t namespace_id, uint16_t req_type = REQ_PUT);

    // Release a ShmBuffer without pushing (e.g. on error).
    void release_shm_buffer(ShmBuffer& buf);

    bool pull(PullResult& out, int timeout_ms);
    void release_pull_result(const PullResult& r);

private:
    Client() = default;
    std::shared_ptr<ShmPool> pool_for_size(size_t nbytes);
    std::string request_ring_name(uint32_t cid) const;
    std::string response_ring_name(uint32_t cid) const;
    std::string pool_name(uint32_t pool_id) const;
    std::string request_fifo_path(uint32_t cid) const;
    std::string response_fifo_path(uint32_t cid) const;

    ClientConfig cfg_;
    std::vector<std::shared_ptr<ShmPool>> pools_;
    std::shared_ptr<RingShm> req_ring_;
    std::shared_ptr<RingShm> resp_ring_;
    int epoll_fd_ = -1;
    int req_fifo_write_fd_ = -1;   // client writes (notify server)
    int resp_fifo_read_fd_ = -1;   // client reads (server notifies us)

    std::atomic<int64_t> task_counter_{0};
    int64_t task_id_base_ = 0;
};

} // namespace fastipc
