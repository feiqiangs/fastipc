// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

#include "fastipc/server.hpp"
#include "fastipc/client.hpp"

using namespace fastipc;

static void basic_echo_test() {
    printf("[selftest] basic echo: start\n");
    ServerConfig cfg;
    cfg.shm_prefix  = "fipc_selftest";
    cfg.max_clients = 2;
    cfg.num_workers = 2;
    cfg.ring_capacity = 64;
    cfg.resp_capacity = 64;
    cfg.pools = {
        {64 * 1024u, 128u},
        {1u * 1024u * 1024u, 32u},
        {8u * 1024u * 1024u, 8u},
    };

    auto srv = Server::create(cfg);
    srv->start();

    std::atomic<int> processed{0};
    std::atomic<bool> stop{false};
    std::thread consumer([&]{
        while (!stop.load()) {
            DeliveredRequest req;
            if (!srv->pull(req, 200)) continue;
            srv->ack(req.pod.dp_client_id, req.pod.task_id, 0);
            srv->release_request(req);
            processed.fetch_add(1);
        }
    });

    ClientConfig ccfg;
    ccfg.shm_prefix  = "fipc_selftest";
    ccfg.dp_client_id = 0;
    ccfg.pools = cfg.pools;
    auto cli = Client::create(ccfg);

    std::vector<int64_t> token_ids(1024);
    for (size_t i = 0; i < token_ids.size(); ++i) token_ids[i] = static_cast<int64_t>(i);

    const int N = 100;
    for (int i = 0; i < N; ++i) {
        int64_t tid = cli->push_put(
            token_ids.data(), token_ids.size() * sizeof(int64_t), DT_INT64,
            nullptr, 0, 0, nullptr, 0, 0, -1, 0);
        PullResult r;
        bool ok = cli->pull(r, 2000);
        if (!ok || r.pod.task_id != tid) {
            fprintf(stderr, "basic echo: mismatch (iter %d): ok=%d got=%lld want=%lld\n",
                    i, (int)ok, (long long)r.pod.task_id, (long long)tid);
            std::abort();
        }
    }

    stop.store(true);
    consumer.join();
    srv->stop();
    if (processed.load() < N) {
        fprintf(stderr, "basic echo: processed=%d < %d\n", processed.load(), N);
        std::abort();
    }
    printf("[selftest] basic echo: OK (%d round-trips)\n", N);
}

static void large_payload_test() {
    printf("[selftest] large payload: start\n");
    ServerConfig cfg;
    cfg.shm_prefix  = "fipc_selftest2";
    cfg.max_clients = 1;
    cfg.num_workers = 1;
    cfg.ring_capacity = 16;
    cfg.resp_capacity = 16;
    cfg.pools = {
        {8u * 1024u * 1024u, 8u},
    };
    auto srv = Server::create(cfg);
    srv->start();

    std::atomic<bool> stop{false};
    std::thread consumer([&]{
        while (!stop.load()) {
            DeliveredRequest req;
            if (!srv->pull(req, 200)) continue;
            int64_t* p = static_cast<int64_t*>(req.token_ids_ptr);
            size_t n = req.token_ids_nitems;
            if (p[0] != 0 || p[n-1] != static_cast<int64_t>(n-1)) {
                fprintf(stderr, "large: data corrupted! p[0]=%lld p[%zu]=%lld\n",
                        (long long)p[0], n-1, (long long)p[n-1]);
                std::abort();
            }
            srv->ack(req.pod.dp_client_id, req.pod.task_id, 0);
            srv->release_request(req);
        }
    });

    ClientConfig ccfg;
    ccfg.shm_prefix = "fipc_selftest2";
    ccfg.dp_client_id = 0;
    ccfg.pools = cfg.pools;
    auto cli = Client::create(ccfg);

    const size_t N = 512 * 1024;
    std::vector<int64_t> big(N);
    for (size_t i = 0; i < N; ++i) big[i] = static_cast<int64_t>(i);

    for (int i = 0; i < 5; ++i) {
        int64_t tid = cli->push_put(
            big.data(), N * sizeof(int64_t), DT_INT64,
            nullptr, 0, 0, nullptr, 0, 0, -1, 0);
        PullResult r;
        bool ok = cli->pull(r, 3000);
        if (!ok || r.pod.task_id != tid) {
            fprintf(stderr, "large: mismatch\n");
            std::abort();
        }
    }

    stop.store(true);
    consumer.join();
    srv->stop();
    printf("[selftest] large payload (512K tokens): OK\n");
}

int main() {
    try {
        basic_echo_test();
        large_payload_test();
        printf("[selftest] all tests passed.\n");
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[selftest] exception: %s\n", e.what());
        return 1;
    }
}
