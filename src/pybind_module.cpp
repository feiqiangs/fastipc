// SPDX-License-Identifier: Apache-2.0
// Python bindings for fastipc. All heavy work runs without holding the GIL.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "fastipc/server.hpp"
#include "fastipc/client.hpp"

namespace py = pybind11;
using namespace fastipc;

// ---- numpy dtype <-> our code ----------------------------------------------
static uint16_t np_to_code(const py::dtype& dt) {
    int num = dt.num();
    switch (num) {
        case 1:  return DT_INT8;
        case 2:  return DT_UINT8;
        case 3:  return DT_INT16;
        case 4:  return DT_UINT16;
        case 5:  return DT_INT32;
        case 6:  return DT_UINT32;
        case 7:  return DT_INT64;      // NPY_LONG on 64-bit linux
        case 8:  return DT_UINT64;
        case 9:  return DT_INT64;      // NPY_LONGLONG
        case 10: return DT_UINT64;
        case 11: return DT_FLOAT32;
        case 12: return DT_FLOAT64;
        case 23: return DT_FLOAT16;
        case 0:  return DT_BOOL;
        default: return DT_UINT8;
    }
}

static py::dtype code_to_np(uint16_t c) {
    switch (c) {
        case DT_INT8:    return py::dtype("int8");
        case DT_UINT8:   return py::dtype("uint8");
        case DT_INT16:   return py::dtype("int16");
        case DT_UINT16:  return py::dtype("uint16");
        case DT_INT32:   return py::dtype("int32");
        case DT_UINT32:  return py::dtype("uint32");
        case DT_INT64:   return py::dtype("int64");
        case DT_UINT64:  return py::dtype("uint64");
        case DT_FLOAT16: return py::dtype("float16");
        case DT_FLOAT32: return py::dtype("float32");
        case DT_FLOAT64: return py::dtype("float64");
        case DT_BOOL:    return py::dtype("bool_");
        default:         return py::dtype("uint8");
    }
}

// ---- Python-facing helpers -------------------------------------------------

// Build a numpy zero-copy view over a raw pointer. The view "owns" the slot
// through a capsule that drops a refcount on destruction.
static py::object make_view_or_none(void* ptr, size_t nbytes, uint16_t dtype_code,
                                    std::shared_ptr<Server> /*keep_alive*/,
                                    std::function<void()> release_cb)
{
    if (!ptr || nbytes == 0) return py::none();
    py::dtype dt = code_to_np(dtype_code);
    size_t item = dtype_itemsize(dtype_code);
    if (item == 0) item = 1;
    py::ssize_t nitems = static_cast<py::ssize_t>(nbytes / item);
    auto* cb = new std::function<void()>(std::move(release_cb));
    py::capsule deleter(cb, [](void* p){
        auto* f = static_cast<std::function<void()>*>(p);
        (*f)();
        delete f;
    });
    return py::array(dt, {nitems}, {static_cast<py::ssize_t>(item)}, ptr, deleter);
}

PYBIND11_MODULE(_fastipc, m) {
    m.doc() = "fastipc: zero-copy shm+ring IPC for Linux";

    py::class_<Server, std::shared_ptr<Server>>(m, "Server")
        .def_static("create", [](const std::string& shm_prefix,
                                 uint32_t max_clients,
                                 uint32_t ring_capacity,
                                 uint32_t resp_capacity,
                                 uint32_t num_workers,
                                 int spin_iters,
                                 bool auto_ack,
                                 const std::vector<std::pair<uint32_t,uint32_t>>& pools) {
            ServerConfig cfg;
            cfg.shm_prefix    = shm_prefix;
            cfg.max_clients   = max_clients;
            cfg.ring_capacity = ring_capacity;
            cfg.resp_capacity = resp_capacity;
            cfg.num_workers   = num_workers;
            cfg.spin_iters    = spin_iters;
            cfg.auto_ack      = auto_ack;
            if (!pools.empty()) cfg.pools = pools;
            return Server::create(cfg);
        },
        py::arg("shm_prefix") = "fipc",
        py::arg("max_clients") = 16,
        py::arg("ring_capacity") = 1024,
        py::arg("resp_capacity") = 1024,
        py::arg("num_workers") = 4,
        py::arg("spin_iters") = 0,
        py::arg("auto_ack") = false,
        py::arg("pools") = std::vector<std::pair<uint32_t,uint32_t>>{})
        .def("start", &Server::start)
        .def("stop",  &Server::stop)
        .def("pool_configs", &Server::pool_configs)

        // pull: blocks until one request is available, returns a dict with
        // zero-copy numpy views. The server keeps refs on the shm slots; the
        // returned views carry a capsule that auto-releases those refs when
        // the view is GC'ed.
        .def("pull", [](std::shared_ptr<Server> self, int timeout_ms) -> py::object {
            DeliveredRequest d;
            bool ok;
            {
                py::gil_scoped_release nogil;
                ok = self->pull(d, timeout_ms);
            }
            if (!ok) return py::none();

            // Build views. Each ArrayRef was ref'ed on drain; we transfer each
            // ref into a capsule so numpy lifetime controls release.
            py::dict out;
            out["task_id"]      = d.pod.task_id;
            out["dp_client_id"] = d.pod.dp_client_id;
            out["req_type"]     = d.pod.req_type;
            out["layer_granularity"] = d.pod.layer_granularity;
            out["namespace_id"] = d.pod.namespace_id;

            // Build per-field release callbacks using the Server's
            // release_request — but since that releases ALL refs at once, we
            // instead split the release into per-field callbacks by hand.
            // We re-construct a tiny DeliveredRequest with only one field per
            // callback and call release_request() on it.
            auto make_unref_cb = [self](ArrayRef a) -> std::function<void()> {
                DeliveredRequest stub;
                stub.pod.flags = 0;  // default: no mask
                stub.pod.token_ids = ArrayRef{0,0,0,0,0,0,0};
                stub.pod.slot_mapping = ArrayRef{0,0,0,0,0,0,0};
                stub.pod.token_mask = ArrayRef{0,0,0,0,0,0,0};
                // We just need to unref ONE field — put it in token_ids slot.
                stub.pod.token_ids = a;
                return [self, stub]() mutable { self->release_request(stub); };
            };

            out["token_ids"] = make_view_or_none(
                d.token_ids_ptr, d.token_ids_nbytes, d.token_ids_dtype,
                self, make_unref_cb(d.pod.token_ids));
            out["slot_mapping"] = make_view_or_none(
                d.slot_mapping_ptr, d.slot_mapping_nbytes, d.slot_mapping_dtype,
                self, make_unref_cb(d.pod.slot_mapping));
            if (d.pod.flags & FLAG_HAS_MASK) {
                out["token_mask"] = make_view_or_none(
                    d.token_mask_ptr, d.token_mask_nbytes, d.token_mask_dtype,
                    self, make_unref_cb(d.pod.token_mask));
            } else {
                out["token_mask"] = py::none();
            }
            return out;
        }, py::arg("timeout_ms") = -1)

        .def("ack", [](Server& self,
                       uint32_t dp_client_id, int64_t task_id, int32_t status,
                       py::object mask) -> bool {
            if (mask.is_none()) {
                bool ok;
                {
                    py::gil_scoped_release nogil;
                    ok = self.ack(dp_client_id, task_id, status, nullptr, 0, 0);
                }
                return ok;
            }
            auto arr = py::cast<py::array>(mask);
            uint16_t dt = np_to_code(arr.dtype());
            py::buffer_info bi = arr.request();
            void* ptr = bi.ptr;
            size_t nb = static_cast<size_t>(bi.size * bi.itemsize);
            bool ok;
            {
                py::gil_scoped_release nogil;
                ok = self.ack(dp_client_id, task_id, status, ptr, nb, dt);
            }
            return ok;
        }, py::arg("dp_client_id"), py::arg("task_id"),
           py::arg("status") = 0, py::arg("mask") = py::none())
        ;

    py::class_<Client, std::shared_ptr<Client>>(m, "Client")
        .def_static("create", [](const std::string& shm_prefix,
                                 uint32_t dp_client_id,
                                 const std::vector<std::pair<uint32_t,uint32_t>>& pools) {
            ClientConfig cfg;
            cfg.shm_prefix = shm_prefix;
            cfg.dp_client_id = dp_client_id;
            cfg.pools = pools;
            return Client::create(cfg);
        },
        py::arg("shm_prefix"), py::arg("dp_client_id"), py::arg("pools"))

        .def("push_put", [](Client& self,
                            py::array token_ids,
                            py::object slot_mapping,
                            py::object token_mask,
                            int32_t layer_granularity,
                            int32_t namespace_id) -> int64_t {
            py::buffer_info ti = token_ids.request();
            uint16_t ti_dt = np_to_code(token_ids.dtype());

            const void* sm_ptr = nullptr; size_t sm_nb = 0; uint16_t sm_dt = 0;
            std::unique_ptr<py::buffer_info> sm_bi;
            if (!slot_mapping.is_none()) {
                auto arr = py::cast<py::array>(slot_mapping);
                sm_bi.reset(new py::buffer_info(arr.request()));
                sm_ptr = sm_bi->ptr;
                sm_nb = static_cast<size_t>(sm_bi->size * sm_bi->itemsize);
                sm_dt = np_to_code(arr.dtype());
            }
            const void* mk_ptr = nullptr; size_t mk_nb = 0; uint16_t mk_dt = 0;
            std::unique_ptr<py::buffer_info> mk_bi;
            if (!token_mask.is_none()) {
                auto arr = py::cast<py::array>(token_mask);
                mk_bi.reset(new py::buffer_info(arr.request()));
                mk_ptr = mk_bi->ptr;
                mk_nb = static_cast<size_t>(mk_bi->size * mk_bi->itemsize);
                mk_dt = np_to_code(arr.dtype());
            }

            int64_t tid;
            {
                py::gil_scoped_release nogil;
                tid = self.push_put(
                    ti.ptr, static_cast<size_t>(ti.size * ti.itemsize), ti_dt,
                    sm_ptr, sm_nb, sm_dt,
                    mk_ptr, mk_nb, mk_dt,
                    layer_granularity, namespace_id);
            }
            return tid;
        },
        py::arg("token_ids"),
        py::arg("slot_mapping") = py::none(),
        py::arg("token_mask") = py::none(),
        py::arg("layer_granularity") = -1,
        py::arg("namespace_id") = 0)

        // alloc_array: allocate an ndarray whose data buffer lives directly in shm.
        // The caller writes into this array, then passes it to push_put_zerocopy.
        .def("alloc_array", [](std::shared_ptr<Client> self,
                               py::ssize_t nitems, py::dtype dt) -> py::array {
            size_t itemsize = dt.itemsize();
            size_t nbytes = static_cast<size_t>(nitems) * itemsize;
            ShmBuffer buf;
            {
                py::gil_scoped_release nogil;
                buf = self->alloc_shm_buffer(nbytes);
            }
            // Wrap in capsule: when the ndarray is GC'ed, unref the slot.
            auto* info = new ShmBuffer(buf);
            py::capsule cap(info, [](void* p) {
                auto* b = static_cast<ShmBuffer*>(p);
                if (b->pool && b->slot_id >= 0) {
                    b->pool->unref(b->slot_id);
                }
                delete b;
            });
            return py::array(dt, {nitems}, {static_cast<py::ssize_t>(itemsize)},
                             buf.ptr, cap);
        }, py::arg("nitems"), py::arg("dtype"),
        "Allocate an ndarray directly in shared memory (zero-copy).")

        // push_put_zerocopy: push pre-allocated shm arrays. NO memcpy.
        .def("push_put_zerocopy", [](Client& self,
                                     py::array token_ids,
                                     py::object slot_mapping,
                                     py::object token_mask,
                                     int32_t layer_granularity,
                                     int32_t namespace_id) -> int64_t {
            // Extract ShmBuffer info from the capsule base of each array.
            auto extract_shm_buf = [](py::array& arr) -> ShmBuffer* {
                py::object base = arr.attr("base");
                if (base.is_none()) return nullptr;
                // The base should be a capsule created by alloc_array
                if (!py::isinstance<py::capsule>(base)) return nullptr;
                auto cap = py::cast<py::capsule>(base);
                return static_cast<ShmBuffer*>(static_cast<void*>(cap));
            };

            ShmBuffer* ti_buf = extract_shm_buf(token_ids);
            if (!ti_buf || ti_buf->slot_id < 0)
                throw std::runtime_error("token_ids is not a shm-allocated array (use alloc_array)");
            uint16_t ti_dt = np_to_code(token_ids.dtype());
            size_t ti_nb = static_cast<size_t>(token_ids.nbytes());

            ShmBuffer* sm_buf = nullptr;
            size_t sm_nb = 0; uint16_t sm_dt = 0;
            if (!slot_mapping.is_none()) {
                auto arr = py::cast<py::array>(slot_mapping);
                sm_buf = extract_shm_buf(arr);
                if (!sm_buf || sm_buf->slot_id < 0)
                    throw std::runtime_error("slot_mapping is not a shm-allocated array");
                sm_nb = static_cast<size_t>(arr.nbytes());
                sm_dt = np_to_code(arr.dtype());
            }

            ShmBuffer* mk_buf = nullptr;
            size_t mk_nb = 0; uint16_t mk_dt = 0;
            if (!token_mask.is_none()) {
                auto arr = py::cast<py::array>(token_mask);
                mk_buf = extract_shm_buf(arr);
                if (!mk_buf || mk_buf->slot_id < 0)
                    throw std::runtime_error("token_mask is not a shm-allocated array");
                mk_nb = static_cast<size_t>(arr.nbytes());
                mk_dt = np_to_code(arr.dtype());
            }

            // After push, ownership of the slot transfers to the server.
            // We must NOT let the capsule's destructor unref the slot again,
            // so we invalidate the ShmBuffer inside the capsule.
            int64_t tid;
            {
                py::gil_scoped_release nogil;
                tid = self.push_put_prealloc(
                    *ti_buf, ti_nb, ti_dt,
                    sm_buf, sm_nb, sm_dt,
                    mk_buf, mk_nb, mk_dt,
                    layer_granularity, namespace_id);
            }
            // Invalidate capsules so GC won't double-unref.
            ti_buf->slot_id = -1; ti_buf->ptr = nullptr;
            if (sm_buf) { sm_buf->slot_id = -1; sm_buf->ptr = nullptr; }
            if (mk_buf) { mk_buf->slot_id = -1; mk_buf->ptr = nullptr; }
            return tid;
        },
        py::arg("token_ids"),
        py::arg("slot_mapping") = py::none(),
        py::arg("token_mask") = py::none(),
        py::arg("layer_granularity") = -1,
        py::arg("namespace_id") = 0,
        "Push a request using pre-allocated shm arrays (true zero-copy).")

        .def("pull", [](Client& self, int timeout_ms) -> py::object {
            PullResult r;
            bool ok;
            {
                py::gil_scoped_release nogil;
                ok = self.pull(r, timeout_ms);
            }
            if (!ok) return py::none();
            py::dict d;
            d["task_id"] = r.pod.task_id;
            d["status"]  = r.pod.status;
            d["dp_client_id"] = r.pod.dp_client_id;
            if (r.mask_nbytes > 0) {
                py::array arr(code_to_np(r.mask_dtype),
                              {static_cast<py::ssize_t>(r.mask_nitems)});
                std::memcpy(arr.mutable_data(), r.mask_ptr, r.mask_nbytes);
                d["mask"] = arr;
                self.release_pull_result(r);
            } else {
                d["mask"] = py::none();
            }
            return d;
        }, py::arg("timeout_ms") = -1)
        ;
}
