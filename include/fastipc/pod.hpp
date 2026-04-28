// SPDX-License-Identifier: Apache-2.0
// FastIPC POD message format (fixed-size, no pickle, no serialization).
#pragma once

#include <cstdint>
#include <cstddef>

namespace fastipc {

constexpr uint32_t FAST_MAGIC = 0x46415354u;  // "FAST"

enum ReqType : uint16_t {
    REQ_PUT       = 1,
    REQ_GET       = 2,
    REQ_PUT_MATCH = 3,
    REQ_GET_MATCH = 4,
    REQ_PREFETCH  = 5,
    REQ_SHUTDOWN  = 99,
};

enum ReqFlag : uint16_t {
    FLAG_HAS_MASK     = 1u << 0,
    FLAG_AS_BATCH     = 1u << 1,
    FLAG_HAS_SLOT_MAP = 1u << 2,
};

// Numpy dtype code (stable, we pick our own).
enum DTypeCode : uint16_t {
    DT_INVALID = 0,
    DT_INT8    = 1,
    DT_UINT8   = 2,
    DT_INT16   = 3,
    DT_UINT16  = 4,
    DT_INT32   = 5,
    DT_UINT32  = 6,
    DT_INT64   = 7,
    DT_UINT64  = 8,
    DT_FLOAT16 = 9,
    DT_FLOAT32 = 10,
    DT_FLOAT64 = 11,
    DT_BOOL    = 12,
};

inline size_t dtype_itemsize(uint16_t code) {
    switch (code) {
        case DT_INT8: case DT_UINT8: case DT_BOOL: return 1;
        case DT_INT16: case DT_UINT16: case DT_FLOAT16: return 2;
        case DT_INT32: case DT_UINT32: case DT_FLOAT32: return 4;
        case DT_INT64: case DT_UINT64: case DT_FLOAT64: return 8;
        default: return 0;
    }
}

// A single ndarray reference in shm.
struct ArrayRef {
    uint32_t pool_id;
    uint32_t slot_id;
    uint32_t offset;     // byte offset within slot
    uint32_t nbytes;
    uint16_t dtype;
    uint16_t ndim;       // currently only 1D is supported
    uint32_t shape0;
};
static_assert(sizeof(ArrayRef) == 24, "ArrayRef must be 24 bytes");

#pragma pack(push, 1)

// Fixed-size request sent through a SPSC ring. No serialization.
struct PutRequestPOD {
    uint32_t magic;               //  0
    uint16_t req_type;            //  4
    uint16_t flags;               //  6
    uint32_t dp_client_id;        //  8
    int32_t  namespace_id;        // 12
    int64_t  task_id;             // 16
    int32_t  layer_granularity;   // 24
    uint32_t _pad0;               // 28
    ArrayRef token_ids;           // 32 .. 55
    ArrayRef slot_mapping;        // 56 .. 79
    ArrayRef token_mask;          // 80 .. 103
};

struct ResponsePOD {
    uint32_t magic;               //  0
    uint16_t resp_type;           //  4
    uint16_t status;              //  6
    uint32_t dp_client_id;        //  8
    uint32_t _pad0;               // 12
    int64_t  task_id;             // 16
    ArrayRef mask;                // 24 .. 47 (optional, nbytes==0 means none)
    uint64_t _rsvd[2];            // 48 .. 63
};

#pragma pack(pop)

static_assert(sizeof(PutRequestPOD) == 104, "PutRequestPOD must be 104 bytes");
static_assert(sizeof(ResponsePOD) == 64, "ResponsePOD must be 64 bytes");

} // namespace fastipc
