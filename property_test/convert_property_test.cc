/*
 *  Copyright 2026 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

// Randomized property test for the row-based YUV conversion functions.
//
// For each generated case the harness checks:
//   1. Guard bytes before and after every output buffer are untouched.
//   2. Bytes past the logical end of every output row are untouched.
//   3. The same input and CPU mask produce the same output twice.
//   4. Every optimized CPU mask is bit exact to the C reference.
//
// A failing case is shrunk (width, height, source/destination padding,
// color range and vertical flip) before it is printed.

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "libyuv/convert.h"
#include "libyuv/cpu_id.h"

using namespace libyuv;  // NOLINT

namespace {

constexpr size_t kGuardSize = 64;
constexpr uint8_t kGuardByte = 0xa5;
constexpr uint8_t kOutputFillA = 0x5a;
constexpr uint8_t kOutputFillB = 0x6b;
bool g_inject_mismatch = false;

uint64_t SplitMix64(uint64_t* state) {
  uint64_t z = (*state += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

class Rng {
 public:
  explicit Rng(uint64_t seed) : state_(seed) {}

  uint32_t Next() { return static_cast<uint32_t>(SplitMix64(&state_) >> 32); }
  uint64_t Next64() { return SplitMix64(&state_); }
  uint32_t Uniform(uint32_t max_inclusive) {
    if (max_inclusive == 0) {
      return 0;
    }
    return Next() % (max_inclusive + 1);
  }

 private:
  uint64_t state_;
};

class GuardedBuffer {
 public:
  GuardedBuffer() = default;
  explicit GuardedBuffer(size_t size) { Allocate(size); }
  GuardedBuffer(const GuardedBuffer&) = delete;
  GuardedBuffer& operator=(const GuardedBuffer&) = delete;
  GuardedBuffer(GuardedBuffer&& other) noexcept { Swap(other); }
  GuardedBuffer& operator=(GuardedBuffer&& other) noexcept {
    if (this != &other) {
      Free();
      Swap(other);
    }
    return *this;
  }
  ~GuardedBuffer() { Free(); }

  bool Allocate(size_t size) {
    Free();
    size_ = size;
    raw_ = static_cast<uint8_t*>(
        std::malloc(kGuardSize + size + kGuardSize));
    if (!raw_) {
      return false;
    }
    std::memset(raw_, kGuardByte, kGuardSize + size + kGuardSize);
    data_ = raw_ + kGuardSize;
    return true;
  }

  bool CheckGuard(bool before, size_t* bad_offset, uint8_t* actual) const {
    const uint8_t* begin = before ? data_ - kGuardSize : data_ + size_;
    for (size_t i = 0; i < kGuardSize; ++i) {
      if (begin[i] != kGuardByte) {
        if (bad_offset) {
          *bad_offset = i;
        }
        if (actual) {
          *actual = begin[i];
        }
        return false;
      }
    }
    return true;
  }

  uint8_t* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  void Swap(GuardedBuffer& other) {
    std::swap(raw_, other.raw_);
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }

  void Free() {
    std::free(raw_);
    raw_ = nullptr;
    data_ = nullptr;
    size_ = 0;
  }

  uint8_t* raw_ = nullptr;
  uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

struct Plane {
  GuardedBuffer buffer;
  int stride = 0;
  int row_bytes = 0;
  int rows = 0;

  Plane() = default;
  Plane(Plane&&) noexcept = default;
  Plane& operator=(Plane&&) noexcept = default;
  Plane(const Plane&) = delete;
  Plane& operator=(const Plane&) = delete;

  bool Allocate(int rows_in, int row_bytes_in, int pad) {
    rows = rows_in;
    row_bytes = row_bytes_in;
    stride = row_bytes_in + pad;
    return buffer.Allocate(static_cast<size_t>(stride) * rows);
  }

  uint8_t* Row(int y) const {
    return buffer.data() + static_cast<size_t>(y) * stride;
  }
};

struct Surface {
  int bpc = 1;
  bool semiplanar = false;
  Plane y;
  Plane u;
  Plane v;
  Plane uv;
};

struct PlaneView {
  uint8_t* data = nullptr;
  int stride = 0;
  int row_bytes = 0;
  int rows = 0;
};

struct SurfaceView {
  int bpc = 1;
  PlaneView y;
  PlaneView u;
  PlaneView v;
  PlaneView uv;
};

PlaneView MakePlaneView(const Plane& plane) {
  PlaneView view;
  view.data = plane.buffer.data();
  view.stride = plane.stride;
  view.row_bytes = plane.row_bytes;
  view.rows = plane.rows;
  return view;
}

SurfaceView MakeSurfaceView(const Surface& surface) {
  SurfaceView view;
  view.bpc = surface.bpc;
  view.y = MakePlaneView(surface.y);
  if (surface.semiplanar) {
    view.uv = MakePlaneView(surface.uv);
  } else {
    view.u = MakePlaneView(surface.u);
    view.v = MakePlaneView(surface.v);
  }
  return view;
}

struct Format {
  int bpc;
  int subsample_x;
  int subsample_y;
  bool semiplanar;
};

enum FunctionId {
  kI444ToI422,
  kI444ToI420,
  kI422ToI420,
  kI422ToNV12,
  kI422ToNV21,
  kI444ToNV12,
  kI444ToNV21,
  kNV24ToNV16,
  kNV16ToNV12,
  kNV24ToNV12,
  kNV16ToI420,
  kNV24ToI420,
  kI410ToI210,
  kI410ToI010,
  kI210ToI010,
  kI210ToI420,
  kI410ToI420,
  kI212ToI420,
  kI412ToI420,
  kP010ToI420,
  kP210ToI420,
  kP410ToI420,
  kP410ToP210,
  kP210ToP010,
  kP410ToP010,
  kFunctionCount
};

struct FunctionDesc {
  const char* name;
  Format src;
  Format dst;
  int depth;
};

const FunctionDesc kFunctions[kFunctionCount] = {
    {"I444ToI422", {1, 1, 1, false}, {1, 2, 1, false}, 8},
    {"I444ToI420", {1, 1, 1, false}, {1, 2, 2, false}, 8},
    {"I422ToI420", {1, 2, 1, false}, {1, 2, 2, false}, 8},
    {"I422ToNV12", {1, 2, 1, false}, {1, 2, 2, true}, 8},
    {"I422ToNV21", {1, 2, 1, false}, {1, 2, 2, true}, 8},
    {"I444ToNV12", {1, 1, 1, false}, {1, 2, 2, true}, 8},
    {"I444ToNV21", {1, 1, 1, false}, {1, 2, 2, true}, 8},
    {"NV24ToNV16", {1, 1, 1, true}, {1, 2, 1, true}, 8},
    {"NV16ToNV12", {1, 2, 1, true}, {1, 2, 2, true}, 8},
    {"NV24ToNV12", {1, 1, 1, true}, {1, 2, 2, true}, 8},
    {"NV16ToI420", {1, 2, 1, true}, {1, 2, 2, false}, 8},
    {"NV24ToI420", {1, 1, 1, true}, {1, 2, 2, false}, 8},
    {"I410ToI210", {2, 1, 1, false}, {2, 2, 1, false}, 10},
    {"I410ToI010", {2, 1, 1, false}, {2, 2, 2, false}, 10},
    {"I210ToI010", {2, 2, 1, false}, {2, 2, 2, false}, 10},
    {"I210ToI420", {2, 2, 1, false}, {1, 2, 2, false}, 10},
    {"I410ToI420", {2, 1, 1, false}, {1, 2, 2, false}, 10},
    {"I212ToI420", {2, 2, 1, false}, {1, 2, 2, false}, 12},
    {"I412ToI420", {2, 1, 1, false}, {1, 2, 2, false}, 12},
    {"P010ToI420", {2, 2, 2, true}, {1, 2, 2, false}, 10},
    {"P210ToI420", {2, 2, 1, true}, {1, 2, 2, false}, 10},
    {"P410ToI420", {2, 1, 1, true}, {1, 2, 2, false}, 10},
    {"P410ToP210", {2, 1, 1, true}, {2, 2, 1, true}, 10},
    {"P210ToP010", {2, 2, 1, true}, {2, 2, 2, true}, 10},
    {"P410ToP010", {2, 1, 1, true}, {2, 2, 2, true}, 10},
};

int CeilDiv(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

int ChromaWidth(int width, const Format& format) {
  return CeilDiv(width, format.subsample_x);
}

int ChromaHeight(int height, const Format& format) {
  return CeilDiv(height, format.subsample_y);
}

int RowBytes(int width, int bpc, bool semiplanar) {
  return width * bpc * (semiplanar ? 2 : 1);
}

bool AllocateSurface(const Format& format,
                     int width,
                     int height,
                     int pad_y,
                     int pad_uv,
                     Surface* surface) {
  surface->bpc = format.bpc;
  surface->semiplanar = format.semiplanar;
  if (!surface->y.Allocate(height, width * format.bpc, pad_y)) {
    return false;
  }
  const int chroma_width = ChromaWidth(width, format);
  const int chroma_height = ChromaHeight(height, format);
  if (format.semiplanar) {
    return surface->uv.Allocate(
        chroma_height, RowBytes(chroma_width, format.bpc, true), pad_uv);
  }
  if (!surface->u.Allocate(chroma_height, chroma_width * format.bpc,
                           pad_uv)) {
    return false;
  }
  return surface->v.Allocate(chroma_height, chroma_width * format.bpc,
                             pad_uv);
}

uint32_t EdgeValue(Rng* rng, uint32_t color_max, int index) {
  switch (index & 7) {
    case 0:
      return 0;
    case 1:
      return color_max;
    case 2:
      return color_max / 2;
    case 3:
      return std::min<uint32_t>(1, color_max);
    case 4:
      return (color_max / 2) + std::min<uint32_t>(1, color_max - color_max / 2);
    default:
      return rng->Uniform(color_max);
  }
}

void FillPlane(Plane* plane,
               int bpc,
               uint32_t color_max,
               uint64_t seed,
               int plane_index) {
  Rng rng(seed ^ (0x9e3779b97f4a7c15ull * (plane_index + 1)));
  std::memset(plane->buffer.data(), 0xcd,
              static_cast<size_t>(plane->stride) * plane->rows);
  for (int y = 0; y < plane->rows; ++y) {
    uint8_t* row = plane->Row(y);
    if (bpc == 1) {
      for (int x = 0; x < plane->row_bytes; ++x) {
        row[x] = static_cast<uint8_t>(EdgeValue(&rng, color_max, x + y));
      }
    } else {
      uint16_t* row16 = reinterpret_cast<uint16_t*>(row);
      const int samples = plane->row_bytes / 2;
      for (int x = 0; x < samples; ++x) {
        row16[x] = static_cast<uint16_t>(EdgeValue(&rng, color_max, x + y));
      }
    }
  }
}

void FillSurface(Surface* surface,
                 const Format& format,
                 uint32_t color_max,
                 uint64_t seed,
                 bool random_input) {
  if (!random_input) {
    return;
  }
  FillPlane(&surface->y, format.bpc, color_max, seed, 0);
  if (format.semiplanar) {
    FillPlane(&surface->uv, format.bpc, color_max, seed, 1);
  } else {
    FillPlane(&surface->u, format.bpc, color_max, seed, 1);
    FillPlane(&surface->v, format.bpc, color_max, seed, 2);
  }
}

int InvokeConversion(FunctionId id,
                     const SurfaceView& src,
                     const SurfaceView& dst,
                     int width,
                     int height) {
#define SRC_Y_U8 (src.y.data)
#define SRC_U_U8 (src.u.data)
#define SRC_V_U8 (src.v.data)
#define SRC_UV_U8 (src.uv.data)
#define DST_Y_U8 (dst.y.data)
#define DST_U_U8 (dst.u.data)
#define DST_V_U8 (dst.v.data)
#define DST_UV_U8 (dst.uv.data)
#define SRC_Y_U16 (reinterpret_cast<const uint16_t*>(src.y.data))
#define SRC_U_U16 (reinterpret_cast<const uint16_t*>(src.u.data))
#define SRC_V_U16 (reinterpret_cast<const uint16_t*>(src.v.data))
#define SRC_UV_U16 (reinterpret_cast<const uint16_t*>(src.uv.data))
#define DST_Y_U16 (reinterpret_cast<uint16_t*>(dst.y.data))
#define DST_U_U16 (reinterpret_cast<uint16_t*>(dst.u.data))
#define DST_V_U16 (reinterpret_cast<uint16_t*>(dst.v.data))
#define DST_UV_U16 (reinterpret_cast<uint16_t*>(dst.uv.data))
// 16-bit API strides are expressed in samples, while the harness stores
// byte strides so that its row pointers and guard checks remain uniform.
#define SRC_STRIDE(plane) ((src.bpc == 1) ? (plane).stride : (plane).stride / 2)
#define DST_STRIDE(plane) ((dst.bpc == 1) ? (plane).stride : (plane).stride / 2)

  switch (id) {
    case kI444ToI422:
      return I444ToI422(SRC_Y_U8, SRC_STRIDE(src.y), SRC_U_U8,
                        SRC_STRIDE(src.u), SRC_V_U8, SRC_STRIDE(src.v),
                        DST_Y_U8, DST_STRIDE(dst.y), DST_U_U8,
                        DST_STRIDE(dst.u), DST_V_U8, DST_STRIDE(dst.v), width,
                        height);
    case kI444ToI420:
      return I444ToI420(SRC_Y_U8, SRC_STRIDE(src.y), SRC_U_U8,
                        SRC_STRIDE(src.u), SRC_V_U8, SRC_STRIDE(src.v),
                        DST_Y_U8, DST_STRIDE(dst.y), DST_U_U8,
                        DST_STRIDE(dst.u), DST_V_U8, DST_STRIDE(dst.v), width,
                        height);
    case kI422ToI420:
      return I422ToI420(SRC_Y_U8, SRC_STRIDE(src.y), SRC_U_U8,
                        SRC_STRIDE(src.u), SRC_V_U8, SRC_STRIDE(src.v),
                        DST_Y_U8, DST_STRIDE(dst.y), DST_U_U8,
                        DST_STRIDE(dst.u), DST_V_U8, DST_STRIDE(dst.v), width,
                        height);
    case kI422ToNV12:
      return I422ToNV12(SRC_Y_U8, SRC_STRIDE(src.y), SRC_U_U8,
                        SRC_STRIDE(src.u), SRC_V_U8, SRC_STRIDE(src.v),
                        DST_Y_U8, DST_STRIDE(dst.y), DST_UV_U8,
                        DST_STRIDE(dst.uv), width, height);
    case kI422ToNV21:
      return I422ToNV21(SRC_Y_U8, SRC_STRIDE(src.y), SRC_U_U8,
                        SRC_STRIDE(src.u), SRC_V_U8, SRC_STRIDE(src.v),
                        DST_Y_U8, DST_STRIDE(dst.y), DST_UV_U8,
                        DST_STRIDE(dst.uv), width, height);
    case kI444ToNV12:
      return I444ToNV12(SRC_Y_U8, SRC_STRIDE(src.y), SRC_U_U8,
                        SRC_STRIDE(src.u), SRC_V_U8, SRC_STRIDE(src.v),
                        DST_Y_U8, DST_STRIDE(dst.y), DST_UV_U8,
                        DST_STRIDE(dst.uv), width, height);
    case kI444ToNV21:
      return I444ToNV21(SRC_Y_U8, SRC_STRIDE(src.y), SRC_U_U8,
                        SRC_STRIDE(src.u), SRC_V_U8, SRC_STRIDE(src.v),
                        DST_Y_U8, DST_STRIDE(dst.y), DST_UV_U8,
                        DST_STRIDE(dst.uv), width, height);
    case kNV24ToNV16:
      return NV24ToNV16(SRC_Y_U8, SRC_STRIDE(src.y), SRC_UV_U8,
                        SRC_STRIDE(src.uv), DST_Y_U8, DST_STRIDE(dst.y),
                        DST_UV_U8, DST_STRIDE(dst.uv), width, height);
    case kNV16ToNV12:
      return NV16ToNV12(SRC_Y_U8, SRC_STRIDE(src.y), SRC_UV_U8,
                        SRC_STRIDE(src.uv), DST_Y_U8, DST_STRIDE(dst.y),
                        DST_UV_U8, DST_STRIDE(dst.uv), width, height);
    case kNV24ToNV12:
      return NV24ToNV12(SRC_Y_U8, SRC_STRIDE(src.y), SRC_UV_U8,
                        SRC_STRIDE(src.uv), DST_Y_U8, DST_STRIDE(dst.y),
                        DST_UV_U8, DST_STRIDE(dst.uv), width, height);
    case kNV16ToI420:
      return NV16ToI420(SRC_Y_U8, SRC_STRIDE(src.y), SRC_UV_U8,
                        SRC_STRIDE(src.uv), DST_Y_U8, DST_STRIDE(dst.y),
                        DST_U_U8, DST_STRIDE(dst.u), DST_V_U8,
                        DST_STRIDE(dst.v), width, height);
    case kNV24ToI420:
      return NV24ToI420(SRC_Y_U8, SRC_STRIDE(src.y), SRC_UV_U8,
                        SRC_STRIDE(src.uv), DST_Y_U8, DST_STRIDE(dst.y),
                        DST_U_U8, DST_STRIDE(dst.u), DST_V_U8,
                        DST_STRIDE(dst.v), width, height);
    case kI410ToI210:
      return I410ToI210(SRC_Y_U16, SRC_STRIDE(src.y), SRC_U_U16,
                        SRC_STRIDE(src.u), SRC_V_U16, SRC_STRIDE(src.v),
                        DST_Y_U16, DST_STRIDE(dst.y), DST_U_U16,
                        DST_STRIDE(dst.u), DST_V_U16, DST_STRIDE(dst.v), width,
                        height);
    case kI410ToI010:
      return I410ToI010(SRC_Y_U16, SRC_STRIDE(src.y), SRC_U_U16,
                        SRC_STRIDE(src.u), SRC_V_U16, SRC_STRIDE(src.v),
                        DST_Y_U16, DST_STRIDE(dst.y), DST_U_U16,
                        DST_STRIDE(dst.u), DST_V_U16, DST_STRIDE(dst.v), width,
                        height);
    case kI210ToI010:
      return I210ToI010(SRC_Y_U16, SRC_STRIDE(src.y), SRC_U_U16,
                        SRC_STRIDE(src.u), SRC_V_U16, SRC_STRIDE(src.v),
                        DST_Y_U16, DST_STRIDE(dst.y), DST_U_U16,
                        DST_STRIDE(dst.u), DST_V_U16, DST_STRIDE(dst.v), width,
                        height);
    case kI210ToI420:
      return I210ToI420(SRC_Y_U16, SRC_STRIDE(src.y), SRC_U_U16,
                        SRC_STRIDE(src.u), SRC_V_U16, SRC_STRIDE(src.v),
                        DST_Y_U8, DST_STRIDE(dst.y), DST_U_U8,
                        DST_STRIDE(dst.u), DST_V_U8, DST_STRIDE(dst.v), width,
                        height);
    case kI410ToI420:
      return I410ToI420(SRC_Y_U16, SRC_STRIDE(src.y), SRC_U_U16,
                        SRC_STRIDE(src.u), SRC_V_U16, SRC_STRIDE(src.v),
                        DST_Y_U8, DST_STRIDE(dst.y), DST_U_U8,
                        DST_STRIDE(dst.u), DST_V_U8, DST_STRIDE(dst.v), width,
                        height);
    case kI212ToI420:
      return I212ToI420(SRC_Y_U16, SRC_STRIDE(src.y), SRC_U_U16,
                        SRC_STRIDE(src.u), SRC_V_U16, SRC_STRIDE(src.v),
                        DST_Y_U8, DST_STRIDE(dst.y), DST_U_U8,
                        DST_STRIDE(dst.u), DST_V_U8, DST_STRIDE(dst.v), width,
                        height);
    case kI412ToI420:
      return I412ToI420(SRC_Y_U16, SRC_STRIDE(src.y), SRC_U_U16,
                        SRC_STRIDE(src.u), SRC_V_U16, SRC_STRIDE(src.v),
                        DST_Y_U8, DST_STRIDE(dst.y), DST_U_U8,
                        DST_STRIDE(dst.u), DST_V_U8, DST_STRIDE(dst.v), width,
                        height);
    case kP010ToI420:
      return P010ToI420(SRC_Y_U16, SRC_STRIDE(src.y), SRC_UV_U16,
                        SRC_STRIDE(src.uv), DST_Y_U8, DST_STRIDE(dst.y),
                        DST_U_U8, DST_STRIDE(dst.u), DST_V_U8,
                        DST_STRIDE(dst.v), width, height);
    case kP210ToI420:
      return P210ToI420(SRC_Y_U16, SRC_STRIDE(src.y), SRC_UV_U16,
                        SRC_STRIDE(src.uv), DST_Y_U8, DST_STRIDE(dst.y),
                        DST_U_U8, DST_STRIDE(dst.u), DST_V_U8,
                        DST_STRIDE(dst.v), width, height);
    case kP410ToI420:
      return P410ToI420(SRC_Y_U16, SRC_STRIDE(src.y), SRC_UV_U16,
                        SRC_STRIDE(src.uv), DST_Y_U8, DST_STRIDE(dst.y),
                        DST_U_U8, DST_STRIDE(dst.u), DST_V_U8,
                        DST_STRIDE(dst.v), width, height);
    case kP410ToP210:
      return P410ToP210(SRC_Y_U16, SRC_STRIDE(src.y), SRC_UV_U16,
                        SRC_STRIDE(src.uv), DST_Y_U16, DST_STRIDE(dst.y),
                        DST_UV_U16, DST_STRIDE(dst.uv), width, height);
    case kP210ToP010:
      return P210ToP010(SRC_Y_U16, SRC_STRIDE(src.y), SRC_UV_U16,
                        SRC_STRIDE(src.uv), DST_Y_U16, DST_STRIDE(dst.y),
                        DST_UV_U16, DST_STRIDE(dst.uv), width, height);
    case kP410ToP010:
      return P410ToP010(SRC_Y_U16, SRC_STRIDE(src.y), SRC_UV_U16,
                        SRC_STRIDE(src.uv), DST_Y_U16, DST_STRIDE(dst.y),
                        DST_UV_U16, DST_STRIDE(dst.uv), width, height);
    case kFunctionCount:
      break;
  }

#undef SRC_Y_U8
#undef SRC_U_U8
#undef SRC_V_U8
#undef SRC_UV_U8
#undef DST_Y_U8
#undef DST_U_U8
#undef DST_V_U8
#undef DST_UV_U8
#undef SRC_Y_U16
#undef SRC_U_U16
#undef SRC_V_U16
#undef SRC_UV_U16
#undef DST_Y_U16
#undef DST_U_U16
#undef DST_V_U16
#undef DST_UV_U16
#undef SRC_STRIDE
#undef DST_STRIDE
  return -1;
}

struct Case {
  int fn = 0;
  int width = 0;
  int height = 0;
  int src_pad_y = 0;
  int src_pad_uv = 0;
  int dst_pad_y = 0;
  int dst_pad_uv = 0;
  uint32_t color_max = 0;
  int flip = 0;
  uint64_t seed = 0;
};

enum FailureKind {
  kNoFailure,
  kAllocation,
  kReturnCode,
  kOutputGuardBefore,
  kOutputGuardAfter,
  kOutputRowPadding,
  kReferenceMismatch,
  kNonDeterministic,
  kSourceGuardBefore,
  kSourceGuardAfter,
};

const char* FailureName(FailureKind kind) {
  switch (kind) {
    case kNoFailure:
      return "none";
    case kAllocation:
      return "allocation";
    case kReturnCode:
      return "return-code";
    case kOutputGuardBefore:
      return "output-guard-before";
    case kOutputGuardAfter:
      return "output-guard-after";
    case kOutputRowPadding:
      return "output-row-padding";
    case kReferenceMismatch:
      return "reference-mismatch";
    case kNonDeterministic:
      return "non-deterministic";
    case kSourceGuardBefore:
      return "source-guard-before";
    case kSourceGuardAfter:
      return "source-guard-after";
  }
  return "unknown";
}

struct Failure {
  FailureKind kind = kNoFailure;
  const char* plane = "";
  int x = -1;
  int y = -1;
  int offset = -1;
  uint32_t expected = 0;
  uint32_t actual = 0;
  int return_code = 0;
  int cpu_mask = 0;
};

struct RunOutput {
  Surface surface;
  Failure failure;
  bool ok = false;
};

bool IsValidCase(const Case& c) {
  if (c.fn < 0 || c.fn >= kFunctionCount || c.width <= 0 || c.height <= 0) {
    return false;
  }
  const FunctionDesc& desc = kFunctions[c.fn];
  if (desc.src.bpc == 2 &&
      ((c.src_pad_y & 1) || (c.src_pad_uv & 1))) {
    return false;
  }
  if (desc.dst.bpc == 2 &&
      ((c.dst_pad_y & 1) || (c.dst_pad_uv & 1))) {
    return false;
  }
  if (c.color_max > static_cast<uint32_t>((1 << desc.depth) - 1)) {
    return false;
  }
  return c.src_pad_y >= 0 && c.src_pad_uv >= 0 && c.dst_pad_y >= 0 &&
         c.dst_pad_uv >= 0;
}

bool CheckOutputPlane(const Plane& plane,
                      const char* name,
                      uint8_t output_fill,
                      Failure* failure) {
  size_t bad_offset = 0;
  uint8_t actual = 0;
  if (!plane.buffer.CheckGuard(true, &bad_offset, &actual)) {
    failure->kind = kOutputGuardBefore;
    failure->plane = name;
    failure->offset = -static_cast<int>(bad_offset + 1);
    failure->expected = kGuardByte;
    failure->actual = actual;
    return false;
  }
  if (!plane.buffer.CheckGuard(false, &bad_offset, &actual)) {
    failure->kind = kOutputGuardAfter;
    failure->plane = name;
    failure->offset = static_cast<int>(bad_offset);
    failure->expected = kGuardByte;
    failure->actual = actual;
    return false;
  }
  for (int y = 0; y < plane.rows; ++y) {
    const uint8_t* row = plane.Row(y);
    for (int x = plane.row_bytes; x < plane.stride; ++x) {
      if (row[x] != output_fill) {
        failure->kind = kOutputRowPadding;
        failure->plane = name;
        failure->x = x - plane.row_bytes;
        failure->y = y;
        failure->expected = output_fill;
        failure->actual = row[x];
        return false;
      }
    }
  }
  return true;
}

bool CheckOutput(const Surface& surface,
                 uint8_t output_fill,
                 Failure* failure) {
  if (!CheckOutputPlane(surface.y, "Y", output_fill, failure)) {
    return false;
  }
  if (surface.semiplanar) {
    return CheckOutputPlane(surface.uv, "UV", output_fill, failure);
  }
  if (!CheckOutputPlane(surface.u, "U", output_fill, failure)) {
    return false;
  }
  return CheckOutputPlane(surface.v, "V", output_fill, failure);
}

bool CheckSourceGuard(const Surface& surface, Failure* failure) {
  const Plane* planes[] = {&surface.y, &surface.u, &surface.v, &surface.uv};
  const char* names[] = {"Y", "U", "V", "UV"};
  for (int i = 0; i < 4; ++i) {
    const Plane* plane = planes[i];
    if (!plane->buffer.data()) {
      continue;
    }
    size_t bad_offset = 0;
    uint8_t actual = 0;
    if (!plane->buffer.CheckGuard(true, &bad_offset, &actual)) {
      failure->kind = kSourceGuardBefore;
      failure->plane = names[i];
      failure->offset = -static_cast<int>(bad_offset + 1);
      failure->expected = kGuardByte;
      failure->actual = actual;
      return false;
    }
    if (!plane->buffer.CheckGuard(false, &bad_offset, &actual)) {
      failure->kind = kSourceGuardAfter;
      failure->plane = names[i];
      failure->offset = static_cast<int>(bad_offset);
      failure->expected = kGuardByte;
      failure->actual = actual;
      return false;
    }
  }
  return true;
}

struct PlaneCompare {
  const char* name;
  const Plane* a;
  const Plane* b;
};

bool CompareSurface(const Surface& a,
                    const Surface& b,
                    const FunctionDesc& desc,
                    FailureKind kind,
                    Failure* failure) {
  PlaneCompare planes[4];
  int count = 1;
  planes[0] = {"Y", &a.y, &b.y};
  if (desc.dst.semiplanar) {
    planes[1] = {"UV", &a.uv, &b.uv};
    count = 2;
  } else {
    planes[1] = {"U", &a.u, &b.u};
    planes[2] = {"V", &a.v, &b.v};
    count = 3;
  }
  for (int p = 0; p < count; ++p) {
    for (int y = 0; y < planes[p].a->rows; ++y) {
      const uint8_t* row_a = planes[p].a->Row(y);
      const uint8_t* row_b = planes[p].b->Row(y);
      if (desc.dst.bpc == 1) {
        for (int x = 0; x < planes[p].a->row_bytes; ++x) {
          if (row_a[x] != row_b[x]) {
            failure->kind = kind;
            failure->plane = planes[p].name;
            failure->x = x;
            failure->y = y;
            failure->expected = row_a[x];
            failure->actual = row_b[x];
            return false;
          }
        }
      } else {
        const uint16_t* row_a16 = reinterpret_cast<const uint16_t*>(row_a);
        const uint16_t* row_b16 = reinterpret_cast<const uint16_t*>(row_b);
        const int samples = planes[p].a->row_bytes / 2;
        for (int x = 0; x < samples; ++x) {
          if (row_a16[x] != row_b16[x]) {
            failure->kind = kind;
            failure->plane = planes[p].name;
            failure->x = x;
            failure->y = y;
            failure->expected = row_a16[x];
            failure->actual = row_b16[x];
            return false;
          }
        }
      }
    }
  }
  return true;
}

RunOutput RunOne(const SurfaceView& src,
                 const Case& c,
                 int cpu_mask,
                 uint8_t output_fill) {
  RunOutput output;
  const FunctionDesc& desc = kFunctions[c.fn];
  if (!AllocateSurface(desc.dst, c.width, c.height, c.dst_pad_y,
                       c.dst_pad_uv, &output.surface)) {
    output.failure.kind = kAllocation;
    output.failure.cpu_mask = cpu_mask;
    return output;
  }
  if (desc.dst.bpc == 1) {
    std::memset(output.surface.y.buffer.data(), output_fill,
                output.surface.y.buffer.size());
    if (desc.dst.semiplanar) {
      std::memset(output.surface.uv.buffer.data(), output_fill,
                  output.surface.uv.buffer.size());
    } else {
      std::memset(output.surface.u.buffer.data(), output_fill,
                  output.surface.u.buffer.size());
      std::memset(output.surface.v.buffer.data(), output_fill,
                  output.surface.v.buffer.size());
    }
  } else {
    const uint16_t fill16 =
        static_cast<uint16_t>(output_fill * 0x0101u);
    const size_t y_samples = output.surface.y.buffer.size() / 2;
    uint16_t* y16 = reinterpret_cast<uint16_t*>(output.surface.y.buffer.data());
    for (size_t i = 0; i < y_samples; ++i) {
      y16[i] = fill16;
    }
    if (desc.dst.semiplanar) {
      const size_t uv_samples = output.surface.uv.buffer.size() / 2;
      uint16_t* uv16 =
          reinterpret_cast<uint16_t*>(output.surface.uv.buffer.data());
      for (size_t i = 0; i < uv_samples; ++i) {
        uv16[i] = fill16;
      }
    } else {
      const size_t u_samples = output.surface.u.buffer.size() / 2;
      uint16_t* u16 =
          reinterpret_cast<uint16_t*>(output.surface.u.buffer.data());
      for (size_t i = 0; i < u_samples; ++i) {
        u16[i] = fill16;
      }
      const size_t v_samples = output.surface.v.buffer.size() / 2;
      uint16_t* v16 =
          reinterpret_cast<uint16_t*>(output.surface.v.buffer.data());
      for (size_t i = 0; i < v_samples; ++i) {
        v16[i] = fill16;
      }
    }
  }

  MaskCpuFlags(cpu_mask);
  const int pass_height = c.flip ? -c.height : c.height;
  const int ret = InvokeConversion(static_cast<FunctionId>(c.fn), src,
                                   MakeSurfaceView(output.surface), c.width,
                                   pass_height);
  output.failure.cpu_mask = cpu_mask;
  if (ret != 0) {
    output.failure.kind = kReturnCode;
    output.failure.return_code = ret;
    return output;
  }
  if (g_inject_mismatch && cpu_mask != 1) {
    output.surface.y.Row(0)[0] ^= 0x80;
  }
  if (!CheckOutput(output.surface, output_fill, &output.failure)) {
    return output;
  }
  output.ok = true;
  return output;
}

struct EvalOutput {
  Failure failure;
  bool ok = false;
};

EvalOutput EvaluateOne(const Case& c, int cpu_mask, bool check_determinism) {
  EvalOutput result;
  const FunctionDesc& desc = kFunctions[c.fn];
  Surface src;
  if (!AllocateSurface(desc.src, c.width, c.height, c.src_pad_y,
                       c.src_pad_uv, &src)) {
    result.failure.kind = kAllocation;
    result.failure.cpu_mask = cpu_mask;
    return result;
  }
  FillSurface(&src, desc.src, c.color_max, c.seed, true);
  const SurfaceView src_view = MakeSurfaceView(src);

  RunOutput reference = RunOne(src_view, c, 1, kOutputFillA);
  if (!reference.ok) {
    result.failure = reference.failure;
    return result;
  }

  RunOutput first = RunOne(src_view, c, cpu_mask, kOutputFillA);
  if (!first.ok) {
    result.failure = first.failure;
    return result;
  }
  if (!CompareSurface(reference.surface, first.surface, desc,
                      kReferenceMismatch, &result.failure)) {
    result.failure.cpu_mask = cpu_mask;
    return result;
  }

  if (check_determinism) {
    RunOutput second = RunOne(src_view, c, cpu_mask, kOutputFillB);
    if (!second.ok) {
      result.failure = second.failure;
      return result;
    }
    if (!CompareSurface(first.surface, second.surface, desc,
                        kNonDeterministic, &result.failure)) {
      result.failure.cpu_mask = cpu_mask;
      return result;
    }
  }

  if (!CheckSourceGuard(src, &result.failure)) {
    result.failure.cpu_mask = cpu_mask;
    return result;
  }
  result.ok = true;
  return result;
}

std::vector<int> GetCpuMasks() {
  std::vector<int> masks;
  const int all = MaskCpuFlags(-1);
  masks.push_back(all);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
  if (all & kCpuHasAVX512BW) {
    masks.push_back(all & ~kCpuHasAVX512BW);
  }
  if (all & kCpuHasAVX2) {
    masks.push_back(all & ~(kCpuHasAVX512BW | kCpuHasAVX2));
  }
  if (all & kCpuHasSSSE3) {
    masks.push_back(
        all & ~(kCpuHasAVX512BW | kCpuHasAVX2 | kCpuHasSSSE3));
  }
#elif defined(__aarch64__)
  if (all & kCpuHasSME) {
    masks.push_back(all & ~kCpuHasSME);
  }
  if (all & kCpuHasSVE2) {
    masks.push_back(all & ~(kCpuHasSVE2 | kCpuHasSME));
  }
  if (all & kCpuHasNEON) {
    masks.push_back(all & ~(kCpuHasNEON | kCpuHasSVE2 | kCpuHasSME));
  }
#endif
  MaskCpuFlags(-1);
  std::sort(masks.begin(), masks.end());
  masks.erase(std::unique(masks.begin(), masks.end()), masks.end());
  return masks;
}

const char* MaskName(int mask) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
  if (mask & kCpuHasAVX512BW) {
    return "all(+AVX512BW)";
  }
  if (mask & kCpuHasAVX2) {
    return "AVX2";
  }
  if (mask & kCpuHasSSSE3) {
    return "SSSE3";
  }
  return "C/SSE2";
#elif defined(__aarch64__)
  if (mask & kCpuHasSME) {
    return "all(+SME)";
  }
  if (mask & kCpuHasSVE2) {
    return "SVE2";
  }
  if (mask & kCpuHasNEON) {
    return "NEON";
  }
  return "C";
#else
  (void)mask;
  return "all";
#endif
}

bool SameFailure(const EvalOutput& result, FailureKind kind) {
  return !result.ok && result.failure.kind == kind;
}

Case ShrinkCase(const Case& original, int cpu_mask, FailureKind kind) {
  const bool check_determinism = kind == kNonDeterministic;
  Case best = original;
  bool changed = true;
  while (changed) {
    changed = false;

    for (int width = 1; width < best.width; ++width) {
      Case candidate = best;
      candidate.width = width;
      EvalOutput result =
          EvaluateOne(candidate, cpu_mask, check_determinism);
      if (SameFailure(result, kind)) {
        best = candidate;
        changed = true;
        break;
      }
    }
    if (changed) {
      continue;
    }

    for (int height = 1; height < best.height; ++height) {
      Case candidate = best;
      candidate.height = height;
      EvalOutput result =
          EvaluateOne(candidate, cpu_mask, check_determinism);
      if (SameFailure(result, kind)) {
        best = candidate;
        changed = true;
        break;
      }
    }
    if (changed) {
      continue;
    }

    const FunctionDesc& desc = kFunctions[best.fn];
    const int src_step = desc.src.bpc == 2 ? 2 : 1;
    const int dst_step = desc.dst.bpc == 2 ? 2 : 1;
    int Case::*pads[] = {&Case::src_pad_y, &Case::src_pad_uv,
                         &Case::dst_pad_y, &Case::dst_pad_uv};
    const int steps[] = {src_step, src_step, dst_step, dst_step};
    for (int i = 0; i < 4 && !changed; ++i) {
      for (int pad = 0; pad < best.*pads[i]; pad += steps[i]) {
        Case candidate = best;
        candidate.*pads[i] = pad;
        EvalOutput result =
            EvaluateOne(candidate, cpu_mask, check_determinism);
        if (SameFailure(result, kind)) {
          best = candidate;
          changed = true;
          break;
        }
      }
    }
    if (changed) {
      continue;
    }

    for (uint32_t color_max = 0; color_max < best.color_max; ++color_max) {
      Case candidate = best;
      candidate.color_max = color_max;
      EvalOutput result =
          EvaluateOne(candidate, cpu_mask, check_determinism);
      if (SameFailure(result, kind)) {
        best = candidate;
        changed = true;
        break;
      }
    }
    if (changed) {
      continue;
    }

    if (best.flip) {
      Case candidate = best;
      candidate.flip = 0;
      EvalOutput result =
          EvaluateOne(candidate, cpu_mask, check_determinism);
      if (SameFailure(result, kind)) {
        best = candidate;
        changed = true;
      }
    }
  }
  return best;
}

void PrintFailure(const char* label, const Case& c, const Failure& failure) {
  const FunctionDesc& desc = kFunctions[c.fn];
  std::printf("%s: %s\n", label, desc.name);
  std::printf("  width=%d height=%d flip=%d color_max=%u seed=0x%016" PRIx64
              "\n",
              c.width, c.height, c.flip, c.color_max, c.seed);
  std::printf("  src_pad_y=%d src_pad_uv=%d dst_pad_y=%d dst_pad_uv=%d\n",
              c.src_pad_y, c.src_pad_uv, c.dst_pad_y, c.dst_pad_uv);
  std::printf("  cpu_mask=0x%x (%s) reason=%s", failure.cpu_mask,
              MaskName(failure.cpu_mask), FailureName(failure.kind));
  if (failure.return_code) {
    std::printf(" return_code=%d", failure.return_code);
  }
  if (failure.plane[0]) {
    std::printf(" plane=%s", failure.plane);
  }
  if (failure.y >= 0) {
    std::printf(" x=%d y=%d", failure.x, failure.y);
  }
  if (failure.offset >= 0) {
    std::printf(" guard_offset=%d", failure.offset);
  }
  if (failure.kind == kReferenceMismatch ||
      failure.kind == kNonDeterministic ||
      failure.kind == kOutputRowPadding ||
      failure.kind == kOutputGuardBefore ||
      failure.kind == kOutputGuardAfter) {
    std::printf(" expected=%u actual=%u", failure.expected, failure.actual);
  }
  std::printf("\n");
}

uint32_t ChooseDimension(Rng* rng, int max_value) {
  static const int kEdges[] = {1,  2,  3,  4,  7,  8,  9,  15, 16,
                               17, 31, 32, 33, 63, 64, 65, 127, 128, 129};
  if ((rng->Next() & 3) == 0) {
    int value = kEdges[rng->Next() % (sizeof(kEdges) / sizeof(kEdges[0]))];
    return static_cast<uint32_t>(std::min(value, max_value));
  }
  return 1 + rng->Uniform(static_cast<uint32_t>(max_value - 1));
}

uint32_t ChoosePad(Rng* rng, int bpc) {
  static const int kPads[] = {0, 1, 2, 3, 7, 8, 15, 16, 31, 32};
  int pad = kPads[rng->Next() % (sizeof(kPads) / sizeof(kPads[0]))];
  if (bpc == 2) {
    pad &= ~1;
  }
  return static_cast<uint32_t>(pad);
}

Case MakeCase(Rng* rng, int fn, int max_width, int max_height) {
  const FunctionDesc& desc = kFunctions[fn];
  Case c;
  c.fn = fn;
  c.width = static_cast<int>(ChooseDimension(rng, max_width));
  c.height = static_cast<int>(ChooseDimension(rng, max_height));
  c.src_pad_y = static_cast<int>(ChoosePad(rng, desc.src.bpc));
  c.src_pad_uv = static_cast<int>(ChoosePad(rng, desc.src.bpc));
  c.dst_pad_y = static_cast<int>(ChoosePad(rng, desc.dst.bpc));
  c.dst_pad_uv = static_cast<int>(ChoosePad(rng, desc.dst.bpc));
  const uint32_t full_color_max = (1u << desc.depth) - 1;
  c.color_max = (rng->Next() & 1) ? full_color_max
                                   : rng->Uniform(full_color_max);
  c.flip = (rng->Next() % 5) == 0 ? 1 : 0;
  c.seed = rng->Next64();
  return c;
}

struct Stats {
  uint64_t cases = 0;
  uint64_t passed = 0;
  uint64_t failed = 0;
};

void PrintUsage(const char* program) {
  std::printf(
      "usage: %s [--cases N] [--seed N] [--max-width N] [--max-height N] "
      "[--filter NAME] [--no-shrink] [--inject-mismatch]\n",
      program);
}

}  // namespace

int main(int argc, char** argv) {
  uint64_t cases = 10000;
  uint64_t seed = 0x3712;
  int max_width = 160;
  int max_height = 96;
  std::string filter;
  bool shrink = true;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--cases") == 0 && i + 1 < argc) {
      cases = std::strtoull(argv[++i], nullptr, 0);
    } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      seed = std::strtoull(argv[++i], nullptr, 0);
    } else if (std::strcmp(argv[i], "--max-width") == 0 && i + 1 < argc) {
      max_width = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--max-height") == 0 && i + 1 < argc) {
      max_height = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
      filter = argv[++i];
    } else if (std::strcmp(argv[i], "--no-shrink") == 0) {
      shrink = false;
    } else if (std::strcmp(argv[i], "--inject-mismatch") == 0) {
      g_inject_mismatch = true;
    } else if (std::strcmp(argv[i], "--help") == 0) {
      PrintUsage(argv[0]);
      return 0;
    } else {
      PrintUsage(argv[0]);
      return 2;
    }
  }
  if (cases == 0 || max_width < 1 || max_height < 1) {
    PrintUsage(argv[0]);
    return 2;
  }

  std::vector<int> functions;
  for (int fn = 0; fn < kFunctionCount; ++fn) {
    if (filter.empty() || std::strstr(kFunctions[fn].name, filter.c_str())) {
      functions.push_back(fn);
    }
  }
  if (functions.empty()) {
    std::printf("no function matched filter '%s'\n", filter.c_str());
    return 2;
  }

  const std::vector<int> masks = GetCpuMasks();
  std::printf("functions=%zu cases=%" PRIu64 " seed=0x%016" PRIx64
              " max=%dx%d masks=%zu\n",
              functions.size(), cases, seed, max_width, max_height,
              masks.size());
  std::printf("cpu masks:");
  for (int mask : masks) {
    std::printf(" %s(0x%x)", MaskName(mask), mask);
  }
  std::printf("\n");

  uint64_t fn_cases[kFunctionCount] = {};
  uint64_t fn_failed[kFunctionCount] = {};
  uint64_t fn_shrunk[kFunctionCount] = {};
  bool printed[kFunctionCount][10] = {};
  Stats stats;
  Rng rng(seed);

  for (uint64_t i = 0; i < cases; ++i) {
    const int fn = functions[i % functions.size()];
    Case c = MakeCase(&rng, fn, max_width, max_height);
    if (!IsValidCase(c)) {
      std::printf("internal error: generated invalid case\n");
      return 2;
    }
    fn_cases[fn]++;
    stats.cases++;

    bool case_failed = false;
    for (int mask : masks) {
      EvalOutput result = EvaluateOne(c, mask, true);
      if (result.ok) {
        continue;
      }
      case_failed = true;
      const FailureKind kind = result.failure.kind;
      if (kind < 10 && !printed[fn][kind]) {
        printed[fn][kind] = true;
        Case minimal = c;
        if (shrink) {
          minimal = ShrinkCase(c, mask, kind);
          fn_shrunk[fn]++;
        }
        char label[128];
        std::snprintf(label, sizeof(label), "MINIMAL %s case #%" PRIu64,
                      shrink ? "COUNTEREXAMPLE" : "FAILURE", i);
        PrintFailure(label, minimal, result.failure);
      }
      break;
    }
    if (case_failed) {
      stats.failed++;
      fn_failed[fn]++;
    } else {
      stats.passed++;
    }
    if ((i + 1) % 5000 == 0) {
      std::printf("progress: %" PRIu64 "/%" PRIu64 " cases, %" PRIu64
                  " failed\n",
                  i + 1, cases, stats.failed);
      std::fflush(stdout);
    }
  }

  MaskCpuFlags(-1);
  std::printf("\nper-function summary:\n");
  for (int fn : functions) {
    std::printf("  %-12s cases=%6" PRIu64 " failed=%6" PRIu64
                " shrunk=%" PRIu64 "\n",
                kFunctions[fn].name, fn_cases[fn], fn_failed[fn],
                fn_shrunk[fn]);
  }
  std::printf(
      "\nsummary: cases=%" PRIu64 " passed=%" PRIu64 " failed=%" PRIu64
      " masks=%zu\n",
      stats.cases, stats.passed, stats.failed, masks.size());
  if (stats.failed == 0) {
    std::printf("RESULT: PASS\n");
    return 0;
  }
  std::printf("RESULT: FAIL\n");
  return 1;
}
