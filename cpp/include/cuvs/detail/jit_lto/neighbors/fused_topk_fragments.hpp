/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/fused_distance_nn/fused_1nn_fragments.hpp>

namespace cuvs::neighbors::detail {

using cuvs::distance::detail::cutile_tile_config;

template <typename DataTag,
          typename IndexTag,
          int PhysicalK,
          typename TileTag,
          typename AbiTag,
          typename ArchTag>
struct fragment_tag_fused_topk_cubin {
  static constexpr int cc_major   = ArchTag::cc_major;
  static constexpr int cc_minor   = ArchTag::cc_minor;
  static constexpr int physical_k = PhysicalK;
};

}  // namespace cuvs::neighbors::detail
