// UYVY camera payload -> upright RGB rows. Given (Claude), not part of the 200 lines.
#pragma once

#include <cstdint>
#include <vector>

#include "md/types.h"

namespace md {

// Decode upright rows [row0, row0 + rows) of the frame into `out` (rows x 360 x 3 RGB, row-major).
// Upright row r is raw column r (np.rot90(raw, 3)), so a search window decodes only its 360 raw
// columns. The arithmetic is vision/uyvy.decode's, in float32 and in the same order, rounded half
// to even and clipped, so the bytes equal the Python decode exactly (build with -ffp-contract=off:
// a fused multiply-add rounds differently).
void decode_upright_rows(const uint8_t* payload, int row0, int rows, std::vector<uint8_t>& out);

}  // namespace md
