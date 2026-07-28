#pragma once

// Turns the authored graph (members, panels) into renderable triangle soup,
// batched by material so the demo app can issue one draw call per material
// instead of one per part. This is a view of the assembly, not a second
// source of truth -- physics (Assembly.cpp) and rendering both read the same
// per-part world-space frames.

#include <vector>

#include "Assembly.h"
#include "Mesh.h"

namespace Ship
{

    struct RenderBatch
    {
        MaterialId material{ InvalidMaterialId };
        std::vector<Core::Vertex> vertices;
        std::vector<uint32_t> indices;
    };

    std::vector<RenderBatch> BuildRenderBatches(const Assembly& assembly);

} // namespace Ship
