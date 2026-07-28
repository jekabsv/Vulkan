#pragma once

// Hand-placing every frame and plate of a hull is the failure mode a
// generator exists to avoid. This one takes a spine, a cross-section preset,
// a length, and a taper, and emits a complete ring-and-longeron structure
// with skin panels -- a procedural blockout the player then hand-details.
//
// The output is plain Members and Panels added to the Assembly through the
// same public API a human editing session would use; there is nothing here
// the assembly doesn't already know how to represent. What's *not* here yet
// is the "stays live as a re-runnable stamp until first hand-edit" behaviour
// the design spec calls for -- this first pass always bakes immediately.

#include <vector>

#include "Assembly.h"

namespace Ship
{

    enum class CrossSectionPreset : uint8_t
    {
        Circular = 0,
        Hexagonal,
        Rectangular,
        Faceted
    };

    struct HullGeneratorParams
    {
        glm::dvec3 startPoint{ 0.0 };
        glm::dvec3 axis{ 1.0, 0.0, 0.0 };
        double length{ 20.0 };
        int stationCount{ 10 };

        CrossSectionPreset preset{ CrossSectionPreset::Circular };
        int facetedSegments{ 12 };

        double startRadius{ 2.0 };
        double endRadius{ 0.8 };

        MemberProfile longeronProfile;
        MemberProfile ringProfile;
        MaterialId structureMaterial{ InvalidMaterialId };

        double skinThickness{ 0.003 };
        MaterialId skinMaterial{ InvalidMaterialId };
    };

    struct HullGeneratorResult
    {
        std::vector<std::vector<NodeId>> ringNodes; // [station][segment]
        std::vector<MemberId> longerons;
        std::vector<MemberId> rings;
        std::vector<PanelId> skinPanels;
    };

    GridCoord SnapToGrid(const glm::dvec3& worldPos, double unitMetres);

    HullGeneratorResult GenerateHullSection(Assembly& assembly, const HullGeneratorParams& params);

} // namespace Ship
