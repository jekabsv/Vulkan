#include "HullGenerator.h"

#define M_PI 3.14159265358979323846

#include <cmath>

namespace Ship
{

    namespace
    {
        int SegmentCount(const HullGeneratorParams& params)
        {
            switch (params.preset)
            {
            case CrossSectionPreset::Circular: return params.facetedSegments;
            case CrossSectionPreset::Hexagonal: return 6;
            case CrossSectionPreset::Rectangular: return 4;
            case CrossSectionPreset::Faceted: return params.facetedSegments;
            }
            return params.facetedSegments;
        }

        double PhaseOffset(const HullGeneratorParams& params)
        {
            // Flat-bottom / flat-side orientation instead of a vertex pointing
            // straight down, which is what a hull cross-section wants visually.
            switch (params.preset)
            {
            case CrossSectionPreset::Rectangular: return M_PI / 4.0;
            case CrossSectionPreset::Hexagonal: return M_PI / 6.0;
            default: return 0.0;
            }
        }
    }

    GridCoord SnapToGrid(const glm::dvec3& worldPos, double unitMetres)
    {
        const glm::dvec3 units = worldPos / unitMetres;
        return GridCoord(
            static_cast<int32_t>(std::lround(units.x)),
            static_cast<int32_t>(std::lround(units.y)),
            static_cast<int32_t>(std::lround(units.z)));
    }

    HullGeneratorResult GenerateHullSection(Assembly& assembly, const HullGeneratorParams& params)
    {
        HullGeneratorResult result;

        const int segments = std::max(3, SegmentCount(params));
        const int stations = std::max(2, params.stationCount);
        const double phase = PhaseOffset(params);
        const double unitMetres = assembly.Grid().unitMetres;

        const glm::dvec3 axis = glm::normalize(params.axis);
        const glm::dvec3 right = StableOrthogonal(axis);
        const glm::dvec3 up = glm::normalize(glm::cross(axis, right));

        result.ringNodes.resize(stations);

        for (int s = 0; s < stations; s++)
        {
            const double t = static_cast<double>(s) / static_cast<double>(stations - 1);
            const glm::dvec3 center = params.startPoint + axis * (t * params.length);
            const double radius = params.startRadius + (params.endRadius - params.startRadius) * t;

            result.ringNodes[s].resize(segments);

            for (int i = 0; i < segments; i++)
            {
                const double angle = phase + (2.0 * M_PI * i) / segments;
                const glm::dvec3 worldPos = center + right * (radius * std::cos(angle)) + up * (radius * std::sin(angle));
                const GridCoord coord = SnapToGrid(worldPos, unitMetres);
                result.ringNodes[s][i] = assembly.FindOrAddNode(coord);
            }
        }

        // Ring frames: the hoop at each station.
        for (int s = 0; s < stations; s++)
        {
            for (int i = 0; i < segments; i++)
            {
                Member ring;
                ring.nodeA = result.ringNodes[s][i];
                ring.nodeB = result.ringNodes[s][(i + 1) % segments];
                ring.profile = params.ringProfile;
                ring.material = params.structureMaterial;
                const MemberId id = assembly.AddMember(ring);
                if (id != InvalidId) result.rings.push_back(id);
            }
        }

        // Longerons: run the length of the hull.
        for (int s = 0; s + 1 < stations; s++)
        {
            for (int i = 0; i < segments; i++)
            {
                Member longeron;
                longeron.nodeA = result.ringNodes[s][i];
                longeron.nodeB = result.ringNodes[s + 1][i];
                longeron.profile = params.longeronProfile;
                longeron.material = params.structureMaterial;
                const MemberId id = assembly.AddMember(longeron);
                if (id != InvalidId) result.longerons.push_back(id);
            }
        }

        // Skin: one quad panel per hull facet between consecutive stations.
        for (int s = 0; s + 1 < stations; s++)
        {
            for (int i = 0; i < segments; i++)
            {
                const int iNext = (i + 1) % segments;

                Panel skin;
                skin.loop = {
                    result.ringNodes[s][i],
                    result.ringNodes[s][iNext],
                    result.ringNodes[s + 1][iNext],
                    result.ringNodes[s + 1][i]
                };
                skin.thickness = params.skinThickness;
                skin.material = params.skinMaterial;
                const PanelId id = assembly.AddPanel(skin);
                if (id != InvalidId) result.skinPanels.push_back(id);
            }
        }

        return result;
    }

} // namespace Ship
