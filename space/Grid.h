#pragma once

// Integer grid coordinates for assembly nodes.
//
// Everything a builder places snaps to an integer lattice point. This buys
// exact snapping with no accumulated float drift, a deterministic geometric
// hash for Marks (see Mark.h), and reproducible mass/inertia results
// regardless of machine or edit order. The physical spacing of the lattice
// (metres per grid unit) is a per-assembly constant: a ship might use 0.5 m,
// a torpedo 20 mm.

#include <cstdint>
#include <functional>

#include <glm/glm.hpp>

namespace Ship
{

    using GridCoord = glm::ivec3;

    struct GridCoordHash
    {
        size_t operator()(const GridCoord& coord) const noexcept
        {
            size_t h = std::hash<int32_t>()(coord.x);
            h ^= std::hash<int32_t>()(coord.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int32_t>()(coord.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    inline bool operator==(const GridCoord& lhs, const GridCoord& rhs)
    {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }

    // Assembly-local grid resolution: world metres per lattice unit.
    struct GridFrame
    {
        double unitMetres{ 0.5 };

        glm::dvec3 ToWorld(const GridCoord& coord) const
        {
            return glm::dvec3(coord) * unitMetres;
        }
    };

} // namespace Ship
