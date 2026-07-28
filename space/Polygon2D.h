#pragma once

// 2D polygon geometry used for two things that turn out to be the same math:
//   - a panel's face loop (plus an optional opening loop), in the panel's own plane
//   - a member profile's cross-section outline, in the plane perpendicular to its axis
//
// Both need: signed area, centroid, and the second moments of area about the
// centroid (so mass properties "fall out of geometry" instead of being
// authored), plus a triangulation for rendering. Loops with a single hole
// (an opening) are supported by bridging the hole into the outer loop before
// ear-clipping, and by summing signed moments across loops (a hole wound
// opposite the outer loop simply subtracts its contribution).

#include <vector>

#include <glm/glm.hpp>

namespace Ship
{

    // Closed-form second moments of area of a simple polygon, about its own centroid,
    // in its own local 2D coordinate system (p, q).
    struct PolygonMoments2D
    {
        double area{ 0.0 };
        glm::dvec2 centroid{ 0.0 };

        // About the centroid: Ipp = d/y-analogue (int q^2 dA), Iqq = int p^2 dA, Ipq = int p*q dA.
        double Ipp{ 0.0 };
        double Iqq{ 0.0 };
        double Ipq{ 0.0 };

        double Polar() const { return Ipp + Iqq; }
    };

    // A simple (non-self-intersecting) closed polygon loop, wound CCW for solid
    // material and CW for a hole, in some local 2D coordinate system.
    using Loop2D = std::vector<glm::dvec2>;

    // Moments of the region described by `outer` minus every loop in `holes`.
    // Loop winding is normalized internally (outer forced CCW, holes forced CW),
    // so callers do not need to get winding right by hand.
    PolygonMoments2D ComputeMoments(const Loop2D& outer, const std::vector<Loop2D>& holes = {});

    double SignedArea(const Loop2D& loop);

    struct Triangle2D
    {
        uint32_t a, b, c;
    };

    // Ear-clipping triangulation of a simple polygon (`outer`), with at most
    // one hole bridged in. Returns triangle indices into the vertex list that
    // is written to `outVertices` (bridge duplicates included). Assumes the
    // hole lies strictly inside the outer loop and loops do not self-intersect.
    std::vector<Triangle2D> Triangulate(
        const Loop2D& outer,
        const std::vector<Loop2D>& holes,
        std::vector<glm::dvec2>& outVertices);

} // namespace Ship
