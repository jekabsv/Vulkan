#include "Polygon2D.h"

#include <algorithm>
#include <limits>

namespace Ship
{

    namespace
    {
        double Cross(const glm::dvec2& a, const glm::dvec2& b)
        {
            return a.x * b.y - a.y * b.x;
        }

        bool PointInTriangle(const glm::dvec2& p, const glm::dvec2& a, const glm::dvec2& b, const glm::dvec2& c)
        {
            const double d1 = Cross(b - a, p - a);
            const double d2 = Cross(c - b, p - b);
            const double d3 = Cross(a - c, p - c);

            const bool hasNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
            const bool hasPos = (d1 > 0) || (d2 > 0) || (d3 > 0);

            return !(hasNeg && hasPos);
        }

        Loop2D ForceWinding(const Loop2D& loop, bool wantCCW)
        {
            const double area = SignedArea(loop);
            const bool isCCW = area > 0.0;

            if (isCCW == wantCCW)
            {
                return loop;
            }

            Loop2D reversed(loop.rbegin(), loop.rend());
            return reversed;
        }

        // Adds the region bounded by `loop` (any winding) to running vertex-pair
        // sums used by both SignedArea and ComputeMoments. The shoelace-family
        // formulas are linear in the loop, so a hole wound opposite the outer
        // loop naturally subtracts its contribution when the sums are combined.
        struct MomentAccumulator
        {
            double area2{ 0.0 };
            double cx6{ 0.0 };
            double cy6{ 0.0 };
            double ixx12{ 0.0 };
            double iyy12{ 0.0 };
            double ixy24{ 0.0 };

            void Add(const Loop2D& loop)
            {
                const size_t n = loop.size();
                for (size_t i = 0; i < n; i++)
                {
                    const glm::dvec2& p0 = loop[i];
                    const glm::dvec2& p1 = loop[(i + 1) % n];
                    const double cross = p0.x * p1.y - p1.x * p0.y;

                    area2 += cross;
                    cx6 += (p0.x + p1.x) * cross;
                    cy6 += (p0.y + p1.y) * cross;
                    ixx12 += (p0.y * p0.y + p0.y * p1.y + p1.y * p1.y) * cross;
                    iyy12 += (p0.x * p0.x + p0.x * p1.x + p1.x * p1.x) * cross;
                    ixy24 += (p0.x * p1.y + 2.0 * p0.x * p0.y + 2.0 * p1.x * p1.y + p1.x * p0.y) * cross;
                }
            }
        };
    }

    double SignedArea(const Loop2D& loop)
    {
        double area2 = 0.0;
        const size_t n = loop.size();
        for (size_t i = 0; i < n; i++)
        {
            const glm::dvec2& p0 = loop[i];
            const glm::dvec2& p1 = loop[(i + 1) % n];
            area2 += p0.x * p1.y - p1.x * p0.y;
        }
        return 0.5 * area2;
    }

    PolygonMoments2D ComputeMoments(const Loop2D& outer, const std::vector<Loop2D>& holes)
    {
        MomentAccumulator sum;
        sum.Add(ForceWinding(outer, true));
        for (const Loop2D& hole : holes)
        {
            sum.Add(ForceWinding(hole, false));
        }

        PolygonMoments2D moments;
        moments.area = sum.area2 * 0.5;

        if (std::abs(moments.area) < 1e-12)
        {
            return moments;
        }

        // Standard polygon centroid is cx6/(6A); sum.area2 == 2A, so divide by 3*area2.
        moments.centroid.x = sum.cx6 / (3.0 * sum.area2);
        moments.centroid.y = sum.cy6 / (3.0 * sum.area2);

        const double ixxOrigin = sum.ixx12 / 12.0;
        const double iyyOrigin = sum.iyy12 / 12.0;
        const double ixyOrigin = sum.ixy24 / 24.0;

        // Parallel axis theorem back to the centroid.
        moments.Ipp = ixxOrigin - moments.area * moments.centroid.y * moments.centroid.y;
        moments.Iqq = iyyOrigin - moments.area * moments.centroid.x * moments.centroid.x;
        moments.Ipq = ixyOrigin - moments.area * moments.centroid.x * moments.centroid.y;

        return moments;
    }

    namespace
    {
        // Finds an outer-loop vertex that can be safely connected to `anchor`
        // (a hole vertex) with a bridge edge that stays inside the polygon and
        // crosses no other edge. Standard "rightmost hole vertex, ray cast,
        // reflex-safe candidate" hole-elimination heuristic (as used by earcut
        // and other polygon triangulators).
        size_t FindBridgeOuterVertex(const std::vector<glm::dvec2>& outer, const glm::dvec2& anchor)
        {
            const size_t n = outer.size();
            double bestX = std::numeric_limits<double>::infinity();
            size_t edgeStart = 0;
            glm::dvec2 intersection{};
            bool found = false;

            for (size_t i = 0; i < n; i++)
            {
                const glm::dvec2& p0 = outer[i];
                const glm::dvec2& p1 = outer[(i + 1) % n];

                const bool crosses = (p0.y > anchor.y) != (p1.y > anchor.y);
                if (!crosses)
                {
                    continue;
                }

                const double t = (anchor.y - p0.y) / (p1.y - p0.y);
                const double x = p0.x + t * (p1.x - p0.x);

                if (x >= anchor.x && x < bestX)
                {
                    bestX = x;
                    edgeStart = i;
                    intersection = glm::dvec2(x, anchor.y);
                    found = true;
                }
            }

            if (!found)
            {
                // Degenerate fallback: nearest outer vertex.
                size_t best = 0;
                double bestDist = std::numeric_limits<double>::infinity();
                for (size_t i = 0; i < n; i++)
                {
                    const double d = glm::length(outer[i] - anchor);
                    if (d < bestDist)
                    {
                        bestDist = d;
                        best = i;
                    }
                }
                return best;
            }

            const glm::dvec2& p0 = outer[edgeStart];
            const glm::dvec2& p1 = outer[(edgeStart + 1) % n];
            size_t candidate = (p0.x > p1.x) ? edgeStart : (edgeStart + 1) % n;

            // Refine: if any reflex vertex lies inside triangle(anchor, intersection, candidate),
            // pick the one making the smallest angle with the anchor->intersection ray instead.
            const glm::dvec2 candidatePoint = outer[candidate];
            double bestAngle = std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < n; i++)
            {
                if (!PointInTriangle(outer[i], anchor, intersection, candidatePoint))
                {
                    continue;
                }

                const glm::dvec2 toPoint = outer[i] - anchor;
                const glm::dvec2 toRay(1.0, 0.0);
                const double angle = std::abs(std::atan2(Cross(toRay, toPoint), glm::dot(toRay, toPoint)));
                if (angle < bestAngle)
                {
                    bestAngle = angle;
                    candidate = i;
                }
            }

            return candidate;
        }

        // Merges `hole` into `outer` via a bridge edge, returning the new simple loop.
        std::vector<glm::dvec2> BridgeHole(const std::vector<glm::dvec2>& outer, const std::vector<glm::dvec2>& hole)
        {
            size_t anchorIndex = 0;
            double bestX = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < hole.size(); i++)
            {
                if (hole[i].x > bestX)
                {
                    bestX = hole[i].x;
                    anchorIndex = i;
                }
            }

            const size_t bridgeOuter = FindBridgeOuterVertex(outer, hole[anchorIndex]);

            std::vector<glm::dvec2> merged;
            merged.reserve(outer.size() + hole.size() + 2);

            for (size_t i = 0; i <= bridgeOuter; i++)
            {
                merged.push_back(outer[i]);
            }
            for (size_t i = 0; i <= hole.size(); i++)
            {
                merged.push_back(hole[(anchorIndex + i) % hole.size()]);
            }
            for (size_t i = bridgeOuter; i < outer.size(); i++)
            {
                merged.push_back(outer[i]);
            }

            return merged;
        }

        bool IsConvex(const glm::dvec2& prev, const glm::dvec2& cur, const glm::dvec2& next)
        {
            return Cross(cur - prev, next - cur) > 0.0;
        }

        std::vector<Triangle2D> EarClip(const std::vector<glm::dvec2>& polygon)
        {
            std::vector<Triangle2D> triangles;
            std::vector<uint32_t> indices(polygon.size());
            for (uint32_t i = 0; i < polygon.size(); i++)
            {
                indices[i] = i;
            }

            size_t guard = 0;
            const size_t guardLimit = polygon.size() * polygon.size() + 16;

            while (indices.size() > 3 && guard++ < guardLimit)
            {
                bool clipped = false;
                const size_t n = indices.size();

                for (size_t i = 0; i < n; i++)
                {
                    const size_t iPrev = (i + n - 1) % n;
                    const size_t iNext = (i + 1) % n;

                    const glm::dvec2& prev = polygon[indices[iPrev]];
                    const glm::dvec2& cur = polygon[indices[i]];
                    const glm::dvec2& next = polygon[indices[iNext]];

                    if (!IsConvex(prev, cur, next))
                    {
                        continue;
                    }

                    bool anyInside = false;
                    for (size_t j = 0; j < n; j++)
                    {
                        if (j == i || j == iPrev || j == iNext)
                        {
                            continue;
                        }
                        if (PointInTriangle(polygon[indices[j]], prev, cur, next))
                        {
                            anyInside = true;
                            break;
                        }
                    }

                    if (anyInside)
                    {
                        continue;
                    }

                    triangles.push_back({ indices[iPrev], indices[i], indices[iNext] });
                    indices.erase(indices.begin() + i);
                    clipped = true;
                    break;
                }

                if (!clipped)
                {
                    // Numerically degenerate polygon; stop rather than spin forever.
                    break;
                }
            }

            if (indices.size() == 3)
            {
                triangles.push_back({ indices[0], indices[1], indices[2] });
            }

            return triangles;
        }
    }

    std::vector<Triangle2D> Triangulate(
        const Loop2D& outer,
        const std::vector<Loop2D>& holes,
        std::vector<glm::dvec2>& outVertices)
    {
        std::vector<glm::dvec2> working = ForceWinding(outer, true);

        for (const Loop2D& hole : holes)
        {
            std::vector<glm::dvec2> h = ForceWinding(hole, false);
            working = BridgeHole(working, h);
        }

        outVertices = working;
        return EarClip(working);
    }

} // namespace Ship
