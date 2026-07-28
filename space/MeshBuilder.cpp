#include "MeshBuilder.h"

#include <unordered_map>

namespace Ship
{

    namespace
    {
        Loop2D ForceCCW(const Loop2D& loop)
        {
            if (SignedArea(loop) < 0.0)
            {
                return Loop2D(loop.rbegin(), loop.rend());
            }
            return loop;
        }

        Core::Vertex MakeVertex(const glm::dvec3& worldPos, const glm::dvec3& worldNormal)
        {
            Core::Vertex v;
            v.position = glm::vec3(worldPos);
            v.normal = glm::vec3(glm::normalize(worldNormal));
            v.uv = glm::vec2(0.0f, 0.0f);
            return v;
        }

        // Sweeps a solid prism: a cross-section (outer loop, optional holes) in the
        // local (p, q) plane, extruded along n from startOffset to endOffset. Used
        // for both members (n = axis, extrude = length) and panels (n = normal,
        // extrude = thickness) -- see the comment on PrismInertiaLocal for why this
        // is the same shape either way.
        //
        // Caps are triangulated with holes (so a hollow tube's or a hatch's true
        // cross-section is visible at the ends / faces). Side walls are only swept
        // for the loops passed in `sweptLoops` -- the caller decides which ones
        // (typically the outer loop always, hole rims optionally).
        void AppendPrismMesh(
            std::vector<Core::Vertex>& vertices,
            std::vector<uint32_t>& indices,
            const glm::dvec3& origin,
            const glm::dvec3& pHat,
            const glm::dvec3& qHat,
            const glm::dvec3& nHat,
            const Loop2D& outerLocal,
            const std::vector<Loop2D>& holesLocal,
            double startOffset,
            double endOffset)
        {
            auto toWorld = [&](const glm::dvec2& local, double n)
            {
                return origin + pHat * local.x + qHat * local.y + nHat * n;
            };

            auto emitTriangle = [&](const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c)
            {
                const glm::dvec3 normal = glm::cross(b - a, c - a);
                const uint32_t base = static_cast<uint32_t>(vertices.size());
                vertices.push_back(MakeVertex(a, normal));
                vertices.push_back(MakeVertex(b, normal));
                vertices.push_back(MakeVertex(c, normal));
                indices.push_back(base + 0);
                indices.push_back(base + 1);
                indices.push_back(base + 2);
            };

            // Caps (only meaningful when the prism has axial extent).
            if (std::abs(endOffset - startOffset) > 1e-9)
            {
                std::vector<glm::dvec2> capVerts;
                const std::vector<Triangle2D> tris = Triangulate(outerLocal, holesLocal, capVerts);

                for (const Triangle2D& tri : tris)
                {
                    const glm::dvec3 a = toWorld(capVerts[tri.a], endOffset);
                    const glm::dvec3 b = toWorld(capVerts[tri.b], endOffset);
                    const glm::dvec3 c = toWorld(capVerts[tri.c], endOffset);
                    emitTriangle(a, b, c); // outward = +n, matches ear-clip CCW winding

                    const glm::dvec3 a0 = toWorld(capVerts[tri.a], startOffset);
                    const glm::dvec3 b0 = toWorld(capVerts[tri.b], startOffset);
                    const glm::dvec3 c0 = toWorld(capVerts[tri.c], startOffset);
                    emitTriangle(a0, c0, b0); // reversed: outward = -n
                }
            }

            // Side walls: sweep the outer loop, and each hole rim, between the caps.
            std::vector<Loop2D> sweptLoops;
            sweptLoops.push_back(ForceCCW(outerLocal));
            for (const Loop2D& hole : holesLocal)
            {
                sweptLoops.push_back(hole);
            }

            for (const Loop2D& loop : sweptLoops)
            {
                const size_t n = loop.size();
                for (size_t i = 0; i < n; i++)
                {
                    const glm::dvec2& p0 = loop[i];
                    const glm::dvec2& p1 = loop[(i + 1) % n];

                    const glm::dvec3 v0 = toWorld(p0, startOffset);
                    const glm::dvec3 v1 = toWorld(p1, startOffset);
                    const glm::dvec3 v2 = toWorld(p1, endOffset);
                    const glm::dvec3 v3 = toWorld(p0, endOffset);

                    emitTriangle(v0, v1, v2);
                    emitTriangle(v0, v2, v3);
                }
            }
        }
    }

    std::vector<RenderBatch> BuildRenderBatches(const Assembly& assembly)
    {
        std::unordered_map<MaterialId, RenderBatch> batches;

        auto batchFor = [&](MaterialId material) -> RenderBatch&
        {
            auto it = batches.find(material);
            if (it == batches.end())
            {
                RenderBatch batch;
                batch.material = material;
                it = batches.emplace(material, std::move(batch)).first;
            }
            return it->second;
        };

        for (const auto& [id, member] : assembly.Members())
        {
            const MemberFrame frame = assembly.ComputeMemberFrame(member);
            const Loop2D outline = member.profile.Outline();
            const std::vector<Loop2D> holes = member.profile.Holes();

            RenderBatch& batch = batchFor(member.material);
            AppendPrismMesh(
                batch.vertices, batch.indices,
                frame.worldA, frame.pHat, frame.qHat, frame.nHat,
                outline, holes,
                0.0, frame.length);
        }

        for (const auto& [id, panel] : assembly.Panels())
        {
            const PanelFrame frame = assembly.ComputePanelFrame(panel);

            RenderBatch& batch = batchFor(panel.material);
            AppendPrismMesh(
                batch.vertices, batch.indices,
                frame.origin, frame.pHat, frame.qHat, frame.nHat,
                frame.outerLocal, frame.holesLocal,
                -panel.thickness * 0.5, panel.thickness * 0.5);
        }

        std::vector<RenderBatch> result;
        result.reserve(batches.size());
        for (auto& [material, batch] : batches)
        {
            result.push_back(std::move(batch));
        }
        return result;
    }

} // namespace Ship
