#include "Assembly.h"

#include <cmath>
#include <cstdio>

namespace Ship
{

    glm::dvec3 StableOrthogonal(const glm::dvec3& axis)
    {
        const glm::dvec3 helper = (std::abs(axis.x) < 0.9) ? glm::dvec3(1.0, 0.0, 0.0) : glm::dvec3(0.0, 1.0, 0.0);
        return glm::normalize(glm::cross(helper, axis));
    }

    glm::dmat3 PrismInertiaLocal(double mass, const PolygonMoments2D& poly, double extrudeLength)
    {
        if (poly.area <= 0.0 || mass <= 0.0)
        {
            return glm::dmat3(0.0);
        }

        const double density2D = mass / poly.area; // mass per unit area of the poly, i.e. rho * extrudeLength
        const double thinTerm = mass * extrudeLength * extrudeLength / 12.0;

        const double Ipp = density2D * poly.Ipp + thinTerm;
        const double Iqq = density2D * poly.Iqq + thinTerm;
        const double Inn = density2D * (poly.Ipp + poly.Iqq);
        const double Ipq = -density2D * poly.Ipq;

        return glm::dmat3(
            Ipp, Ipq, 0.0,
            Ipq, Iqq, 0.0,
            0.0, 0.0, Inn);
    }

    namespace
    {
        glm::dmat3 ParallelAxisToOrigin(const glm::dmat3& inertiaAboutCentroid, double mass, const glm::dvec3& centroid)
        {
            const double d2 = glm::dot(centroid, centroid);
            return inertiaAboutCentroid + mass * (d2 * glm::dmat3(1.0) - glm::outerProduct(centroid, centroid));
        }
    }

    Assembly::Assembly(GridFrame grid, MaterialCatalog materials)
        : m_Grid(grid)
        , m_Materials(std::move(materials))
    {
    }

    NodeId Assembly::AddNode(const GridCoord& coord)
    {
        const NodeId id = static_cast<NodeId>(m_Nodes.size());
        m_Nodes.push_back(coord);
        m_NodeLookup[coord] = id;
        return id;
    }

    NodeId Assembly::FindOrAddNode(const GridCoord& coord)
    {
        auto it = m_NodeLookup.find(coord);
        if (it != m_NodeLookup.end())
        {
            return it->second;
        }
        return AddNode(coord);
    }

    MemberFrame Assembly::ComputeMemberFrame(const Member& member) const
    {
        MemberFrame frame;
        frame.worldA = WorldPosition(member.nodeA);
        frame.worldB = WorldPosition(member.nodeB);

        const glm::dvec3 delta = frame.worldB - frame.worldA;
        frame.length = glm::length(delta);

        if (frame.length < 1e-9)
        {
            frame.nHat = glm::dvec3(0.0, 0.0, 1.0);
        }
        else
        {
            frame.nHat = delta / frame.length;
        }

        frame.pHat = StableOrthogonal(frame.nHat);
        frame.qHat = glm::normalize(glm::cross(frame.nHat, frame.pHat));

        return frame;
    }

    PanelFrame Assembly::ComputePanelFrame(const Panel& panel) const
    {
        PanelFrame frame;

        std::vector<glm::dvec3> worldOuter;
        worldOuter.reserve(panel.loop.size());
        for (NodeId n : panel.loop)
        {
            worldOuter.push_back(WorldPosition(n));
        }

        // Newell's method: robust normal estimate even for near-planar loops.
        glm::dvec3 normal(0.0);
        const size_t n = worldOuter.size();
        for (size_t i = 0; i < n; i++)
        {
            const glm::dvec3& cur = worldOuter[i];
            const glm::dvec3& next = worldOuter[(i + 1) % n];
            normal.x += (cur.y - next.y) * (cur.z + next.z);
            normal.y += (cur.z - next.z) * (cur.x + next.x);
            normal.z += (cur.x - next.x) * (cur.y + next.y);
        }

        frame.origin = worldOuter[0];
        frame.nHat = glm::length(normal) > 1e-12 ? glm::normalize(normal) : glm::dvec3(0.0, 0.0, 1.0);

        glm::dvec3 pRaw = worldOuter[1] - worldOuter[0];
        pRaw -= frame.nHat * glm::dot(pRaw, frame.nHat); // Gram-Schmidt against the normal
        frame.pHat = glm::length(pRaw) > 1e-12 ? glm::normalize(pRaw) : StableOrthogonal(frame.nHat);
        frame.qHat = glm::normalize(glm::cross(frame.nHat, frame.pHat));

        auto project = [&](const std::vector<glm::dvec3>& worldPts)
        {
            Loop2D local;
            local.reserve(worldPts.size());
            for (const glm::dvec3& p : worldPts)
            {
                const glm::dvec3 rel = p - frame.origin;
                local.emplace_back(glm::dot(rel, frame.pHat), glm::dot(rel, frame.qHat));
            }
            return local;
        };

        frame.outerLocal = project(worldOuter);

        for (const std::vector<NodeId>& hole : panel.openings)
        {
            std::vector<glm::dvec3> worldHole;
            worldHole.reserve(hole.size());
            for (NodeId nid : hole)
            {
                worldHole.push_back(WorldPosition(nid));
            }
            frame.holesLocal.push_back(project(worldHole));
        }

        return frame;
    }

    PartContribution Assembly::ComputeMemberContribution(const Member& member) const
    {
        const MemberFrame frame = ComputeMemberFrame(member);
        const PolygonMoments2D poly = member.profile.Moments();
        const PartMaterial& material = m_Materials.Get(member.material);

        PartContribution c;
        c.mass = poly.area * frame.length * material.density;

        const glm::dvec3 midpoint = 0.5 * (frame.worldA + frame.worldB);
        c.centroidWorld = midpoint + frame.pHat * poly.centroid.x + frame.qHat * poly.centroid.y;

        const glm::dmat3 localInertia = PrismInertiaLocal(c.mass, poly, frame.length);
        const glm::dmat3 R(frame.pHat, frame.qHat, frame.nHat);
        const glm::dmat3 worldInertia = R * localInertia * glm::transpose(R);
        c.inertiaAboutOrigin = ParallelAxisToOrigin(worldInertia, c.mass, c.centroidWorld);

        c.mark = HashMemberMark(
            static_cast<uint32_t>(member.profile.kind),
            member.profile.a, member.profile.b, member.profile.t, member.profile.t2,
            member.material, frame.length);

        return c;
    }

    PartContribution Assembly::ComputePanelContribution(const Panel& panel) const
    {
        const PanelFrame frame = ComputePanelFrame(panel);
        const PolygonMoments2D poly = ComputeMoments(frame.outerLocal, frame.holesLocal);
        const PartMaterial& material = m_Materials.Get(panel.material);

        PartContribution c;
        c.mass = poly.area * panel.thickness * material.density;
        c.centroidWorld = frame.origin + frame.pHat * poly.centroid.x + frame.qHat * poly.centroid.y;

        const glm::dmat3 localInertia = PrismInertiaLocal(c.mass, poly, panel.thickness);
        const glm::dmat3 R(frame.pHat, frame.qHat, frame.nHat);
        const glm::dmat3 worldInertia = R * localInertia * glm::transpose(R);
        c.inertiaAboutOrigin = ParallelAxisToOrigin(worldInertia, c.mass, c.centroidWorld);

        double perimeter = 0.0;
        for (size_t i = 0; i < frame.outerLocal.size(); i++)
        {
            perimeter += glm::length(frame.outerLocal[i] - frame.outerLocal[(i + 1) % frame.outerLocal.size()]);
        }

        double openingArea = 0.0;
        for (const Loop2D& hole : frame.holesLocal)
        {
            openingArea += std::abs(SignedArea(hole));
        }

        c.mark = HashPanelMark(panel.material, panel.thickness, poly.area, perimeter,
            static_cast<int>(panel.openings.size()), openingArea);

        return c;
    }

    void Assembly::ApplyContribution(const PartContribution& c, double sign)
    {
        m_SumMass += sign * c.mass;
        m_SumFirstMoment += sign * c.mass * c.centroidWorld;
        m_SumInertiaAboutOrigin += sign * c.inertiaAboutOrigin;
    }

    void Assembly::RecomputeDerived()
    {
        m_MassProperties.mass = m_SumMass;

        if (m_SumMass > 1e-12)
        {
            m_MassProperties.centerOfMass = m_SumFirstMoment / m_SumMass;
            const glm::dvec3& com = m_MassProperties.centerOfMass;
            const double d2 = glm::dot(com, com);
            m_MassProperties.inertiaAboutCOM = m_SumInertiaAboutOrigin - m_SumMass * (d2 * glm::dmat3(1.0) - glm::outerProduct(com, com));
        }
        else
        {
            m_MassProperties.centerOfMass = glm::dvec3(0.0);
            m_MassProperties.inertiaAboutCOM = glm::dmat3(0.0);
        }
    }

    MemberId Assembly::AddMember(const Member& member)
    {
        if (member.nodeA == member.nodeB)
        {
            // Two grid coordinates collapsed onto the same lattice point (typically
            // a generator ring sampled finer than the grid resolves at that radius).
            // A zero-length member has no physical meaning; skip it rather than
            // pollute the BOM with a degenerate zero-mass entry.
            return InvalidId;
        }

        const MemberId id = m_NextMemberId++;
        const PartContribution c = ComputeMemberContribution(member);

        m_Members[id] = member;
        m_MemberContributions[id] = c;
        ApplyContribution(c, +1.0);
        RecomputeDerived();

        m_Bom.AddPart(c.mark, member.profile.Describe() + " / " + m_Materials.Get(member.material).name, c.mass);

        return id;
    }

    PanelId Assembly::AddPanel(const Panel& panel)
    {
        const PartContribution c = ComputePanelContribution(panel);
        if (c.mass <= 1e-9)
        {
            // Degenerate loop (collapsed grid points leave near-zero area); see
            // the equivalent AddMember guard for why this is skipped, not stored.
            return InvalidId;
        }

        const PanelId id = m_NextPanelId++;

        m_Panels[id] = panel;
        m_PanelContributions[id] = c;
        ApplyContribution(c, +1.0);
        RecomputeDerived();

        char thicknessLabel[32];
        std::snprintf(thicknessLabel, sizeof(thicknessLabel), "%.1fmm", panel.thickness * 1000.0);
        m_Bom.AddPart(c.mark, std::string("Panel ") + thicknessLabel + " / " + m_Materials.Get(panel.material).name, c.mass);

        return id;
    }

    void Assembly::RemoveMember(MemberId id)
    {
        auto it = m_MemberContributions.find(id);
        if (it == m_MemberContributions.end())
        {
            return;
        }

        ApplyContribution(it->second, -1.0);
        RecomputeDerived();
        m_Bom.RemovePart(it->second.mark, it->second.mass);

        m_MemberContributions.erase(it);
        m_Members.erase(id);
    }

    void Assembly::RemovePanel(PanelId id)
    {
        auto it = m_PanelContributions.find(id);
        if (it == m_PanelContributions.end())
        {
            return;
        }

        ApplyContribution(it->second, -1.0);
        RecomputeDerived();
        m_Bom.RemovePart(it->second.mark, it->second.mass);

        m_PanelContributions.erase(it);
        m_Panels.erase(id);
    }

} // namespace Ship
