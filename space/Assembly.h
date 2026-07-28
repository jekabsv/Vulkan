#pragma once

// The assembly graph: nodes, members (edges), panels (face loops). This is
// the one thing that is authored; mass properties and the bill of materials
// are derived views maintained incrementally as the graph changes.
//
// Mass properties never lag an edit and never get recomputed from scratch:
// each part contributes a cached (mass, first-moment, inertia-about-origin)
// triple, the assembly keeps running sums of those, and the parallel axis
// theorem composes contributions and, at query time, shifts the total from
// "about the origin" to "about the centre of mass". Adding or removing a
// part is O(1); nothing here is O(part count) except iteration for
// rendering or the bill of materials.

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "Grid.h"
#include "Mark.h"
#include "PartCatalog.h"
#include "Polygon2D.h"

namespace Ship
{

    using NodeId = uint32_t;
    using MemberId = uint32_t;
    using PanelId = uint32_t;
    constexpr uint32_t InvalidId = UINT32_MAX;

    struct Member
    {
        NodeId nodeA{ InvalidId };
        NodeId nodeB{ InvalidId };
        MemberProfile profile;
        MaterialId material{ InvalidMaterialId };
    };

    struct Panel
    {
        std::vector<NodeId> loop;                     // outer boundary, grid-snapped, coplanar
        std::vector<std::vector<NodeId>> openings;     // hole loops (hatches, ports, penetrations)
        double thickness{ 0.003 };                     // metres
        MaterialId material{ InvalidMaterialId };
    };

    // World-space frame shared by physics and mesh generation so the two
    // never disagree about where a part actually is.
    struct MemberFrame
    {
        glm::dvec3 worldA{ 0.0 };
        glm::dvec3 worldB{ 0.0 };
        glm::dvec3 pHat{ 1.0, 0.0, 0.0 };
        glm::dvec3 qHat{ 0.0, 1.0, 0.0 };
        glm::dvec3 nHat{ 0.0, 0.0, 1.0 };
        double length{ 0.0 };
    };

    struct PanelFrame
    {
        glm::dvec3 origin{ 0.0 };
        glm::dvec3 pHat{ 1.0, 0.0, 0.0 };
        glm::dvec3 qHat{ 0.0, 1.0, 0.0 };
        glm::dvec3 nHat{ 0.0, 0.0, 1.0 };
        Loop2D outerLocal;
        std::vector<Loop2D> holesLocal;
    };

    // Cached per-part contribution to the assembly's running sums, kept so a
    // removal can subtract in O(1) instead of forcing a full recompute.
    struct PartContribution
    {
        double mass{ 0.0 };
        glm::dvec3 centroidWorld{ 0.0 };
        glm::dmat3 inertiaAboutOrigin{ 0.0 }; // this part's own I_cm (world axes) + parallel-axis shift to origin
        MarkId mark{ 0 };
    };

    struct MassProperties
    {
        double mass{ 0.0 };
        glm::dvec3 centerOfMass{ 0.0 };
        glm::dmat3 inertiaAboutCOM{ 0.0 }; // symmetric, world axes, about the centre of mass
    };

    class Assembly
    {
    public:
        explicit Assembly(GridFrame grid, MaterialCatalog materials);

        NodeId AddNode(const GridCoord& coord);
        NodeId FindOrAddNode(const GridCoord& coord);
        glm::dvec3 WorldPosition(NodeId node) const { return m_Grid.ToWorld(m_Nodes[node]); }
        const GridCoord& Coord(NodeId node) const { return m_Nodes[node]; }

        MemberId AddMember(const Member& member);
        PanelId AddPanel(const Panel& panel);

        void RemoveMember(MemberId id);
        void RemovePanel(PanelId id);

        const std::vector<GridCoord>& Nodes() const { return m_Nodes; }
        const std::unordered_map<MemberId, Member>& Members() const { return m_Members; }
        const std::unordered_map<PanelId, Panel>& Panels() const { return m_Panels; }

        const MaterialCatalog& Materials() const { return m_Materials; }
        const GridFrame& Grid() const { return m_Grid; }

        MemberFrame ComputeMemberFrame(const Member& member) const;
        PanelFrame ComputePanelFrame(const Panel& panel) const;

        const MassProperties& GetMassProperties() const { return m_MassProperties; }
        const BillOfMaterials& GetBillOfMaterials() const { return m_Bom; }

    private:
        PartContribution ComputeMemberContribution(const Member& member) const;
        PartContribution ComputePanelContribution(const Panel& panel) const;

        void ApplyContribution(const PartContribution& c, double sign);
        void RecomputeDerived();

    private:
        GridFrame m_Grid;
        MaterialCatalog m_Materials;

        std::vector<GridCoord> m_Nodes;
        std::unordered_map<GridCoord, NodeId, GridCoordHash> m_NodeLookup;

        std::unordered_map<MemberId, Member> m_Members;
        std::unordered_map<MemberId, PartContribution> m_MemberContributions;
        MemberId m_NextMemberId{ 0 };

        std::unordered_map<PanelId, Panel> m_Panels;
        std::unordered_map<PanelId, PartContribution> m_PanelContributions;
        PanelId m_NextPanelId{ 0 };

        // Running sums (O(1) to update per edit).
        double m_SumMass{ 0.0 };
        glm::dvec3 m_SumFirstMoment{ 0.0 };
        glm::dmat3 m_SumInertiaAboutOrigin{ 0.0 };

        MassProperties m_MassProperties;
        BillOfMaterials m_Bom;
    };

    // Builds the local (p, q, n) orthonormal frame for a member axis, and the
    // symmetric mass-inertia tensor of a uniform prism (a cross-section shape
    // in the p,q plane extruded along n) about its own centroid, in that local
    // frame. The same formula serves members (n = beam axis, extrude = length)
    // and panels (n = face normal, extrude = thickness) -- a beam and a plate
    // are the same prism integral with the roles of "long" and "thin" swapped.
    glm::dvec3 StableOrthogonal(const glm::dvec3& axis);
    glm::dmat3 PrismInertiaLocal(double mass, const PolygonMoments2D& poly, double extrudeLength);

} // namespace Ship
