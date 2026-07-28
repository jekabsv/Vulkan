#pragma once

// The catalog a part can point into. A Member or Panel never carries an
// authored stat (mass, stiffness, ...) -- it carries a profile/thickness and
// a material index, and everything physical is derived from those plus
// geometry. This file only holds the catalog entries and the pure geometry
// (cross-section outlines, analytic areas/moments); Assembly.h does the
// deriving.

#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "Polygon2D.h"

namespace Ship
{

    // --- Materials -----------------------------------------------------------

    struct PartMaterial
    {
        std::string name;
        double density{ 7850.0 };       // kg / m^3
        double yieldStrength{ 250e6 };  // Pa, for future structural analysis
        glm::vec3 renderColor{ 0.6f, 0.6f, 0.6f };
    };

    using MaterialId = uint32_t;
    constexpr MaterialId InvalidMaterialId = UINT32_MAX;

    class MaterialCatalog
    {
    public:
        MaterialCatalog();

        MaterialId Add(PartMaterial material);
        const PartMaterial& Get(MaterialId id) const { return m_Materials.at(id); }
        size_t Count() const { return m_Materials.size(); }

        MaterialId Steel() const { return m_Steel; }
        MaterialId Aluminum() const { return m_Aluminum; }
        MaterialId Titanium() const { return m_Titanium; }

    private:
        std::vector<PartMaterial> m_Materials;
        MaterialId m_Steel{ InvalidMaterialId };
        MaterialId m_Aluminum{ InvalidMaterialId };
        MaterialId m_Titanium{ InvalidMaterialId };
    };

    // --- Member profiles -------------------------------------------------------

    enum class ProfileKind : uint8_t
    {
        Box = 0,     // hollow rectangular tube
        Tube,        // hollow round tube
        IBeam,
        Angle,
        Channel,
        SolidBar     // solid rectangular bar (also used as generic strip stock)
    };

    // A profile is fully described by its kind and dimensions; the outline and
    // every derived quantity are computed from that, never authored separately.
    struct MemberProfile
    {
        ProfileKind kind{ ProfileKind::Box };

        // Meaning depends on `kind`:
        //  Box:      a=outer width, b=outer height, t=wall thickness
        //  Tube:     a=outer radius, t=wall thickness (b unused)
        //  IBeam:    a=height, b=flange width, t=web thickness, t2=flange thickness
        //  Angle:    a=leg A length, b=leg B length, t=thickness
        //  Channel:  a=height, b=flange width, t=web thickness, t2=flange thickness
        //  SolidBar: a=width, b=height
        double a{ 0.05 };
        double b{ 0.05 };
        double t{ 0.003 };
        double t2{ 0.003 };

        // Cross-section outline in the profile's local (p, q) plane, perpendicular
        // to the member axis. Path nodes sit at the local origin.
        Loop2D Outline() const;
        std::vector<Loop2D> Holes() const;

        PolygonMoments2D Moments() const { return ComputeMoments(Outline(), Holes()); }

        std::string Describe() const;

        bool operator==(const MemberProfile& other) const
        {
            return kind == other.kind && a == other.a && b == other.b && t == other.t && t2 == other.t2;
        }
    };

    MemberProfile MakeBoxProfile(double outerWidth, double outerHeight, double wallThickness);
    MemberProfile MakeTubeProfile(double outerRadius, double wallThickness);
    MemberProfile MakeIBeamProfile(double height, double flangeWidth, double webThickness, double flangeThickness);
    MemberProfile MakeAngleProfile(double legA, double legB, double thickness);
    MemberProfile MakeChannelProfile(double height, double flangeWidth, double webThickness, double flangeThickness);
    MemberProfile MakeSolidBarProfile(double width, double height);

} // namespace Ship
