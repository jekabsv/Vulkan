#include "PartCatalog.h"

#define M_PI 3.14159265358979323846

#include <cmath>
#include <sstream>

namespace Ship
{

    MaterialCatalog::MaterialCatalog()
    {
        PartMaterial steel;
        steel.name = "Structural Steel";
        steel.density = 7850.0;
        steel.yieldStrength = 250e6;
        steel.renderColor = glm::vec3(0.55f, 0.56f, 0.58f);
        m_Steel = Add(steel);

        PartMaterial aluminum;
        aluminum.name = "Aluminum 6061-T6";
        aluminum.density = 2700.0;
        aluminum.yieldStrength = 276e6;
        aluminum.renderColor = glm::vec3(0.78f, 0.80f, 0.82f);
        m_Aluminum = Add(aluminum);

        PartMaterial titanium;
        titanium.name = "Titanium 6Al-4V";
        titanium.density = 4430.0;
        titanium.yieldStrength = 880e6;
        titanium.renderColor = glm::vec3(0.62f, 0.60f, 0.66f);
        m_Titanium = Add(titanium);
    }

    MaterialId MaterialCatalog::Add(PartMaterial material)
    {
        const MaterialId id = static_cast<MaterialId>(m_Materials.size());
        m_Materials.push_back(std::move(material));
        return id;
    }

    namespace
    {
        Loop2D CircleOutline(double radius, int segments)
        {
            Loop2D loop;
            loop.reserve(segments);
            for (int i = 0; i < segments; i++)
            {
                const double theta = (2.0 * M_PI * i) / segments;
                loop.emplace_back(radius * std::cos(theta), radius * std::sin(theta));
            }
            return loop;
        }
    }

    Loop2D MemberProfile::Outline() const
    {
        switch (kind)
        {
        case ProfileKind::Box:
        {
            const double hw = a * 0.5;
            const double hh = b * 0.5;
            return { { -hw, -hh }, { hw, -hh }, { hw, hh }, { -hw, hh } };
        }
        case ProfileKind::Tube:
        {
            return CircleOutline(a, 16);
        }
        case ProfileKind::IBeam:
        {
            const double H = a;
            const double W = b;
            const double tw = t;
            const double tf = t2;
            const double hh = H * 0.5;
            const double hw = W * 0.5;
            const double htw = tw * 0.5;
            return {
                { -hw, -hh }, { hw, -hh }, { hw, -hh + tf }, { htw, -hh + tf },
                { htw, hh - tf }, { hw, hh - tf }, { hw, hh }, { -hw, hh },
                { -hw, hh - tf }, { -htw, hh - tf }, { -htw, -hh + tf }, { -hw, -hh + tf }
            };
        }
        case ProfileKind::Angle:
        {
            const double La = a;
            const double Lb = b;
            return { { 0.0, 0.0 }, { La, 0.0 }, { La, t }, { t, t }, { t, Lb }, { 0.0, Lb } };
        }
        case ProfileKind::Channel:
        {
            const double H = a;
            const double W = b;
            const double tw = t;
            const double tf = t2;
            const double hh = H * 0.5;
            return {
                { 0.0, -hh }, { W, -hh }, { W, -hh + tf }, { tw, -hh + tf },
                { tw, hh - tf }, { W, hh - tf }, { W, hh }, { 0.0, hh }
            };
        }
        case ProfileKind::SolidBar:
        default:
        {
            const double hw = a * 0.5;
            const double hh = b * 0.5;
            return { { -hw, -hh }, { hw, -hh }, { hw, hh }, { -hw, hh } };
        }
        }
    }

    std::vector<Loop2D> MemberProfile::Holes() const
    {
        switch (kind)
        {
        case ProfileKind::Box:
        {
            const double hw = a * 0.5 - t;
            const double hh = b * 0.5 - t;
            if (hw <= 0.0 || hh <= 0.0)
            {
                return {};
            }
            return { { { -hw, -hh }, { hw, -hh }, { hw, hh }, { -hw, hh } } };
        }
        case ProfileKind::Tube:
        {
            const double innerRadius = a - t;
            if (innerRadius <= 0.0)
            {
                return {};
            }
            return { CircleOutline(innerRadius, 16) };
        }
        default:
            return {};
        }
    }

    std::string MemberProfile::Describe() const
    {
        std::ostringstream out;
        out.setf(std::ios::fixed);
        out.precision(1);

        switch (kind)
        {
        case ProfileKind::Box:
            out << "Box " << (a * 1000.0) << "x" << (b * 1000.0) << "x" << (t * 1000.0) << "mm";
            break;
        case ProfileKind::Tube:
            out << "Tube OD" << (a * 2000.0) << "x" << (t * 1000.0) << "mm";
            break;
        case ProfileKind::IBeam:
            out << "I-Beam " << (a * 1000.0) << "x" << (b * 1000.0) << "mm";
            break;
        case ProfileKind::Angle:
            out << "Angle " << (a * 1000.0) << "x" << (b * 1000.0) << "x" << (t * 1000.0) << "mm";
            break;
        case ProfileKind::Channel:
            out << "Channel " << (a * 1000.0) << "x" << (b * 1000.0) << "mm";
            break;
        case ProfileKind::SolidBar:
            out << "Bar " << (a * 1000.0) << "x" << (b * 1000.0) << "mm";
            break;
        }

        return out.str();
    }

    MemberProfile MakeBoxProfile(double outerWidth, double outerHeight, double wallThickness)
    {
        MemberProfile profile;
        profile.kind = ProfileKind::Box;
        profile.a = outerWidth;
        profile.b = outerHeight;
        profile.t = wallThickness;
        return profile;
    }

    MemberProfile MakeTubeProfile(double outerRadius, double wallThickness)
    {
        MemberProfile profile;
        profile.kind = ProfileKind::Tube;
        profile.a = outerRadius;
        profile.t = wallThickness;
        return profile;
    }

    MemberProfile MakeIBeamProfile(double height, double flangeWidth, double webThickness, double flangeThickness)
    {
        MemberProfile profile;
        profile.kind = ProfileKind::IBeam;
        profile.a = height;
        profile.b = flangeWidth;
        profile.t = webThickness;
        profile.t2 = flangeThickness;
        return profile;
    }

    MemberProfile MakeAngleProfile(double legA, double legB, double thickness)
    {
        MemberProfile profile;
        profile.kind = ProfileKind::Angle;
        profile.a = legA;
        profile.b = legB;
        profile.t = thickness;
        return profile;
    }

    MemberProfile MakeChannelProfile(double height, double flangeWidth, double webThickness, double flangeThickness)
    {
        MemberProfile profile;
        profile.kind = ProfileKind::Channel;
        profile.a = height;
        profile.b = flangeWidth;
        profile.t = webThickness;
        profile.t2 = flangeThickness;
        return profile;
    }

    MemberProfile MakeSolidBarProfile(double width, double height)
    {
        MemberProfile profile;
        profile.kind = ProfileKind::SolidBar;
        profile.a = width;
        profile.b = height;
        return profile;
    }

} // namespace Ship
