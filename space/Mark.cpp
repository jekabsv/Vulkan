#include "Mark.h"

#include <cmath>
#include <cstring>

namespace Ship
{

    namespace
    {
        // 64-bit FNV-1a; deterministic across machines and runs, which matters
        // because Marks are meant to be reproducible, not just unique-in-process.
        void HashCombine(uint64_t& h, uint64_t value)
        {
            h ^= value;
            h *= 1099511628211ull;
        }

        uint64_t HashDouble(double v)
        {
            uint64_t bits;
            static_assert(sizeof(bits) == sizeof(v), "size mismatch");
            std::memcpy(&bits, &v, sizeof(v));
            return bits;
        }
    }

    int64_t QuantizeDimension(double metres)
    {
        return static_cast<int64_t>(std::llround(metres * 10000.0)); // 0.1 mm steps
    }

    MarkId HashMemberMark(uint32_t profileKind, double a, double b, double t, double t2, uint32_t materialId, double length)
    {
        uint64_t h = 1469598103934665603ull; // FNV offset basis
        HashCombine(h, 0x4d454d42u); // "MEMB" tag: distinguishes member marks from panel marks
        HashCombine(h, profileKind);
        HashCombine(h, static_cast<uint64_t>(QuantizeDimension(a)));
        HashCombine(h, static_cast<uint64_t>(QuantizeDimension(b)));
        HashCombine(h, static_cast<uint64_t>(QuantizeDimension(t)));
        HashCombine(h, static_cast<uint64_t>(QuantizeDimension(t2)));
        HashCombine(h, materialId);
        HashCombine(h, static_cast<uint64_t>(QuantizeDimension(length)));
        return h;
    }

    MarkId HashPanelMark(uint32_t materialId, double thickness, double area, double perimeter, int openingCount, double openingArea)
    {
        uint64_t h = 1469598103934665603ull;
        HashCombine(h, 0x50414e4cu); // "PANL" tag
        HashCombine(h, materialId);
        HashCombine(h, static_cast<uint64_t>(QuantizeDimension(thickness)));
        HashCombine(h, static_cast<uint64_t>(QuantizeDimension(area * 1000.0))); // area quantized coarser (mm^2-ish)
        HashCombine(h, static_cast<uint64_t>(QuantizeDimension(perimeter)));
        HashCombine(h, static_cast<uint64_t>(openingCount));
        HashCombine(h, static_cast<uint64_t>(QuantizeDimension(openingArea * 1000.0)));
        return h;
    }

    void BillOfMaterials::AddPart(MarkId mark, const std::string& description, double massKg)
    {
        MarkEntry& entry = m_Entries[mark];
        if (entry.count == 0)
        {
            entry.description = description;
            entry.unitMassKg = massKg;
        }
        entry.count++;
    }

    void BillOfMaterials::RemovePart(MarkId mark, double massKg)
    {
        auto it = m_Entries.find(mark);
        if (it == m_Entries.end())
        {
            return;
        }

        (void)massKg;
        if (--it->second.count == 0)
        {
            m_Entries.erase(it);
        }
    }

} // namespace Ship
