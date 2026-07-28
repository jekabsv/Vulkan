#pragma once

// A Mark is a canonical hash of (class, profile, material, dimensions, end
// conditions). Two parts that hash the same are the same stock item: same
// cut, same tooling setup. The BOM the assembly emits is a count over Marks,
// not over parts, because tooling -- not raw part count -- is what standardization
// actually saves.

#include <cstdint>
#include <string>
#include <unordered_map>

namespace Ship
{

    using MarkId = uint64_t;

    // Quantizes a physical dimension (metres) to an integral key so that
    // float noise never splits one real-world stock size into two Marks.
    // 0.1 mm resolution is far finer than anything a grid-snapped design can
    // produce, so this never merges genuinely different parts either.
    int64_t QuantizeDimension(double metres);

    MarkId HashMemberMark(uint32_t profileKind, double a, double b, double t, double t2, uint32_t materialId, double length);
    MarkId HashPanelMark(uint32_t materialId, double thickness, double area, double perimeter, int openingCount, double openingArea);

    struct MarkEntry
    {
        std::string description;
        uint32_t count{ 0 };
        double unitMassKg{ 0.0 };
    };

    // Bill of materials keyed by Mark. Maintained incrementally by Assembly:
    // O(1) per part add/remove, never recomputed from scratch.
    class BillOfMaterials
    {
    public:
        void AddPart(MarkId mark, const std::string& description, double massKg);
        void RemovePart(MarkId mark, double massKg);

        size_t UniqueMarkCount() const { return m_Entries.size(); }
        const std::unordered_map<MarkId, MarkEntry>& Entries() const { return m_Entries; }

    private:
        std::unordered_map<MarkId, MarkEntry> m_Entries;
    };

} // namespace Ship
