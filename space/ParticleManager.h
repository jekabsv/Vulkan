#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

namespace Core
{

    using ParticleId = size_t;

    // Plain value describing a particle. Used both when creating a particle and when
    // reading one back out of the structure-of-arrays storage.
    struct ParticleState
    {
        glm::vec3 position{ 0.0f, 0.0f, 0.0f };
        glm::vec3 velocity{ 0.0f, 0.0f, 0.0f };
        float mass{ 1.0f };
    };

    class ParticleManager;

    // Implemented by whatever owns the device-side copy of the particles (ParticleSimulation).
    // Going through this interface keeps ParticleManager free of any Vulkan dependency, so it
    // stays usable — and testable — with no GPU attached at all.
    class IParticleSyncSource
    {
    public:
        virtual ~IParticleSyncSource() = default;

        // Halt the device, copy the live state back, and overwrite the manager's arrays.
        virtual bool AcquireLatest(ParticleManager& particles) = 0;

        // Fill the manager from the newest snapshot that has already completed. No stall, and
        // the simulation keeps running, so the data is a few frames old.
        virtual bool PeekLatest(ParticleManager& particles) = 0;

        // Push host edits back to the device and let it run again.
        virtual void ReleaseToDevice(ParticleManager& particles) = 0;
    };

    // CPU-side particle storage.
    //
    // Particles live in parallel arrays (positions / velocities / masses / ids) so the
    // simulation can walk one attribute at a time and so a whole array can later be
    // handed to the GPU as a single upload. Handles are stable ParticleIds; the packed
    // index of a particle is not, because removal swaps the last particle into the hole.
    class ParticleManager
    {
    public:
        static constexpr ParticleId InvalidId = static_cast<ParticleId>(-1);
        static constexpr size_t InvalidIndex = static_cast<size_t>(-1);

        ParticleManager() = default;
        explicit ParticleManager(size_t capacity);

        // --- Creation / removal ---

        // Returns the handle of the new particle. Ids are never recycled.
        ParticleId CreateParticle(const ParticleState& state = {});
        ParticleId CreateParticle(const glm::vec3& position, const glm::vec3& velocity = { 0.0f, 0.0f, 0.0f }, float mass = 1.0f);

        // False if no particle carries this handle.
        bool RemoveParticle(ParticleId id);

        void Clear();
        void Reserve(size_t capacity);

        // --- Device synchronisation ---
        // Without a sync source attached the manager is the only copy of the data and all of
        // these are trivially satisfied, so the same call site works with or without a GPU.

        void AttachSyncSource(IParticleSyncSource* source) { m_SyncSource = source; }
        IParticleSyncSource* GetSyncSource() const { return m_SyncSource; }

        // Blocking. Halts the simulation, refreshes the arrays from the device, and returns true
        // once the mirror is exact and safe to edit. Pair every true with Resume().
        //
        //   if (particles.GetLatest())
        //   {
        //       particles.SetPosition(id, p);
        //       particles.CreateParticle(...);
        //       particles.Resume();
        //   }
        bool GetLatest();

        // Non-blocking. Refreshes from the last completed snapshot (several frames stale) and
        // leaves the simulation running. Deliberately does NOT mark the mirror fresh: this is for
        // reading — inspection, picking, UI — never for a read-modify-write cycle.
        bool PeekLatest();

        // Uploads whatever changed and releases the device to run again.
        void Resume();

        // True when the mirror matches the device exactly and edits will not clobber simulation
        // progress. Always true when no sync source is attached.
        bool IsMirrorFresh() const { return m_SyncSource == nullptr || m_MirrorFresh; }

        bool IsHostDirty() const { return m_HostDirty; }
        bool IsStructureDirty() const { return m_StructureDirty; }

        // Any value written on the host. Call after writing through the Mutable* accessors.
        void MarkHostDirty() { m_HostDirty = true; }
        // The particle count changed, so device buffers may need reallocating.
        void MarkStructureDirty() { m_HostDirty = true; m_StructureDirty = true; }
        void ClearDirtyFlags() { m_HostDirty = false; m_StructureDirty = false; }

        // Called by the sync source once device state has been written into the arrays. Marks the
        // mirror fresh and clears the dirty flags, because this data came *from* the device.
        // Sizes must all equal GetCount().
        void AdoptDeviceState(
            const std::vector<glm::vec3>& positions,
            const std::vector<glm::vec3>& velocities,
            const std::vector<float>& masses,
            bool markFresh);

        void SetMirrorFresh(bool fresh) { m_MirrorFresh = fresh; }

        // --- Queries ---

        bool HasParticle(ParticleId id) const;
        size_t GetCount() const { return m_Ids.size(); }
        bool IsEmpty() const { return m_Ids.empty(); }

        // Packed index of a handle, or InvalidIndex. Only valid until the next removal.
        size_t IndexOf(ParticleId id) const;
        // Handle stored at a packed index, or InvalidId if the index is out of range.
        ParticleId GetIdAt(size_t index) const;

        // --- Per-particle get / set ---
        // Every accessor returns false and leaves its output untouched when the handle
        // is unknown, so a stale handle is never a crash.

        bool GetParticle(ParticleId id, ParticleState& outState) const;
        bool SetParticle(ParticleId id, const ParticleState& state);

        bool GetPosition(ParticleId id, glm::vec3& outPosition) const;
        bool SetPosition(ParticleId id, const glm::vec3& position);

        bool GetVelocity(ParticleId id, glm::vec3& outVelocity) const;
        bool SetVelocity(ParticleId id, const glm::vec3& velocity);

        bool GetMass(ParticleId id, float& outMass) const;
        bool SetMass(ParticleId id, float mass);

        // --- Bulk access ---
        // Parallel arrays, all of size GetCount(); entry i of each describes one particle.

        const std::vector<glm::vec3>& GetPositions() const { return m_Positions; }
        const std::vector<glm::vec3>& GetVelocities() const { return m_Velocities; }
        const std::vector<float>& GetMasses() const { return m_Masses; }
        const std::vector<ParticleId>& GetIds() const { return m_Ids; }

        // Writable whole-array access, for a host-side integrator or bulk edit. Named rather than
        // overloaded on constness so that taking a write handle is deliberate and can mark the
        // data dirty — an overload would let a write slip through as if it were a read. Do not
        // resize these: that would desynchronise the handle map.
        std::vector<glm::vec3>& MutablePositions() { MarkHostDirty(); return m_Positions; }
        std::vector<glm::vec3>& MutableVelocities() { MarkHostDirty(); return m_Velocities; }
        std::vector<float>& MutableMasses() { MarkHostDirty(); return m_Masses; }

    private:
        std::vector<glm::vec3> m_Positions;
        std::vector<glm::vec3> m_Velocities;
        std::vector<float> m_Masses;
        std::vector<ParticleId> m_Ids;

        std::unordered_map<ParticleId, size_t> m_IdToIndex;
        ParticleId m_NextId{ 1 };

        IParticleSyncSource* m_SyncSource{ nullptr };
        bool m_HostDirty{ false };
        bool m_StructureDirty{ false };
        bool m_MirrorFresh{ false };
    };

} // namespace Core
