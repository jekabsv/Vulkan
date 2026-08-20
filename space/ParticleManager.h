#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

using ParticleId = size_t;

struct ParticleState
{
    glm::vec3 position{ 0.0f, 0.0f, 0.0f };
    glm::vec3 velocity{ 0.0f, 0.0f, 0.0f };
    float mass{ 1.0f };
};

class ParticleManager
{
public:
    static constexpr ParticleId InvalidId = static_cast<ParticleId>(-1);
    static constexpr size_t InvalidIndex = static_cast<size_t>(-1);

    ParticleManager() = default;
    explicit ParticleManager(size_t capacity);

    ParticleId CreateParticle(const ParticleState& state = {});
    ParticleId CreateParticle(const glm::vec3& position, const glm::vec3& velocity = { 0.0f, 0.0f, 0.0f }, float mass = 1.0f);

    bool RemoveParticle(ParticleId id);

    void Clear();
    void Reserve(size_t capacity);

    bool HasParticle(ParticleId id) const;
    size_t GetCount() const { return m_Ids.size(); }
    bool IsEmpty() const { return m_Ids.empty(); }

    size_t IndexOf(ParticleId id) const;
    ParticleId GetIdAt(size_t index) const;


    bool GetParticle(ParticleId id, ParticleState& outState) const;
    bool SetParticle(ParticleId id, const ParticleState& state);

    bool GetPosition(ParticleId id, glm::vec3& outPosition) const;
    bool SetPosition(ParticleId id, const glm::vec3& position);

    bool GetVelocity(ParticleId id, glm::vec3& outVelocity) const;
    bool SetVelocity(ParticleId id, const glm::vec3& velocity);

    bool GetMass(ParticleId id, float& outMass) const;
    bool SetMass(ParticleId id, float mass);


    const std::vector<glm::vec3>& GetPositions() const { return m_Positions; }
    const std::vector<glm::vec3>& GetVelocities() const { return m_Velocities; }
    const std::vector<float>& GetMasses() const { return m_Masses; }
    const std::vector<ParticleId>& GetIds() const { return m_Ids; }

    std::vector<glm::vec3>& GetPositions() { return m_Positions; }
    std::vector<glm::vec3>& GetVelocities() { return m_Velocities; }
    std::vector<float>& GetMasses() { return m_Masses; }

private:
    std::vector<glm::vec3> m_Positions;
    std::vector<glm::vec3> m_Velocities;
    std::vector<float> m_Masses;
    std::vector<ParticleId> m_Ids;

    std::unordered_map<ParticleId, size_t> m_IdToIndex;
    ParticleId m_NextId{ 1 };
};