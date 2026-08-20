#include "ParticleManager.h"



ParticleManager::ParticleManager(size_t capacity)
{
    Reserve(capacity);
}

void ParticleManager::Reserve(size_t capacity)
{
    m_Positions.reserve(capacity);
    m_Velocities.reserve(capacity);
    m_Masses.reserve(capacity);
    m_Ids.reserve(capacity);
    m_IdToIndex.reserve(capacity);
}

ParticleId ParticleManager::CreateParticle(const ParticleState& state)
{
    const ParticleId id = m_NextId++;
    const size_t index = m_Ids.size();

    m_Positions.push_back(state.position);
    m_Velocities.push_back(state.velocity);
    m_Masses.push_back(state.mass);
    m_Ids.push_back(id);
    m_IdToIndex.emplace(id, index);

    return id;
}

ParticleId ParticleManager::CreateParticle(const glm::vec3& position, const glm::vec3& velocity, float mass)
{
    ParticleState state;
    state.position = position;
    state.velocity = velocity;
    state.mass = mass;
    return CreateParticle(state);
}

bool ParticleManager::RemoveParticle(ParticleId id)
{
    auto it = m_IdToIndex.find(id);

    if (it == m_IdToIndex.end())
    {
        return false;
    }

    const size_t index = it->second;
    const size_t last = m_Ids.size() - 1;

    if (index != last)
    {
        m_Positions[index] = m_Positions[last];
        m_Velocities[index] = m_Velocities[last];
        m_Masses[index] = m_Masses[last];
        m_Ids[index] = m_Ids[last];
        m_IdToIndex[m_Ids[index]] = index;
    }

    m_Positions.pop_back();
    m_Velocities.pop_back();
    m_Masses.pop_back();
    m_Ids.pop_back();
    m_IdToIndex.erase(it);

    return true;
}

void ParticleManager::Clear()
{
    m_Positions.clear();
    m_Velocities.clear();
    m_Masses.clear();
    m_Ids.clear();
    m_IdToIndex.clear();
}

bool ParticleManager::HasParticle(ParticleId id) const
{
    return m_IdToIndex.find(id) != m_IdToIndex.end();
}

size_t ParticleManager::IndexOf(ParticleId id) const
{
    auto it = m_IdToIndex.find(id);
    return it == m_IdToIndex.end() ? InvalidIndex : it->second;
}

ParticleId ParticleManager::GetIdAt(size_t index) const
{
    return index < m_Ids.size() ? m_Ids[index] : InvalidId;
}

bool ParticleManager::GetParticle(ParticleId id, ParticleState& outState) const
{
    const size_t index = IndexOf(id);

    if (index == InvalidIndex)
    {
        return false;
    }

    outState.position = m_Positions[index];
    outState.velocity = m_Velocities[index];
    outState.mass = m_Masses[index];

    return true;
}

bool ParticleManager::SetParticle(ParticleId id, const ParticleState& state)
{
    const size_t index = IndexOf(id);

    if (index == InvalidIndex)
    {
        return false;
    }

    m_Positions[index] = state.position;
    m_Velocities[index] = state.velocity;
    m_Masses[index] = state.mass;

    return true;
}

bool ParticleManager::GetPosition(ParticleId id, glm::vec3& outPosition) const
{
    const size_t index = IndexOf(id);

    if (index == InvalidIndex)
    {
        return false;
    }

    outPosition = m_Positions[index];
    return true;
}

bool ParticleManager::SetPosition(ParticleId id, const glm::vec3& position)
{
    const size_t index = IndexOf(id);

    if (index == InvalidIndex)
    {
        return false;
    }

    m_Positions[index] = position;
    return true;
}

bool ParticleManager::GetVelocity(ParticleId id, glm::vec3& outVelocity) const
{
    const size_t index = IndexOf(id);

    if (index == InvalidIndex)
    {
        return false;
    }

    outVelocity = m_Velocities[index];
    return true;
}

bool ParticleManager::SetVelocity(ParticleId id, const glm::vec3& velocity)
{
    const size_t index = IndexOf(id);

    if (index == InvalidIndex)
    {
        return false;
    }

    m_Velocities[index] = velocity;
    return true;
}

bool ParticleManager::GetMass(ParticleId id, float& outMass) const
{
    const size_t index = IndexOf(id);

    if (index == InvalidIndex)
    {
        return false;
    }

    outMass = m_Masses[index];
    return true;
}

bool ParticleManager::SetMass(ParticleId id, float mass)
{
    const size_t index = IndexOf(id);

    if (index == InvalidIndex)
    {
        return false;
    }

    m_Masses[index] = mass;
    return true;
}