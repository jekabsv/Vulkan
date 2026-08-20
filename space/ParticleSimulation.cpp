#include "ParticleSimulation.h"

#include "ComputePipeline.h"
#include "VulkanContext.h"

#include <cstring>
#include <stdexcept>

namespace Core
{

    namespace
    {
        struct SimulationPushConstants
        {
            float deltaTime;
            float gravity;
            float softening;
            uint32_t count;
        };
    }

    ParticleSimulation::ParticleSimulation(VulkanContext& context, const ComputePipeline& pipeline, const ParticleSimulationConfig& config)
        : m_Context(&context)
        , m_Pipeline(&pipeline)
        , m_Config(config)
        , m_Descriptors(context, 8)
    {
        if (m_Config.capacity == 0)
        {
            throw std::runtime_error("Particle simulation needs a non-zero capacity");
        }

        if (m_Config.localSizeX == 0)
        {
            throw std::runtime_error("Particle simulation needs a non-zero workgroup size");
        }

        if (m_Config.framesInFlight == 0 || m_Config.framesInFlight > 2)
        {
            // See the m_State comment: the ping-pong pair only covers two frames of overlap.
            throw std::runtime_error("Particle simulation supports at most 2 frames in flight");
        }

        CreateResources();
    }

    ParticleSimulation::ParticleSimulation(VulkanContext& context, const ComputePipeline& pipeline)
        : ParticleSimulation(context, pipeline, ParticleSimulationConfig{})
    {
    }

    VkDeviceSize ParticleSimulation::StateSize() const
    {
        return sizeof(GpuParticle) * static_cast<VkDeviceSize>(m_Config.capacity);
    }

    void ParticleSimulation::CreateResources()
    {
        const VkDeviceSize size = StateSize();

        m_State.clear();
        m_State.reserve(2);
        m_State.emplace_back(*m_Context, size, BufferType::Storage, MemoryUsage::DeviceLocal);
        m_State.emplace_back(*m_Context, size, BufferType::Storage, MemoryUsage::DeviceLocal);

        m_Upload.clear();
        m_Upload.reserve(1);
        m_Upload.emplace_back(*m_Context, size, BufferType::Staging, MemoryUsage::HostVisible);

        m_Readback.clear();
        m_Readback.reserve(1);
        m_Readback.emplace_back(*m_Context, size, BufferType::Staging, MemoryUsage::HostReadback);

        m_PeekRing.clear();
        m_PeekRing.reserve(m_Config.framesInFlight);

        for (uint32_t i = 0; i < m_Config.framesInFlight; i++)
        {
            m_PeekRing.emplace_back(*m_Context, size, BufferType::Staging, MemoryUsage::HostReadback);
        }

        m_PeekValid.assign(m_Config.framesInFlight, false);
        m_PeekSlot = 0;

        m_ReadIndex = 0;
        m_WriteIndex = 1;

        CreateDescriptorSets();
    }

    void ParticleSimulation::CreateDescriptorSets()
    {
        // One set per ping-pong direction, so switching directions is a bind rather than a rewrite.
        m_Descriptors.ResetPools();

        m_ComputeSets[0] = m_Descriptors.Begin(*m_Pipeline, 0)
            .WriteBuffer(0, m_State[0])
            .WriteBuffer(1, m_State[1])
            .Build();

        m_ComputeSets[1] = m_Descriptors.Begin(*m_Pipeline, 0)
            .WriteBuffer(0, m_State[1])
            .WriteBuffer(1, m_State[0])
            .Build();
    }

    void ParticleSimulation::Grow(uint32_t requiredCapacity)
    {
        uint32_t capacity = m_Config.capacity;

        while (capacity < requiredCapacity)
        {
            capacity *= 2;
        }

        if (capacity == m_Config.capacity)
        {
            return;
        }

        // Buffers still referenced by in-flight command buffers must not be destroyed underneath
        // the device, and reallocation only happens while halted or during an explicit upload.
        m_Context->WaitIdle();

        m_Config.capacity = capacity;
        CreateResources();
    }

    void ParticleSimulation::RecordCompute(VkCommandBuffer commandBuffer, float deltaTime)
    {
        if (m_SimState == SimulationState::Halted || m_Count == 0)
        {
            return;
        }

        SimulationPushConstants push{};
        push.deltaTime = deltaTime;
        push.gravity = m_Config.gravity;
        push.softening = m_Config.softening;
        push.count = m_Count;

        m_Pipeline->Bind(commandBuffer);
        m_Pipeline->BindDescriptorSets(commandBuffer, 0, { m_ComputeSets[m_ReadIndex] });
        m_Pipeline->PushConstants(commandBuffer, 0, sizeof(push), &push);
        m_Pipeline->Dispatch(commandBuffer, (m_Count + m_Config.localSizeX - 1) / m_Config.localSizeX);

        // The draw later in this same command buffer reads what was just written, so the compute
        // write has to be visible to the vertex stage before it runs.
        ComputePipeline::BufferBarrier(
            commandBuffer,
            m_State[m_WriteIndex],
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    }

    void ParticleSimulation::RecordPeekCopy(VkCommandBuffer commandBuffer)
    {
        if (m_SimState == SimulationState::Halted || m_Count == 0)
        {
            return;
        }

        ComputePipeline::BufferBarrier(
            commandBuffer,
            m_State[m_WriteIndex],
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

        VkBufferCopy region{};
        region.size = sizeof(GpuParticle) * static_cast<VkDeviceSize>(m_Count);

        vkCmdCopyBuffer(
            commandBuffer,
            m_State[m_WriteIndex].GetHandle(),
            m_PeekRing[m_PeekSlot].GetHandle(),
            1,
            &region);

        m_PeekValid[m_PeekSlot] = true;
    }

    void ParticleSimulation::AdvanceFrame()
    {
        if (m_SimState == SimulationState::Halted)
        {
            return;
        }

        // What was written this frame becomes what is read next frame.
        const uint32_t previousRead = m_ReadIndex;
        m_ReadIndex = m_WriteIndex;
        m_WriteIndex = previousRead;

        m_PeekSlot = (m_PeekSlot + 1) % m_Config.framesInFlight;
    }

    void ParticleSimulation::EncodeFrom(const ParticleManager& particles)
    {
        const size_t count = particles.GetCount();

        const std::vector<glm::vec3>& positions = particles.GetPositions();
        const std::vector<glm::vec3>& velocities = particles.GetVelocities();
        const std::vector<float>& masses = particles.GetMasses();

        m_Scratch.resize(count);

        for (size_t i = 0; i < count; i++)
        {
            m_Scratch[i].positionMass = glm::vec4(positions[i], masses[i]);
            m_Scratch[i].velocity = glm::vec4(velocities[i], 0.0f);
        }
    }

    void ParticleSimulation::DecodeInto(ParticleManager& particles, const Buffer& source, bool markFresh)
    {
        const size_t count = m_Count;

        if (particles.GetCount() != count)
        {
            // The host added or removed particles without going through Upload, so index i on the
            // device no longer refers to the same particle as index i here. Overwriting would
            // silently scramble handles.
            throw std::runtime_error("Particle count drifted between host and device");
        }

        const GpuParticle* mapped = static_cast<const GpuParticle*>(source.GetMappedData());

        if (mapped == nullptr)
        {
            throw std::runtime_error("Readback buffer is not host visible");
        }

        m_ScratchPositions.resize(count);
        m_ScratchVelocities.resize(count);
        m_ScratchMasses.resize(count);

        for (size_t i = 0; i < count; i++)
        {
            m_ScratchPositions[i] = glm::vec3(mapped[i].positionMass);
            m_ScratchMasses[i] = mapped[i].positionMass.w;
            m_ScratchVelocities[i] = glm::vec3(mapped[i].velocity);
        }

        particles.AdoptDeviceState(m_ScratchPositions, m_ScratchVelocities, m_ScratchMasses, markFresh);
    }

    void ParticleSimulation::Upload(ParticleManager& particles)
    {
        const size_t count = particles.GetCount();

        if (count > m_Config.capacity)
        {
            Grow(static_cast<uint32_t>(count));
        }

        m_Count = static_cast<uint32_t>(count);

        if (m_Count > 0)
        {
            EncodeFrom(particles);

            const VkDeviceSize size = sizeof(GpuParticle) * static_cast<VkDeviceSize>(m_Count);
            m_Upload[0].SetData(m_Scratch.data(), size, 0);

            // Seed both halves of the ping-pong pair: the first frame after an upload reads one of
            // them, and whichever it is must hold the uploaded state.
            VkCommandBuffer commandBuffer = m_Context->BeginSingleTimeCommands();

            VkBufferCopy region{};
            region.size = size;

            vkCmdCopyBuffer(commandBuffer, m_Upload[0].GetHandle(), m_State[0].GetHandle(), 1, &region);
            vkCmdCopyBuffer(commandBuffer, m_Upload[0].GetHandle(), m_State[1].GetHandle(), 1, &region);

            m_Context->EndSingleTimeCommands(commandBuffer);
        }

        // Snapshots taken before the upload describe particles that no longer exist.
        m_PeekValid.assign(m_Config.framesInFlight, false);

        particles.ClearDirtyFlags();
        m_SimState = SimulationState::Running;
    }

    bool ParticleSimulation::AcquireLatest(ParticleManager& particles)
    {
        if (m_Count == 0)
        {
            m_SimState = SimulationState::Halted;
            particles.SetMirrorFresh(true);
            return true;
        }

        // Everything already submitted has to finish before the copy, otherwise it would read a
        // buffer a queued frame is still writing. This is the stall the halt is made of.
        m_Context->WaitIdle();

        // m_ReadIndex holds the newest state: AdvanceFrame() moved the last frame's output there.
        VkCommandBuffer commandBuffer = m_Context->BeginSingleTimeCommands();

        VkBufferCopy region{};
        region.size = sizeof(GpuParticle) * static_cast<VkDeviceSize>(m_Count);

        vkCmdCopyBuffer(commandBuffer, m_State[m_ReadIndex].GetHandle(), m_Readback[0].GetHandle(), 1, &region);

        // Submits and waits for the queue to drain.
        m_Context->EndSingleTimeCommands(commandBuffer);

        m_Readback[0].Invalidate(region.size, 0);

        // Halt before decoding: if the decode throws, the simulation must not resume on stale data.
        m_SimState = SimulationState::Halted;

        DecodeInto(particles, m_Readback[0], true);

        return true;
    }

    bool ParticleSimulation::PeekLatest(ParticleManager& particles)
    {
        if (m_Count == 0)
        {
            return false;
        }

        // The oldest slot is the one about to be reused this frame, which means the frame that
        // wrote it has been waited on. Reading any newer slot could race an in-flight copy.
        if (!m_PeekValid[m_PeekSlot])
        {
            return false;
        }

        const VkDeviceSize size = sizeof(GpuParticle) * static_cast<VkDeviceSize>(m_Count);
        m_PeekRing[m_PeekSlot].Invalidate(size, 0);

        // markFresh stays false: this snapshot is several frames old, so it is safe to read and
        // not safe to edit on top of.
        DecodeInto(particles, m_PeekRing[m_PeekSlot], false);

        return true;
    }

    void ParticleSimulation::ReleaseToDevice(ParticleManager& particles)
    {
        // Checked before the state test on purpose: edits made without halting first are still
        // edits, and dropping them because the simulation happened to be running would lose host
        // data silently. Uploading them is the lesser evil, and the count guard below catches the
        // case where those edits also changed the particle set.
        if (particles.IsHostDirty() || particles.IsStructureDirty() || particles.GetCount() != m_Count)
        {
            Upload(particles);
            return;
        }

        // Nothing changed, so device state still stands as it is.
        m_SimState = SimulationState::Running;
    }

} // namespace Core
