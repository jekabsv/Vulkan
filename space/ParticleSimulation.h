#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "Buffer.h"
#include "DescriptorManager.h"
#include "ParticleManager.h"

namespace Core
{

    class VulkanContext;
    class ComputePipeline;

    // Device-side particle layout. Position and mass share a vec4, and velocity is padded to one,
    // because std430 rounds vec3 up to 16 bytes anyway — packing makes that padding carry data and
    // halves the fetches in the force loop. This deliberately differs from ParticleManager's
    // structure-of-arrays layout; the upload and readback paths convert between the two.
    struct GpuParticle
    {
        glm::vec4 positionMass{ 0.0f, 0.0f, 0.0f, 1.0f };
        glm::vec4 velocity{ 0.0f, 0.0f, 0.0f, 0.0f };
    };

    enum class SimulationState : uint8_t
    {
        // Device owns the data and dispatches every frame.
        Running = 0,
        // Host owns the data. No dispatches are recorded until Resume().
        Halted
    };

    struct ParticleSimulationConfig
    {
        uint32_t capacity{ 1u << 16 };
        uint32_t localSizeX{ 256 };
        uint32_t framesInFlight{ 2 };

        float gravity{ 1.0f };
        // Plummer softening: keeps the 1/r^2 force finite when two particles coincide.
        float softening{ 0.05f };
    };

    // Owns the GPU-resident particle state and the host/device sync state machine.
    //
    // The steady state is that nothing crosses back to the host: compute integrates in place and
    // the vertex shader reads the same buffer. Data only comes back when the host asks, and the
    // "halt" is not a device-side wait — no such thing exists in Vulkan — it is simply this class
    // declining to record dispatches until Resume().
    class ParticleSimulation : public IParticleSyncSource
    {
    public:
        ParticleSimulation(VulkanContext& context, const ComputePipeline& pipeline, const ParticleSimulationConfig& config);
        ParticleSimulation(VulkanContext& context, const ComputePipeline& pipeline);
        ~ParticleSimulation() override = default;

        ParticleSimulation(const ParticleSimulation&) = delete;
        ParticleSimulation& operator=(const ParticleSimulation&) = delete;

        // --- Frame recording ---
        // Call between Renderer::BeginFrame() and Renderer::BeginRenderPass(), in this order:
        //
        //   particles.PeekLatest();            // optional, must precede RecordCompute
        //   simulation.RecordCompute(cmd, dt);
        //   simulation.RecordPeekCopy(cmd);    // optional
        //   ... render, reading GetRenderBuffer() ...
        //   simulation.AdvanceFrame();         // after Renderer::EndFrame()

        // Dispatches the integration pass and barriers its output for the vertex shader.
        // A no-op while halted or empty.
        void RecordCompute(VkCommandBuffer commandBuffer, float deltaTime);

        // Copies this frame's result into the peek ring so PeekLatest() has something to read.
        // Costs one buffer copy per frame; skip it if nothing peeks.
        void RecordPeekCopy(VkCommandBuffer commandBuffer);

        // Flips the ping-pong pair and advances the peek ring. Call once per submitted frame.
        void AdvanceFrame();

        // --- IParticleSyncSource ---

        // Call these outside the BeginFrame/EndFrame pair. AcquireLatest reads the buffer that
        // AdvanceFrame() last promoted, so running it mid-recording would hand back the previous
        // frame's state while a newer dispatch sits half-recorded.
        bool AcquireLatest(ParticleManager& particles) override;
        bool PeekLatest(ParticleManager& particles) override;
        void ReleaseToDevice(ParticleManager& particles) override;

        // --- Explicit control ---

        // Replaces device state with the manager's arrays, reallocating if the count outgrew the
        // current capacity, and leaves the simulation running.
        void Upload(ParticleManager& particles);

        SimulationState GetState() const { return m_SimState; }
        bool IsHalted() const { return m_SimState == SimulationState::Halted; }

        uint32_t GetParticleCount() const { return m_Count; }
        uint32_t GetCapacity() const { return m_Config.capacity; }

        // The buffer holding the most recently written state — bind this for rendering. Valid
        // between RecordCompute() and AdvanceFrame(); it changes every frame.
        const Buffer& GetRenderBuffer() const { return m_State[m_WriteIndex]; }

        const ParticleSimulationConfig& GetConfig() const { return m_Config; }

    private:
        void CreateResources();
        void CreateDescriptorSets();
        void Grow(uint32_t requiredCapacity);
        void EncodeFrom(const ParticleManager& particles);
        void DecodeInto(ParticleManager& particles, const Buffer& source, bool markFresh);
        VkDeviceSize StateSize() const;

    private:
        VulkanContext* m_Context{ nullptr };
        const ComputePipeline* m_Pipeline{ nullptr };
        ParticleSimulationConfig m_Config;

        // Ping-pong pair: compute reads m_State[m_ReadIndex] and writes m_State[m_WriteIndex].
        // Two buffers are enough only while framesInFlight <= 2 — with more frames in flight a
        // third frame could overwrite a buffer an earlier frame's draw is still reading, so the
        // constructor rejects that rather than corrupting silently.
        std::vector<Buffer> m_State;
        uint32_t m_ReadIndex{ 0 };
        uint32_t m_WriteIndex{ 1 };

        std::vector<Buffer> m_Upload;
        std::vector<Buffer> m_Readback;

        // One staging buffer per frame in flight. Slot i is safe to read at the start of frame
        // i + framesInFlight, by which point the swapchain fence for that frame has been waited on.
        std::vector<Buffer> m_PeekRing;
        std::vector<bool> m_PeekValid;
        uint32_t m_PeekSlot{ 0 };

        DescriptorManager m_Descriptors;
        VkDescriptorSet m_ComputeSets[2]{ VK_NULL_HANDLE, VK_NULL_HANDLE };

        uint32_t m_Count{ 0 };
        SimulationState m_SimState{ SimulationState::Running };

        // Scratch for the interleave/deinterleave, kept as members so sync does not allocate.
        std::vector<GpuParticle> m_Scratch;
        std::vector<glm::vec3> m_ScratchPositions;
        std::vector<glm::vec3> m_ScratchVelocities;
        std::vector<float> m_ScratchMasses;
    };

} // namespace Core
