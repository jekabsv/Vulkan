#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "Batch.h"
#include "Buffer.h"
#include "DescriptorManager.h"
#include "RenderPass.h"
#include "Swapchain.h"
#include "Texture.h"

namespace Core
{

    class VulkanContext;
    class Window;
    class GraphicsPipeline;
    class Material;
    class Mesh;

    struct BatchKey
    {
        const Mesh* mesh{ nullptr };
        Material* material{ nullptr };

        bool operator==(const BatchKey& other) const
        {
            return mesh == other.mesh && material == other.material;
        }
    };

} // namespace Core

namespace std
{
    template<>
    struct hash<Core::BatchKey>
    {
        size_t operator()(const Core::BatchKey& key) const noexcept
        {
            const size_t meshHash = hash<const void*>()(key.mesh);
            const size_t materialHash = hash<const void*>()(key.material);
            return meshHash ^ (materialHash + 0x9e3779b9 + (meshHash << 6) + (meshHash >> 2));
        }
    };
}

namespace Core
{

    struct Camera
    {
        glm::mat4 view{ 1.0f };
        glm::mat4 projection{ 1.0f };
        glm::vec3 position{ 0.0f, 0.0f, 0.0f };
    };

    struct CameraUniforms
    {
        glm::mat4 view{ 1.0f };
        glm::mat4 projection{ 1.0f };
        glm::mat4 viewProjection{ 1.0f };
        glm::vec4 position{ 0.0f, 0.0f, 0.0f, 1.0f };
    };

    struct RendererConfig
    {
        bool vsync{ true };
        uint32_t desiredImageCount{ 3 };
        uint32_t maxFramesInFlight{ 2 };
        uint32_t cameraSet{ 0 };
        uint32_t cameraBinding{ 0 };
        uint32_t maxDescriptorSetsPerPool{ 1000 };
    };

    class Renderer
    {
    public:
        Renderer(VulkanContext& context, Window& window, const RendererConfig& config);
        Renderer(VulkanContext& context, Window& window);
        ~Renderer();

        Renderer(const Renderer&) = delete;
        Renderer& operator=(const Renderer&) = delete;
        Renderer(Renderer&&) = delete;
        Renderer& operator=(Renderer&&) = delete;

        bool BeginFrame();
        void BeginRenderPass(const glm::vec4& clearColor);
        void SetCamera(const Camera& camera);
        void DrawMesh(const Mesh& mesh, Material& material, const glm::mat4& transform);
        void DrawMesh(const Mesh& mesh, Material& material, const void* pushConstantData, uint32_t pushConstantSize);
        void EndRenderPass();
        void EndFrame();

        void BeginBatch();
        void Submit(const Mesh& mesh, Material& material, const glm::mat4& transform);
        void Submit(const Mesh& mesh, Material& material, const void* instanceData, uint32_t instanceDataSize);
        void FlushBatch();

        Material CreateMaterial(const GraphicsPipeline& pipeline, uint32_t set);
        Material CreateMaterial(const GraphicsPipeline& pipeline);

        VkFormat GetColorFormat() const { return m_Swapchain.GetImageFormat(); }
        VkFormat GetDepthFormat() const { return m_DepthFormat; }
        float GetAspectRatio() const { return m_Swapchain.GetAspectRatio(); }
        VkExtent2D GetExtent() const { return m_Swapchain.GetExtent(); }
        VkCommandBuffer GetCommandBuffer() const { return m_CommandBuffers[m_FrameIndex]; }
        Swapchain& GetSwapchain() { return m_Swapchain; }
        DescriptorManager& GetMaterialDescriptors() { return m_MaterialDescriptors; }
        void SetVSync(bool enabled);

    private:
        void CreateCommandResources();
        void CreateDepthResources();
        void CreateCameraResources();
        void RecreateSwapchainResources();
        void BindCameraSet(const GraphicsPipeline& pipeline);

    private:
        VulkanContext* m_Context{ nullptr };
        Window* m_Window{ nullptr };
        RendererConfig m_Config;

        Swapchain m_Swapchain;
        VkCommandPool m_CommandPool{ VK_NULL_HANDLE };
        std::vector<VkCommandBuffer> m_CommandBuffers;

        VkFormat m_DepthFormat{ VK_FORMAT_UNDEFINED };
        std::vector<Texture> m_DepthTexture;

        DescriptorManager m_MaterialDescriptors;
        std::vector<DescriptorManager> m_FrameDescriptors;
        std::vector<Buffer> m_CameraBuffers;

        RenderPass m_RenderPass;
        CameraUniforms m_CameraUniforms;

        uint32_t m_ImageIndex{ 0 };
        uint32_t m_FrameIndex{ 0 };
        bool m_FrameStarted{ false };
        bool m_CameraDirty{ true };
        const GraphicsPipeline* m_BoundPipeline{ nullptr };

        std::unordered_map<BatchKey, InstanceBatch> m_ActiveBatches;
    };

} // namespace Core