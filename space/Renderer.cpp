#include "Renderer.h"

#include "Material.h"
#include "Mesh.h"
#include "Pipeline.h"
#include "VulkanContext.h"
#include "Window.h"

#include <stdexcept>

namespace Core
{

    namespace
    {
        SwapchainConfig MakeSwapchainConfig(const RendererConfig& config)
        {
            SwapchainConfig swapchainConfig;
            swapchainConfig.vsync = config.vsync;
            swapchainConfig.desiredImageCount = config.desiredImageCount;
            swapchainConfig.maxFramesInFlight = config.maxFramesInFlight;
            return swapchainConfig;
        }
    }

    Renderer::Renderer(VulkanContext& context, Window& window, const RendererConfig& config)
        : m_Context(&context)
        , m_Window(&window)
        , m_Config(config)
        , m_Swapchain(context, window, MakeSwapchainConfig(config))
        , m_MaterialDescriptors(context, config.maxDescriptorSetsPerPool)
    {
        CreateCommandResources();
        CreateDepthResources();
        CreateCameraResources();
    }

    Renderer::Renderer(VulkanContext& context, Window& window)
        : Renderer(context, window, RendererConfig())
    {
    }

    Renderer::~Renderer()
    {
        m_Context->WaitIdle();

        if (m_CommandPool != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(m_Context->GetDevice(), m_CommandPool, nullptr);
            m_CommandPool = VK_NULL_HANDLE;
        }
    }

    void Renderer::CreateCommandResources()
    {
        VkCommandPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        createInfo.queueFamilyIndex = m_Context->GetQueueFamilyIndices().graphicsFamily;

        if (vkCreateCommandPool(m_Context->GetDevice(), &createInfo, nullptr, &m_CommandPool) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create renderer command pool");
        }

        m_CommandBuffers.resize(m_Swapchain.GetMaxFramesInFlight());

        VkCommandBufferAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocateInfo.commandPool = m_CommandPool;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = static_cast<uint32_t>(m_CommandBuffers.size());

        if (vkAllocateCommandBuffers(m_Context->GetDevice(), &allocateInfo, m_CommandBuffers.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate renderer command buffers");
        }
    }

    void Renderer::CreateDepthResources()
    {
        m_DepthFormat = m_Context->FindDepthFormat();

        TextureConfig config;
        config.width = m_Swapchain.GetWidth();
        config.height = m_Swapchain.GetHeight();
        config.depth = 1;
        config.format = m_DepthFormat;
        config.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        config.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        config.mipLevels = 1;
        config.generateMipmaps = false;
        config.createSampler = false;

        m_DepthTexture.clear();
        m_DepthTexture.reserve(1);
        m_DepthTexture.emplace_back(*m_Context, config);
    }

    void Renderer::CreateCameraResources()
    {
        const uint32_t frameCount = m_Swapchain.GetMaxFramesInFlight();

        m_CameraBuffers.reserve(frameCount);
        m_FrameDescriptors.reserve(frameCount);

        for (uint32_t i = 0; i < frameCount; i++)
        {
            m_CameraBuffers.emplace_back(*m_Context, sizeof(CameraUniforms), BufferType::Uniform, MemoryUsage::HostVisible);
            m_FrameDescriptors.emplace_back(*m_Context, m_Config.maxDescriptorSetsPerPool);
        }
    }

    void Renderer::RecreateSwapchainResources()
    {
        m_Swapchain.Recreate();
        CreateDepthResources();
    }

    void Renderer::SetVSync(bool enabled)
    {
        m_Swapchain.SetVSync(enabled);
        CreateDepthResources();
    }

    bool Renderer::BeginFrame()
    {
        if (m_Window->IsMinimized())
        {
            return false;
        }

        if (m_Window->WasResized())
        {
            RecreateSwapchainResources();
        }

        const SwapchainStatus status = m_Swapchain.AcquireNextImage(m_ImageIndex);

        if (status == SwapchainStatus::OutOfDate)
        {
            RecreateSwapchainResources();
            return false;
        }

        m_FrameIndex = m_Swapchain.GetCurrentFrame();
        m_BoundPipeline = nullptr;
        m_CameraDirty = true;

        m_FrameDescriptors[m_FrameIndex].ResetPools();

        VkCommandBuffer commandBuffer = m_CommandBuffers[m_FrameIndex];
        vkResetCommandBuffer(commandBuffer, 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to begin frame command buffer");
        }

        m_FrameStarted = true;
        return true;
    }

    void Renderer::BeginRenderPass(const glm::vec4& clearColor)
    {
        if (!m_FrameStarted)
        {
            throw std::runtime_error("BeginRenderPass called outside of a started frame");
        }

        VkCommandBuffer commandBuffer = m_CommandBuffers[m_FrameIndex];

        RenderPass::TransitionImageLayout(
            commandBuffer,
            m_Swapchain.GetImage(m_ImageIndex),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT);

        m_DepthTexture[0].TransitionLayout(commandBuffer, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

        VkClearColorValue clear{};
        clear.float32[0] = clearColor.r;
        clear.float32[1] = clearColor.g;
        clear.float32[2] = clearColor.b;
        clear.float32[3] = clearColor.a;

        m_RenderPass.Reset();
        m_RenderPass.AddColorAttachment(m_Swapchain.GetImageView(m_ImageIndex), clear);
        m_RenderPass.SetDepthAttachment(m_DepthTexture[0].GetImageView(), 1.0f);
        m_RenderPass.SetRenderArea(m_Swapchain.GetExtent());

        m_RenderPass.Begin(commandBuffer);
        m_RenderPass.SetViewportAndScissor(commandBuffer, true);
    }

    void Renderer::SetCamera(const Camera& camera)
    {
        m_CameraUniforms.view = camera.view;
        m_CameraUniforms.projection = camera.projection;
        m_CameraUniforms.viewProjection = camera.projection * camera.view;
        m_CameraUniforms.position = glm::vec4(camera.position, 1.0f);

        m_CameraBuffers[m_FrameIndex].SetData(&m_CameraUniforms, sizeof(CameraUniforms));

        m_CameraDirty = true;
    }

    void Renderer::BindCameraSet(const GraphicsPipeline& pipeline)
    {
        if (pipeline.GetDescriptorSetLayout(m_Config.cameraSet) == VK_NULL_HANDLE)
        {
            return;
        }

        if (pipeline.FindBinding(m_Config.cameraSet, m_Config.cameraBinding) == nullptr)
        {
            return;
        }

        DescriptorWriter writer = m_FrameDescriptors[m_FrameIndex].Begin(pipeline, m_Config.cameraSet);
        writer.WriteBuffer(m_Config.cameraBinding, m_CameraBuffers[m_FrameIndex]);

        const VkDescriptorSet cameraSet = writer.Build();

        std::vector<VkDescriptorSet> sets;
        sets.push_back(cameraSet);

        pipeline.BindDescriptorSets(m_CommandBuffers[m_FrameIndex], m_Config.cameraSet, sets);
    }

    void Renderer::DrawMesh(const Mesh& mesh, Material& material, const glm::mat4& transform)
    {
        if (!m_FrameStarted)
        {
            throw std::runtime_error("DrawMesh called outside of a started frame");
        }

        VkCommandBuffer commandBuffer = m_CommandBuffers[m_FrameIndex];
        const GraphicsPipeline& pipeline = material.GetPipeline();

        if (m_BoundPipeline != &pipeline)
        {
            pipeline.Bind(commandBuffer);
            m_BoundPipeline = &pipeline;
            m_CameraDirty = true;
        }

        if (m_CameraDirty)
        {
            BindCameraSet(pipeline);
            m_CameraDirty = false;
        }

        material.Flush();

        std::vector<VkDescriptorSet> sets;
        sets.push_back(material.GetDescriptorSet());
        pipeline.BindDescriptorSets(commandBuffer, material.GetSet(), sets);

        if (pipeline.HasPushConstants())
        {
            pipeline.PushConstants(commandBuffer, 0, sizeof(glm::mat4), &transform);
        }

        mesh.Bind(commandBuffer);
        vkCmdDrawIndexed(commandBuffer, mesh.GetIndexCount(), 1, 0, 0, 0);
    }

    void Renderer::BeginBatch()
    {
        for (auto& entry : m_ActiveBatches)
        {
            entry.second.Clear();
        }
    }

    void Renderer::Submit(const Mesh& mesh, Material& material, const glm::mat4& transform)
    {
        if (!m_FrameStarted)
        {
            throw std::runtime_error("Submit called outside of a started frame");
        }

        const BatchKey key{ &mesh, &material };
        auto it = m_ActiveBatches.find(key);

        if (it == m_ActiveBatches.end())
        {
            it = m_ActiveBatches.try_emplace(key, *m_Context, mesh, material, m_Swapchain.GetMaxFramesInFlight()).first;
        }

        it->second.Add(transform);
    }

    void Renderer::FlushBatch()
    {
        if (!m_FrameStarted)
        {
            throw std::runtime_error("FlushBatch called outside of a started frame");
        }

        VkCommandBuffer commandBuffer = m_CommandBuffers[m_FrameIndex];

        for (auto& entry : m_ActiveBatches)
        {
            InstanceBatch& batch = entry.second;

            if (batch.GetInstanceCount() == 0)
            {
                continue;
            }

            Material& material = batch.GetMaterial();
            const GraphicsPipeline& pipeline = material.GetPipeline();

            if (m_BoundPipeline != &pipeline)
            {
                pipeline.Bind(commandBuffer);
                m_BoundPipeline = &pipeline;
                m_CameraDirty = true;
            }

            if (m_CameraDirty)
            {
                BindCameraSet(pipeline);
                m_CameraDirty = false;
            }

            material.Flush();

            std::vector<VkDescriptorSet> sets;
            sets.push_back(material.GetDescriptorSet());
            pipeline.BindDescriptorSets(commandBuffer, material.GetSet(), sets);

            batch.Flush(m_FrameIndex);

            const Mesh& mesh = batch.GetMesh();
            mesh.Bind(commandBuffer);
            batch.Bind(commandBuffer, m_FrameIndex, 1);

            vkCmdDrawIndexed(commandBuffer, mesh.GetIndexCount(), batch.GetInstanceCount(), 0, 0, 0);
        }
    }

    void Renderer::EndRenderPass()
    {
        VkCommandBuffer commandBuffer = m_CommandBuffers[m_FrameIndex];

        m_RenderPass.End(commandBuffer);

        RenderPass::TransitionImageLayout(
            commandBuffer,
            m_Swapchain.GetImage(m_ImageIndex),
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_IMAGE_ASPECT_COLOR_BIT);
    }

    void Renderer::EndFrame()
    {
        if (!m_FrameStarted)
        {
            return;
        }

        VkCommandBuffer commandBuffer = m_CommandBuffers[m_FrameIndex];

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to end frame command buffer");
        }

        const SwapchainStatus status = m_Swapchain.SubmitAndPresent(commandBuffer, m_ImageIndex);

        m_FrameStarted = false;

        if (status != SwapchainStatus::Success || m_Window->WasResized())
        {
            RecreateSwapchainResources();
        }
    }

    Material Renderer::CreateMaterial(const GraphicsPipeline& pipeline, uint32_t set)
    {
        return Material(*m_Context, m_MaterialDescriptors, pipeline, set);
    }

    Material Renderer::CreateMaterial(const GraphicsPipeline& pipeline)
    {
        return Material(*m_Context, m_MaterialDescriptors, pipeline, 2);
    }

} // namespace Core