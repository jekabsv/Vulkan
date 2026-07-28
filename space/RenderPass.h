#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

namespace Core
{

    struct RenderAttachment
    {
        VkImageView imageView{ VK_NULL_HANDLE };
        VkImageLayout imageLayout{ VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
        VkAttachmentLoadOp loadOp{ VK_ATTACHMENT_LOAD_OP_CLEAR };
        VkAttachmentStoreOp storeOp{ VK_ATTACHMENT_STORE_OP_STORE };
        VkClearValue clearValue{};

        VkImageView resolveView{ VK_NULL_HANDLE };
        VkImageLayout resolveLayout{ VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
        VkResolveModeFlagBits resolveMode{ VK_RESOLVE_MODE_NONE };
    };

    class RenderPass
    {
    public:
        RenderPass() = default;

        RenderPass& AddColorAttachment(const RenderAttachment& attachment);
        RenderPass& AddColorAttachment(VkImageView imageView, const VkClearColorValue& clearColor);
        RenderPass& SetDepthAttachment(const RenderAttachment& attachment);
        RenderPass& SetDepthAttachment(VkImageView imageView, float clearDepth);
        RenderPass& SetStencilAttachment(const RenderAttachment& attachment);

        RenderPass& SetRenderArea(const VkRect2D& area);
        RenderPass& SetRenderArea(const VkExtent2D& extent);
        RenderPass& SetLayerCount(uint32_t layerCount);
        RenderPass& SetViewMask(uint32_t viewMask);

        void Reset();

        void Begin(VkCommandBuffer commandBuffer) const;
        void End(VkCommandBuffer commandBuffer) const;

        void SetViewportAndScissor(VkCommandBuffer commandBuffer, bool flipViewport) const;

        const VkRect2D& GetRenderArea() const { return m_RenderArea; }
        uint32_t GetColorAttachmentCount() const { return static_cast<uint32_t>(m_ColorAttachments.size()); }
        bool HasDepthAttachment() const { return m_HasDepth; }

        static void TransitionImageLayout(
            VkCommandBuffer commandBuffer,
            VkImage image,
            VkImageLayout oldLayout,
            VkImageLayout newLayout,
            VkImageAspectFlags aspectMask);

    private:
        static VkRenderingAttachmentInfo MakeAttachmentInfo(const RenderAttachment& attachment);

    private:
        std::vector<RenderAttachment> m_ColorAttachments;
        RenderAttachment m_DepthAttachment;
        RenderAttachment m_StencilAttachment;
        bool m_HasDepth{ false };
        bool m_HasStencil{ false };

        VkRect2D m_RenderArea{ { 0, 0 }, { 0, 0 } };
        uint32_t m_LayerCount{ 1 };
        uint32_t m_ViewMask{ 0 };
    };

} // namespace Core