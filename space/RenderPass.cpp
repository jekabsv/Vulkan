#include "RenderPass.h"

namespace Core
{

    RenderPass& RenderPass::AddColorAttachment(const RenderAttachment& attachment)
    {
        m_ColorAttachments.push_back(attachment);
        return *this;
    }

    RenderPass& RenderPass::AddColorAttachment(VkImageView imageView, const VkClearColorValue& clearColor)
    {
        RenderAttachment attachment;
        attachment.imageView = imageView;
        attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.clearValue.color = clearColor;
        m_ColorAttachments.push_back(attachment);
        return *this;
    }

    RenderPass& RenderPass::SetDepthAttachment(const RenderAttachment& attachment)
    {
        m_DepthAttachment = attachment;
        m_HasDepth = true;
        return *this;
    }

    RenderPass& RenderPass::SetDepthAttachment(VkImageView imageView, float clearDepth)
    {
        RenderAttachment attachment;
        attachment.imageView = imageView;
        attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.clearValue.depthStencil.depth = clearDepth;
        attachment.clearValue.depthStencil.stencil = 0;
        m_DepthAttachment = attachment;
        m_HasDepth = true;
        return *this;
    }

    RenderPass& RenderPass::SetStencilAttachment(const RenderAttachment& attachment)
    {
        m_StencilAttachment = attachment;
        m_HasStencil = true;
        return *this;
    }

    RenderPass& RenderPass::SetRenderArea(const VkRect2D& area)
    {
        m_RenderArea = area;
        return *this;
    }

    RenderPass& RenderPass::SetRenderArea(const VkExtent2D& extent)
    {
        m_RenderArea.offset = { 0, 0 };
        m_RenderArea.extent = extent;
        return *this;
    }

    RenderPass& RenderPass::SetLayerCount(uint32_t layerCount)
    {
        m_LayerCount = layerCount;
        return *this;
    }

    RenderPass& RenderPass::SetViewMask(uint32_t viewMask)
    {
        m_ViewMask = viewMask;
        return *this;
    }

    void RenderPass::Reset()
    {
        m_ColorAttachments.clear();
        m_DepthAttachment = RenderAttachment();
        m_StencilAttachment = RenderAttachment();
        m_HasDepth = false;
        m_HasStencil = false;
        m_LayerCount = 1;
        m_ViewMask = 0;
    }

    VkRenderingAttachmentInfo RenderPass::MakeAttachmentInfo(const RenderAttachment& attachment)
    {
        VkRenderingAttachmentInfo info{};
        info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        info.imageView = attachment.imageView;
        info.imageLayout = attachment.imageLayout;
        info.loadOp = attachment.loadOp;
        info.storeOp = attachment.storeOp;
        info.clearValue = attachment.clearValue;
        info.resolveMode = attachment.resolveMode;
        info.resolveImageView = attachment.resolveView;
        info.resolveImageLayout = attachment.resolveLayout;
        return info;
    }

    void RenderPass::Begin(VkCommandBuffer commandBuffer) const
    {
        std::vector<VkRenderingAttachmentInfo> colorInfos;

        for (const RenderAttachment& attachment : m_ColorAttachments)
        {
            colorInfos.push_back(MakeAttachmentInfo(attachment));
        }

        VkRenderingAttachmentInfo depthInfo = MakeAttachmentInfo(m_DepthAttachment);
        VkRenderingAttachmentInfo stencilInfo = MakeAttachmentInfo(m_StencilAttachment);

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea = m_RenderArea;
        renderingInfo.layerCount = m_LayerCount;
        renderingInfo.viewMask = m_ViewMask;
        renderingInfo.colorAttachmentCount = static_cast<uint32_t>(colorInfos.size());
        renderingInfo.pColorAttachments = colorInfos.data();

        if (m_HasDepth)
        {
            renderingInfo.pDepthAttachment = &depthInfo;
        }

        if (m_HasStencil)
        {
            renderingInfo.pStencilAttachment = &stencilInfo;
        }

        vkCmdBeginRendering(commandBuffer, &renderingInfo);
    }

    void RenderPass::End(VkCommandBuffer commandBuffer) const
    {
        vkCmdEndRendering(commandBuffer);
    }

    void RenderPass::SetViewportAndScissor(VkCommandBuffer commandBuffer, bool flipViewport) const
    {
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.width = static_cast<float>(m_RenderArea.extent.width);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        if (flipViewport)
        {
            viewport.y = static_cast<float>(m_RenderArea.extent.height);
            viewport.height = -static_cast<float>(m_RenderArea.extent.height);
        }
        else
        {
            viewport.y = 0.0f;
            viewport.height = static_cast<float>(m_RenderArea.extent.height);
        }

        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &m_RenderArea);
    }

    void RenderPass::TransitionImageLayout(
        VkCommandBuffer commandBuffer,
        VkImage image,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        VkImageAspectFlags aspectMask)
    {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = aspectMask;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED)
        {
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_NONE;
        }

        if (newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        {
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        }

        if (newLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
        {
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        }

        if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        {
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        }

        if (newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
        {
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_NONE;
        }

        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &barrier;

        vkCmdPipelineBarrier2(commandBuffer, &dependency);
    }

} // namespace Core