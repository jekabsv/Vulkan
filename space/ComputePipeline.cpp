#include "ComputePipeline.h"

#include "Buffer.h"
#include "Shader.h"
#include "VulkanContext.h"

#include <stdexcept>
#include <utility>

namespace Core
{

    namespace
    {
        void CheckResult(VkResult result, const char* message)
        {
            if (result != VK_SUCCESS)
            {
                throw std::runtime_error(message);
            }
        }
    }

    ComputePipeline::ComputePipeline(VulkanContext& context, const Shader& shader)
        : Pipeline(context)
    {
        if (shader.GetStage() != VK_SHADER_STAGE_COMPUTE_BIT)
        {
            throw std::runtime_error("ComputePipeline requires a shader compiled for the compute stage");
        }

        CreateLayout({ &shader });
        CreatePipeline(shader);
    }

    ComputePipeline::~ComputePipeline()
    {
        Destroy();
    }

    ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
        : Pipeline(std::move(other))
    {
    }

    ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        Pipeline::operator=(std::move(other));

        return *this;
    }

    void ComputePipeline::CreatePipeline(const Shader& shader)
    {
        VkComputePipelineCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        createInfo.stage = shader.GetStageCreateInfo();
        createInfo.layout = m_Layout;

        CheckResult(
            vkCreateComputePipelines(m_Context->GetDevice(), VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_Pipeline),
            "Failed to create compute pipeline");
    }

    void ComputePipeline::Dispatch(VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) const
    {
        vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);
    }

    void ComputePipeline::DispatchIndirect(VkCommandBuffer commandBuffer, const Buffer& buffer, VkDeviceSize offset) const
    {
        vkCmdDispatchIndirect(commandBuffer, buffer.GetHandle(), offset);
    }

    void ComputePipeline::BufferBarrier(
        VkCommandBuffer commandBuffer,
        const Buffer& buffer,
        VkPipelineStageFlags2 srcStage,
        VkAccessFlags2 srcAccess,
        VkPipelineStageFlags2 dstStage,
        VkAccessFlags2 dstAccess)
    {
        VkBufferMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barrier.srcStageMask = srcStage;
        barrier.srcAccessMask = srcAccess;
        barrier.dstStageMask = dstStage;
        barrier.dstAccessMask = dstAccess;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = buffer.GetHandle();
        barrier.offset = 0;
        barrier.size = buffer.GetSize();

        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.bufferMemoryBarrierCount = 1;
        dependency.pBufferMemoryBarriers = &barrier;

        vkCmdPipelineBarrier2(commandBuffer, &dependency);
    }

} // namespace Core