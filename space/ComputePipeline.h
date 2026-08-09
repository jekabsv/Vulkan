#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>

#include "Pipeline.h"

namespace Core
{

    class VulkanContext;
    class Shader;
    class Buffer;

    class ComputePipeline : public Pipeline
    {
    public:
        ComputePipeline(VulkanContext& context, const Shader& shader);
        ~ComputePipeline() override;

        ComputePipeline(const ComputePipeline&) = delete;
        ComputePipeline& operator=(const ComputePipeline&) = delete;
        ComputePipeline(ComputePipeline&& other) noexcept;
        ComputePipeline& operator=(ComputePipeline&& other) noexcept;

        // groupCount* is the number of workgroups to dispatch along each axis; each workgroup's
        // thread count is whatever local_size_x/y/z the shader declares.
        void Dispatch(VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1) const;
        void DispatchIndirect(VkCommandBuffer commandBuffer, const Buffer& buffer, VkDeviceSize offset = 0) const;

        // Whole-buffer memory barrier for sequencing compute reads/writes against later access to
        // the same storage buffer (another dispatch, a vertex/fragment read, a host readback copy).
        static void BufferBarrier(
            VkCommandBuffer commandBuffer,
            const Buffer& buffer,
            VkPipelineStageFlags2 srcStage,
            VkAccessFlags2 srcAccess,
            VkPipelineStageFlags2 dstStage,
            VkAccessFlags2 dstAccess);

    protected:
        VkPipelineBindPoint GetBindPoint() const override { return VK_PIPELINE_BIND_POINT_COMPUTE; }

    private:
        void CreatePipeline(const Shader& shader);
    };

} // namespace Core