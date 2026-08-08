#include "Batch.h"

#include "Pipeline.h"

namespace Core
{

    namespace
    {
        constexpr uint32_t kMinInstanceCapacity = 16;
    }

    InstanceBatch::InstanceBatch(VulkanContext& context, const Mesh& mesh, Material& material, uint32_t frameCount)
        : m_Context(&context)
        , m_Mesh(&mesh)
        , m_Material(&material)
    {
        m_InstanceBuffers.reserve(frameCount);
        m_BufferCapacities.resize(frameCount, kMinInstanceCapacity);

        for (uint32_t i = 0; i < frameCount; i++)
        {
            m_InstanceBuffers.emplace_back(context, sizeof(glm::mat4) * kMinInstanceCapacity, BufferType::Vertex, MemoryUsage::HostVisible);
        }
    }

    void InstanceBatch::Add(const glm::mat4& transform)
    {
        m_Transforms.push_back(transform);
    }

    void InstanceBatch::Clear()
    {
        m_Transforms.clear();
    }

    void InstanceBatch::Flush(uint32_t frameIndex)
    {
        const uint32_t instanceCount = GetInstanceCount();

        if (instanceCount == 0)
        {
            return;
        }

        if (instanceCount > m_BufferCapacities[frameIndex])
        {
            uint32_t newCapacity = m_BufferCapacities[frameIndex];

            while (newCapacity < instanceCount)
            {
                newCapacity *= 2;
            }

            m_InstanceBuffers[frameIndex] = Buffer(*m_Context, sizeof(glm::mat4) * newCapacity, BufferType::Vertex, MemoryUsage::HostVisible);
            m_BufferCapacities[frameIndex] = newCapacity;
        }

        m_InstanceBuffers[frameIndex].SetData(m_Transforms.data(), sizeof(glm::mat4) * instanceCount);
    }

    void InstanceBatch::Bind(VkCommandBuffer commandBuffer, uint32_t frameIndex, uint32_t binding) const
    {
        m_InstanceBuffers[frameIndex].BindAsVertexBuffer(commandBuffer, binding);
    }

    void InstanceBatch::AppendInstanceLayout(VertexInputLayout& layout, uint32_t binding, uint32_t startLocation)
    {
        layout.AddBinding(binding, sizeof(glm::mat4), VK_VERTEX_INPUT_RATE_INSTANCE);

        for (uint32_t column = 0; column < 4; column++)
        {
            layout.AddAttribute(startLocation + column, binding, VK_FORMAT_R32G32B32A32_SFLOAT, static_cast<uint32_t>(sizeof(glm::vec4) * column));
        }
    }

} // namespace Core
