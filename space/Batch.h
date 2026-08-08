#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "Buffer.h"

namespace Core
{

    class VulkanContext;
    class Mesh;
    class Material;
    struct VertexInputLayout;

    // Groups draws that share a (Mesh, Material) pair and uploads their per-instance data into a
    // per-frame instance vertex buffer, so the whole group can be drawn with one instanced
    // vkCmdDrawIndexed instead of one draw call each. Per-instance data is an opaque byte blob
    // with a fixed stride set at construction, so this isn't limited to a plain transform - e.g.
    // Font uses it with a stride of sizeof(GlyphInstance) to also carry per-letter UV rect/color.
    class InstanceBatch
    {
    public:
        InstanceBatch(VulkanContext& context, const Mesh& mesh, Material& material, uint32_t frameCount, uint32_t instanceStride);

        InstanceBatch(const InstanceBatch&) = delete;
        InstanceBatch& operator=(const InstanceBatch&) = delete;
        InstanceBatch(InstanceBatch&& other) noexcept = default;
        InstanceBatch& operator=(InstanceBatch&& other) noexcept = default;

        void Add(const void* instanceData, uint32_t size);
        void Clear();

        void Flush(uint32_t frameIndex);
        void Bind(VkCommandBuffer commandBuffer, uint32_t frameIndex, uint32_t binding) const;

        uint32_t GetInstanceCount() const { return m_Stride == 0 ? 0 : static_cast<uint32_t>(m_InstanceData.size() / m_Stride); }
        uint32_t GetStride() const { return m_Stride; }
        const Mesh& GetMesh() const { return *m_Mesh; }
        Material& GetMaterial() const { return *m_Material; }

        // mat4-only per-instance transform (matches triangle_instanced.vert).
        static void AppendInstanceLayout(VertexInputLayout& layout, uint32_t binding, uint32_t startLocation);

    private:
        VulkanContext* m_Context{ nullptr };
        const Mesh* m_Mesh{ nullptr };
        Material* m_Material{ nullptr };

        std::vector<uint8_t> m_InstanceData;
        uint32_t m_Stride{ 0 };

        std::vector<Buffer> m_InstanceBuffers;
        std::vector<uint32_t> m_BufferCapacities;
    };

} // namespace Core
