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

    class InstanceBatch
    {
    public:
        InstanceBatch(VulkanContext& context, const Mesh& mesh, Material& material, uint32_t frameCount);

        InstanceBatch(const InstanceBatch&) = delete;
        InstanceBatch& operator=(const InstanceBatch&) = delete;
        InstanceBatch(InstanceBatch&& other) noexcept = default;
        InstanceBatch& operator=(InstanceBatch&& other) noexcept = default;

        void Add(const glm::mat4& transform);
        void Clear();

        void Flush(uint32_t frameIndex);
        void Bind(VkCommandBuffer commandBuffer, uint32_t frameIndex, uint32_t binding) const;

        uint32_t GetInstanceCount() const { return static_cast<uint32_t>(m_Transforms.size()); }
        const Mesh& GetMesh() const { return *m_Mesh; }
        Material& GetMaterial() const { return *m_Material; }

        static void AppendInstanceLayout(VertexInputLayout& layout, uint32_t binding, uint32_t startLocation);

    private:
        VulkanContext* m_Context{ nullptr };
        const Mesh* m_Mesh{ nullptr };
        Material* m_Material{ nullptr };

        std::vector<glm::mat4> m_Transforms;
        std::vector<Buffer> m_InstanceBuffers;
        std::vector<uint32_t> m_BufferCapacities;
    };

} // namespace Core
