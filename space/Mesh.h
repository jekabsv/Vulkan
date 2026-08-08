#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "Buffer.h"
#include "Pipeline.h"

namespace Core
{

    class VulkanContext;

    struct Vertex
    {
        glm::vec3 position{ 0.0f, 0.0f, 0.0f };
        glm::vec3 normal{ 0.0f, 1.0f, 0.0f };
        glm::vec2 uv{ 0.0f, 0.0f };

        static VertexInputLayout GetLayout();
    };

    class Mesh
    {
    public:
        Mesh(VulkanContext& context, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices);

        Mesh(const Mesh&) = delete;
        Mesh& operator=(const Mesh&) = delete;
        Mesh(Mesh&& other) noexcept = default;
        Mesh& operator=(Mesh&& other) noexcept = default;

        void Bind(VkCommandBuffer commandBuffer) const;

        uint32_t GetIndexCount() const { return m_IndexCount; }
        uint32_t GetVertexCount() const { return m_VertexCount; }

        static Mesh CreateCube(VulkanContext& context);
        static Mesh CreateQuad(VulkanContext& context);
        static Mesh CreateCustom(VulkanContext& context, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices);

    private:
        std::vector<Buffer> m_VertexBuffer;
        std::vector<Buffer> m_IndexBuffer;
        uint32_t m_VertexCount{ 0 };
        uint32_t m_IndexCount{ 0 };
    };

} // namespace Core