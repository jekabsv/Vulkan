#include "Mesh.h"

#include "VulkanContext.h"

#include <cstddef>
#include <stdexcept>

namespace Core
{

    VertexInputLayout Vertex::GetLayout()
    {
        VertexInputLayout layout;
        layout.AddBinding(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX);
        layout.AddAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position));
        layout.AddAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal));
        layout.AddAttribute(2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv));
        return layout;
    }

    Mesh::Mesh(VulkanContext& context, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices)
    {
        if (vertices.empty() || indices.empty())
        {
            throw std::runtime_error("Cannot create a mesh without vertices and indices");
        }

        m_VertexCount = static_cast<uint32_t>(vertices.size());
        m_IndexCount = static_cast<uint32_t>(indices.size());

        const VkDeviceSize vertexSize = sizeof(Vertex) * vertices.size();
        const VkDeviceSize indexSize = sizeof(uint32_t) * indices.size();

        m_VertexBuffer.reserve(1);
        m_VertexBuffer.emplace_back(context, vertexSize, BufferType::Vertex, MemoryUsage::DeviceLocal);
        m_VertexBuffer[0].SetData(vertices.data(), vertexSize);

        m_IndexBuffer.reserve(1);
        m_IndexBuffer.emplace_back(context, indexSize, BufferType::Index, MemoryUsage::DeviceLocal);
        m_IndexBuffer[0].SetData(indices.data(), indexSize);
    }

    void Mesh::Bind(VkCommandBuffer commandBuffer) const
    {
        m_VertexBuffer[0].BindAsVertexBuffer(commandBuffer, 0);
        m_IndexBuffer[0].BindAsIndexBuffer(commandBuffer, VK_INDEX_TYPE_UINT32);
    }

    Mesh Mesh::CreateCube(VulkanContext& context)
    {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        const glm::vec3 normals[6] =
        {
            {  0.0f,  0.0f,  1.0f },
            {  0.0f,  0.0f, -1.0f },
            {  1.0f,  0.0f,  0.0f },
            { -1.0f,  0.0f,  0.0f },
            {  0.0f,  1.0f,  0.0f },
            {  0.0f, -1.0f,  0.0f }
        };

        const glm::vec3 faceVertices[6][4] =
        {
            { { -0.5f, -0.5f,  0.5f }, {  0.5f, -0.5f,  0.5f }, {  0.5f,  0.5f,  0.5f }, { -0.5f,  0.5f,  0.5f } },
            { {  0.5f, -0.5f, -0.5f }, { -0.5f, -0.5f, -0.5f }, { -0.5f,  0.5f, -0.5f }, {  0.5f,  0.5f, -0.5f } },
            { {  0.5f, -0.5f,  0.5f }, {  0.5f, -0.5f, -0.5f }, {  0.5f,  0.5f, -0.5f }, {  0.5f,  0.5f,  0.5f } },
            { { -0.5f, -0.5f, -0.5f }, { -0.5f, -0.5f,  0.5f }, { -0.5f,  0.5f,  0.5f }, { -0.5f,  0.5f, -0.5f } },
            { { -0.5f,  0.5f,  0.5f }, {  0.5f,  0.5f,  0.5f }, {  0.5f,  0.5f, -0.5f }, { -0.5f,  0.5f, -0.5f } },
            { { -0.5f, -0.5f, -0.5f }, {  0.5f, -0.5f, -0.5f }, {  0.5f, -0.5f,  0.5f }, { -0.5f, -0.5f,  0.5f } }
        };

        const glm::vec2 uvs[4] =
        {
            { 0.0f, 1.0f },
            { 1.0f, 1.0f },
            { 1.0f, 0.0f },
            { 0.0f, 0.0f }
        };

        for (uint32_t face = 0; face < 6; face++)
        {
            const uint32_t base = static_cast<uint32_t>(vertices.size());

            for (uint32_t corner = 0; corner < 4; corner++)
            {
                Vertex vertex;
                vertex.position = faceVertices[face][corner];
                vertex.normal = normals[face];
                vertex.uv = uvs[corner];
                vertices.push_back(vertex);
            }

            indices.push_back(base + 0);
            indices.push_back(base + 1);
            indices.push_back(base + 2);
            indices.push_back(base + 2);
            indices.push_back(base + 3);
            indices.push_back(base + 0);
        }

        return Mesh(context, vertices, indices);
    }

    Mesh Mesh::CreateQuad(VulkanContext& context)
    {
        std::vector<Vertex> vertices(4);
        vertices[0].position = { -0.5f, -0.5f, 0.0f };
        vertices[1].position = { 0.5f, -0.5f, 0.0f };
        vertices[2].position = { 0.5f,  0.5f, 0.0f };
        vertices[3].position = { -0.5f,  0.5f, 0.0f };

        vertices[0].uv = { 0.0f, 1.0f };
        vertices[1].uv = { 1.0f, 1.0f };
        vertices[2].uv = { 1.0f, 0.0f };
        vertices[3].uv = { 0.0f, 0.0f };

        for (Vertex& vertex : vertices)
        {
            vertex.normal = { 0.0f, 0.0f, 1.0f };
        }

        const std::vector<uint32_t> indices = { 0, 1, 2, 2, 3, 0 };

        return Mesh(context, vertices, indices);
    }

    Mesh Mesh::CreateCustom(VulkanContext& context, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices)
    {
        return Mesh(context, vertices, indices);
    }

} // namespace Core