#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "Buffer.h"

namespace Core
{

    class VulkanContext;
    class GraphicsPipeline;
    class DescriptorManager;
    class Texture;

    class Material
    {
    public:
        Material(VulkanContext& context, DescriptorManager& descriptors, const GraphicsPipeline& pipeline, uint32_t set);
        Material(VulkanContext& context, DescriptorManager& descriptors, const GraphicsPipeline& pipeline);

        Material(const Material&) = delete;
        Material& operator=(const Material&) = delete;
        Material(Material&& other) noexcept = default;
        Material& operator=(Material&& other) noexcept = default;

        void SetFloat(const std::string& name, float value);
        void SetInt(const std::string& name, int32_t value);
        void SetVec2(const std::string& name, const glm::vec2& value);
        void SetVec3(const std::string& name, const glm::vec3& value);
        void SetVec4(const std::string& name, const glm::vec4& value);
        void SetMat4(const std::string& name, const glm::mat4& value);
        void SetTexture(const std::string& name, const Texture& texture);

        // Binds a storage buffer by its name in the shader. The material keeps only a pointer, so
        // the buffer must outlive it — and because the pointed-at buffer can be swapped every
        // frame (a ping-pong pair, for instance), rebinding the same name with a different buffer
        // is cheap and expected.
        void SetStorageBuffer(const std::string& name, const Buffer& buffer);
        bool HasStorageBuffer(const std::string& name) const;

        bool HasProperty(const std::string& name) const;
        bool HasTexture(const std::string& name) const;

        void Flush();

        VkDescriptorSet GetDescriptorSet() const { return m_DescriptorSet; }
        uint32_t GetSet() const { return m_Set; }
        const GraphicsPipeline& GetPipeline() const { return *m_Pipeline; }
        bool IsDirty() const { return m_UniformDirty || m_DescriptorDirty; }

    private:
        void WriteRaw(const std::string& name, const void* data, uint32_t size);

    private:
        VulkanContext* m_Context{ nullptr };
        DescriptorManager* m_Descriptors{ nullptr };
        const GraphicsPipeline* m_Pipeline{ nullptr };
        uint32_t m_Set{ 2 };

        std::vector<uint8_t> m_UniformData;
        uint32_t m_UniformBinding{ 0 };
        bool m_HasUniformBlock{ false };
        std::vector<Buffer> m_UniformBuffer;

        std::unordered_map<uint32_t, const Texture*> m_Textures;
        std::unordered_map<uint32_t, const Buffer*> m_StorageBuffers;

        VkDescriptorSet m_DescriptorSet{ VK_NULL_HANDLE };
        bool m_UniformDirty{ false };
        bool m_DescriptorDirty{ true };
    };

} // namespace Core