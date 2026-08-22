#include "Material.h"

#include "DescriptorManager.h"
#include "Pipeline.h"
#include "Texture.h"
#include "VulkanContext.h"

#include <cstring>
#include <stdexcept>

namespace Core
{

    Material::Material(VulkanContext& context, DescriptorManager& descriptors, const GraphicsPipeline& pipeline, uint32_t set)
        : m_Context(&context)
        , m_Descriptors(&descriptors)
        , m_Pipeline(&pipeline)
        , m_Set(set)
    {
        if (pipeline.GetDescriptorSetLayout(set) == VK_NULL_HANDLE)
        {
            throw std::runtime_error("Pipeline has no descriptor set layout for the requested material set");
        }

        const ShaderBinding* uniformBlock = pipeline.FindUniformBlock(set);

        if (uniformBlock != nullptr && uniformBlock->blockSize > 0)
        {
            m_HasUniformBlock = true;
            m_UniformBinding = uniformBlock->binding;
            m_UniformData.assign(uniformBlock->blockSize, 0);

            m_UniformBuffer.reserve(1);
            m_UniformBuffer.emplace_back(context, uniformBlock->blockSize, BufferType::Uniform, MemoryUsage::HostVisible);

            m_UniformDirty = true;
        }
    }

    Material::Material(VulkanContext& context, DescriptorManager& descriptors, const GraphicsPipeline& pipeline)
        : Material(context, descriptors, pipeline, 2)
    {
    }

    void Material::WriteRaw(const std::string& name, const void* data, uint32_t size)
    {
        if (!m_HasUniformBlock)
        {
            throw std::runtime_error("Material set declares no uniform block");
        }

        uint32_t binding = 0;
        uint32_t offset = 0;
        uint32_t memberSize = 0;

        if (!m_Pipeline->FindMember(m_Set, name, binding, offset, memberSize))
        {
            throw std::runtime_error("Material property not found in shader reflection: " + name);
        }

        uint32_t writeSize = size;

        if (writeSize > memberSize)
        {
            writeSize = memberSize;
        }

        if (offset + writeSize > m_UniformData.size())
        {
            throw std::runtime_error("Material property write exceeds uniform block size: " + name);
        }

        std::memcpy(m_UniformData.data() + offset, data, writeSize);
        m_UniformDirty = true;
    }

    void Material::SetFloat(const std::string& name, float value)
    {
        WriteRaw(name, &value, sizeof(float));
    }

    void Material::SetInt(const std::string& name, int32_t value)
    {
        WriteRaw(name, &value, sizeof(int32_t));
    }

    void Material::SetVec2(const std::string& name, const glm::vec2& value)
    {
        WriteRaw(name, &value, sizeof(glm::vec2));
    }

    void Material::SetVec3(const std::string& name, const glm::vec3& value)
    {
        WriteRaw(name, &value, sizeof(glm::vec3));
    }

    void Material::SetVec4(const std::string& name, const glm::vec4& value)
    {
        WriteRaw(name, &value, sizeof(glm::vec4));
    }

    void Material::SetMat4(const std::string& name, const glm::mat4& value)
    {
        WriteRaw(name, &value, sizeof(glm::mat4));
    }

    void Material::SetTexture(const std::string& name, const Texture& texture)
    {
        const ShaderBinding* binding = m_Pipeline->FindBindingByName(m_Set, name);

        if (binding == nullptr)
        {
            throw std::runtime_error("Material texture not found in shader reflection: " + name);
        }

        m_Textures[binding->binding] = &texture;
        m_DescriptorDirty = true;
    }

    void Material::SetStorageBuffer(const std::string& name, const Buffer& buffer)
    {
        const ShaderBinding* binding = m_Pipeline->FindBindingByName(m_Set, name);

        if (binding == nullptr)
        {
            throw std::runtime_error("Material storage buffer not found in shader reflection: " + name);
        }

        auto existing = m_StorageBuffers.find(binding->binding);

        if (existing != m_StorageBuffers.end() && existing->second == &buffer)
        {
            // Same buffer as last frame: rewriting the descriptor set would be pure overhead.
            return;
        }

        m_StorageBuffers[binding->binding] = &buffer;
        m_DescriptorDirty = true;
    }

    bool Material::HasStorageBuffer(const std::string& name) const
    {
        const ShaderBinding* binding = m_Pipeline->FindBindingByName(m_Set, name);
        return binding != nullptr && binding->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }

    bool Material::HasProperty(const std::string& name) const
    {
        uint32_t binding = 0;
        uint32_t offset = 0;
        uint32_t size = 0;
        return m_Pipeline->FindMember(m_Set, name, binding, offset, size);
    }

    bool Material::HasTexture(const std::string& name) const
    {
        return m_Pipeline->FindBindingByName(m_Set, name) != nullptr;
    }

    void Material::Flush()
    {
        if (m_UniformDirty && m_HasUniformBlock)
        {
            m_UniformBuffer[0].SetData(m_UniformData.data(), static_cast<VkDeviceSize>(m_UniformData.size()));
            m_UniformDirty = false;
        }

        if (!m_DescriptorDirty)
        {
            return;
        }

        for (const ShaderBinding& declared : m_Pipeline->GetBindings())
        {
            if (declared.set != m_Set)
            {
                continue;
            }

            if (declared.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            {
                if (m_StorageBuffers.find(declared.binding) == m_StorageBuffers.end())
                {
                    throw std::runtime_error("Material is missing a storage buffer required by the shader: " + declared.name);
                }

                continue;
            }

            if (declared.descriptorType != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            {
                continue;
            }

            if (m_Textures.find(declared.binding) == m_Textures.end())
            {
                throw std::runtime_error("Material is missing a texture required by the shader: " + declared.name);
            }
        }

        DescriptorWriter writer = m_Descriptors->Begin(*m_Pipeline, m_Set);

        if (m_HasUniformBlock)
        {
            writer.WriteBuffer(m_UniformBinding, m_UniformBuffer[0]);
        }

        for (const auto& entry : m_Textures)
        {
            writer.WriteImage(entry.first, entry.second->GetDescriptorInfo());
        }

        for (const auto& entry : m_StorageBuffers)
        {
            writer.WriteBuffer(entry.first, *entry.second);
        }

        m_DescriptorSet = writer.Build();
        m_DescriptorDirty = false;
    }

} // namespace Core