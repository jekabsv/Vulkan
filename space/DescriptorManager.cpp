#include "DescriptorManager.h"

#include "Buffer.h"
#include "Pipeline.h"
#include "Texture.h"
#include "VulkanContext.h"

#include <stdexcept>
#include <utility>

namespace Core
{

    namespace
    {
        struct PoolRatio
        {
            VkDescriptorType type;
            float ratio;
        };

        const PoolRatio PoolRatios[] =
        {
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2.0f },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1.0f },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2.0f },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1.0f },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4.0f },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2.0f },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1.0f },
            { VK_DESCRIPTOR_TYPE_SAMPLER, 1.0f },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1.0f }
        };
    }

    DescriptorManager::DescriptorManager(VulkanContext& context, uint32_t maxSetsPerPool)
        : m_Context(&context)
        , m_MaxSetsPerPool(maxSetsPerPool)
    {
        if (m_MaxSetsPerPool == 0)
        {
            m_MaxSetsPerPool = 1;
        }
    }

    DescriptorManager::DescriptorManager(VulkanContext& context)
        : m_Context(&context)
    {
    }

    DescriptorManager::~DescriptorManager()
    {
        Destroy();
    }

    DescriptorManager::DescriptorManager(DescriptorManager&& other) noexcept
        : m_Context(other.m_Context)
        , m_MaxSetsPerPool(other.m_MaxSetsPerPool)
        , m_CurrentPool(other.m_CurrentPool)
        , m_UsedPools(std::move(other.m_UsedPools))
        , m_FreePools(std::move(other.m_FreePools))
    {
        other.m_Context = nullptr;
        other.m_CurrentPool = VK_NULL_HANDLE;
    }

    DescriptorManager& DescriptorManager::operator=(DescriptorManager&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        Destroy();

        m_Context = other.m_Context;
        m_MaxSetsPerPool = other.m_MaxSetsPerPool;
        m_CurrentPool = other.m_CurrentPool;
        m_UsedPools = std::move(other.m_UsedPools);
        m_FreePools = std::move(other.m_FreePools);

        other.m_Context = nullptr;
        other.m_CurrentPool = VK_NULL_HANDLE;

        return *this;
    }

    void DescriptorManager::Destroy()
    {
        if (m_Context == nullptr)
        {
            return;
        }

        for (VkDescriptorPool pool : m_FreePools)
        {
            vkDestroyDescriptorPool(m_Context->GetDevice(), pool, nullptr);
        }

        for (VkDescriptorPool pool : m_UsedPools)
        {
            vkDestroyDescriptorPool(m_Context->GetDevice(), pool, nullptr);
        }

        if (m_CurrentPool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(m_Context->GetDevice(), m_CurrentPool, nullptr);
            m_CurrentPool = VK_NULL_HANDLE;
        }

        m_FreePools.clear();
        m_UsedPools.clear();
    }

    VkDescriptorPool DescriptorManager::CreatePool()
    {
        std::vector<VkDescriptorPoolSize> sizes;

        for (const PoolRatio& ratio : PoolRatios)
        {
            VkDescriptorPoolSize size{};
            size.type = ratio.type;
            size.descriptorCount = static_cast<uint32_t>(ratio.ratio * static_cast<float>(m_MaxSetsPerPool));

            if (size.descriptorCount == 0)
            {
                size.descriptorCount = 1;
            }

            sizes.push_back(size);
        }

        VkDescriptorPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        createInfo.maxSets = m_MaxSetsPerPool;
        createInfo.poolSizeCount = static_cast<uint32_t>(sizes.size());
        createInfo.pPoolSizes = sizes.data();

        VkDescriptorPool pool = VK_NULL_HANDLE;

        if (vkCreateDescriptorPool(m_Context->GetDevice(), &createInfo, nullptr, &pool) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create descriptor pool");
        }

        return pool;
    }

    VkDescriptorSet DescriptorManager::Allocate(VkDescriptorSetLayout layout)
    {
        if (m_CurrentPool == VK_NULL_HANDLE)
        {
            if (!m_FreePools.empty())
            {
                m_CurrentPool = m_FreePools.back();
                m_FreePools.pop_back();
            }
            else
            {
                m_CurrentPool = CreatePool();
            }
        }

        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = m_CurrentPool;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &layout;

        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        VkResult result = vkAllocateDescriptorSets(m_Context->GetDevice(), &allocateInfo, &descriptorSet);

        if (result == VK_SUCCESS)
        {
            return descriptorSet;
        }

        if (result != VK_ERROR_OUT_OF_POOL_MEMORY && result != VK_ERROR_FRAGMENTED_POOL)
        {
            throw std::runtime_error("Failed to allocate descriptor set");
        }

        m_UsedPools.push_back(m_CurrentPool);

        if (!m_FreePools.empty())
        {
            m_CurrentPool = m_FreePools.back();
            m_FreePools.pop_back();
        }
        else
        {
            m_CurrentPool = CreatePool();
        }

        allocateInfo.descriptorPool = m_CurrentPool;
        result = vkAllocateDescriptorSets(m_Context->GetDevice(), &allocateInfo, &descriptorSet);

        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate descriptor set from a fresh pool");
        }

        return descriptorSet;
    }

    void DescriptorManager::ResetPools()
    {
        for (VkDescriptorPool pool : m_UsedPools)
        {
            vkResetDescriptorPool(m_Context->GetDevice(), pool, 0);
            m_FreePools.push_back(pool);
        }

        m_UsedPools.clear();

        if (m_CurrentPool != VK_NULL_HANDLE)
        {
            vkResetDescriptorPool(m_Context->GetDevice(), m_CurrentPool, 0);
        }
    }

    DescriptorWriter DescriptorManager::Begin(const Pipeline& pipeline, uint32_t set)
    {
        return DescriptorWriter(*this, pipeline, set);
    }

    DescriptorWriter::DescriptorWriter(DescriptorManager& manager, const Pipeline& pipeline, uint32_t set)
        : m_Manager(&manager)
        , m_Pipeline(&pipeline)
        , m_Set(set)
    {
    }

    VkDescriptorType DescriptorWriter::ResolveType(uint32_t binding) const
    {
        VkDescriptorType type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

        if (!m_Pipeline->FindDescriptorType(m_Set, binding, type))
        {
            throw std::runtime_error("Shader reflection has no descriptor at the requested set and binding");
        }

        return type;
    }

    DescriptorWriter& DescriptorWriter::WriteBuffer(uint32_t binding, const Buffer& buffer)
    {
        return WriteBuffer(binding, buffer, buffer.GetSize(), 0);
    }

    DescriptorWriter& DescriptorWriter::WriteBuffer(uint32_t binding, const Buffer& buffer, VkDeviceSize size, VkDeviceSize offset)
    {
        PendingWrite write;
        write.binding = binding;
        write.descriptorType = ResolveType(binding);
        write.isBuffer = true;
        write.infoIndex = m_BufferInfos.size();

        m_BufferInfos.push_back(buffer.GetDescriptorInfo(size, offset));
        m_Writes.push_back(write);

        return *this;
    }

    DescriptorWriter& DescriptorWriter::WriteTexture(uint32_t binding, const Texture& texture)
    {
        return WriteImage(binding, texture.GetDescriptorInfo());
    }

    DescriptorWriter& DescriptorWriter::WriteImage(uint32_t binding, const VkDescriptorImageInfo& imageInfo)
    {
        PendingWrite write;
        write.binding = binding;
        write.descriptorType = ResolveType(binding);
        write.isBuffer = false;
        write.infoIndex = m_ImageInfos.size();

        m_ImageInfos.push_back(imageInfo);
        m_Writes.push_back(write);

        return *this;
    }

    void DescriptorWriter::Overwrite(VkDescriptorSet descriptorSet)
    {
        std::vector<VkWriteDescriptorSet> writes;

        for (const PendingWrite& pending : m_Writes)
        {
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptorSet;
            write.dstBinding = pending.binding;
            write.dstArrayElement = 0;
            write.descriptorCount = 1;
            write.descriptorType = pending.descriptorType;

            if (pending.isBuffer)
            {
                write.pBufferInfo = &m_BufferInfos[pending.infoIndex];
            }
            else
            {
                write.pImageInfo = &m_ImageInfos[pending.infoIndex];
            }

            writes.push_back(write);
        }

        if (writes.empty())
        {
            return;
        }

        vkUpdateDescriptorSets(
            m_Manager->GetContext().GetDevice(),
            static_cast<uint32_t>(writes.size()),
            writes.data(),
            0,
            nullptr);
    }

    VkDescriptorSet DescriptorWriter::Build()
    {
        VkDescriptorSetLayout layout = m_Pipeline->GetDescriptorSetLayout(m_Set);

        if (layout == VK_NULL_HANDLE)
        {
            throw std::runtime_error("Pipeline has no descriptor set layout at the requested set index");
        }

        VkDescriptorSet descriptorSet = m_Manager->Allocate(layout);
        Overwrite(descriptorSet);

        return descriptorSet;
    }

} // namespace Core