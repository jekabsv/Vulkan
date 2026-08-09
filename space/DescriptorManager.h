#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

namespace Core
{

    class VulkanContext;
    class Pipeline;
    class Buffer;
    class Texture;
    class DescriptorManager;

    class DescriptorWriter
    {
    public:
        DescriptorWriter(DescriptorManager& manager, const Pipeline& pipeline, uint32_t set);

        DescriptorWriter& WriteBuffer(uint32_t binding, const Buffer& buffer);
        DescriptorWriter& WriteBuffer(uint32_t binding, const Buffer& buffer, VkDeviceSize size, VkDeviceSize offset);
        DescriptorWriter& WriteTexture(uint32_t binding, const Texture& texture);
        DescriptorWriter& WriteImage(uint32_t binding, const VkDescriptorImageInfo& imageInfo);

        VkDescriptorSet Build();
        void Overwrite(VkDescriptorSet descriptorSet);

    private:
        struct PendingWrite
        {
            uint32_t binding{ 0 };
            VkDescriptorType descriptorType{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
            bool isBuffer{ true };
            size_t infoIndex{ 0 };
        };

        VkDescriptorType ResolveType(uint32_t binding) const;

    private:
        DescriptorManager* m_Manager{ nullptr };
        const Pipeline* m_Pipeline{ nullptr };
        uint32_t m_Set{ 0 };

        std::vector<PendingWrite> m_Writes;
        std::vector<VkDescriptorBufferInfo> m_BufferInfos;
        std::vector<VkDescriptorImageInfo> m_ImageInfos;
    };

    class DescriptorManager
    {
    public:
        DescriptorManager(VulkanContext& context, uint32_t maxSetsPerPool);
        explicit DescriptorManager(VulkanContext& context);
        ~DescriptorManager();

        DescriptorManager(const DescriptorManager&) = delete;
        DescriptorManager& operator=(const DescriptorManager&) = delete;
        DescriptorManager(DescriptorManager&& other) noexcept;
        DescriptorManager& operator=(DescriptorManager&& other) noexcept;

        DescriptorWriter Begin(const Pipeline& pipeline, uint32_t set);

        VkDescriptorSet Allocate(VkDescriptorSetLayout layout);
        void ResetPools();

        VulkanContext& GetContext() const { return *m_Context; }

    private:
        VkDescriptorPool CreatePool();
        void Destroy();

    private:
        VulkanContext* m_Context{ nullptr };
        uint32_t m_MaxSetsPerPool{ 1000 };

        VkDescriptorPool m_CurrentPool{ VK_NULL_HANDLE };
        std::vector<VkDescriptorPool> m_UsedPools;
        std::vector<VkDescriptorPool> m_FreePools;
    };

} // namespace Core