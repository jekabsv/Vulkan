#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"

namespace Core
{

    class VulkanContext;

    struct SamplerConfig
    {
        VkFilter magFilter{ VK_FILTER_LINEAR };
        VkFilter minFilter{ VK_FILTER_LINEAR };
        VkSamplerMipmapMode mipmapMode{ VK_SAMPLER_MIPMAP_MODE_LINEAR };
        VkSamplerAddressMode addressModeU{ VK_SAMPLER_ADDRESS_MODE_REPEAT };
        VkSamplerAddressMode addressModeV{ VK_SAMPLER_ADDRESS_MODE_REPEAT };
        VkSamplerAddressMode addressModeW{ VK_SAMPLER_ADDRESS_MODE_REPEAT };
        VkBorderColor borderColor{ VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK };
        bool anisotropyEnable{ true };
        float maxAnisotropy{ 16.0f };
        bool compareEnable{ false };
        VkCompareOp compareOp{ VK_COMPARE_OP_ALWAYS };
    };

    struct TextureConfig
    {
        uint32_t width{ 1 };
        uint32_t height{ 1 };
        uint32_t depth{ 1 };
        VkFormat format{ VK_FORMAT_R8G8B8A8_SRGB };
        VkImageUsageFlags usage{ VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT };
        VkImageAspectFlags aspectMask{ VK_IMAGE_ASPECT_COLOR_BIT };
        VkImageTiling tiling{ VK_IMAGE_TILING_OPTIMAL };
        VkSampleCountFlagBits sampleCount{ VK_SAMPLE_COUNT_1_BIT };
        uint32_t mipLevels{ 1 };
        bool generateMipmaps{ false };
        bool createSampler{ true };
        SamplerConfig sampler;
    };

    class Texture
    {
    public:
        Texture(VulkanContext& context, const TextureConfig& config);
        Texture(VulkanContext& context, const TextureConfig& config, const void* pixels, VkDeviceSize dataSize);
        ~Texture();

        Texture(const Texture&) = delete;
        Texture& operator=(const Texture&) = delete;
        Texture(Texture&& other) noexcept;
        Texture& operator=(Texture&& other) noexcept;

        void SetData(const void* pixels, VkDeviceSize dataSize);
        void GenerateMipmaps();

        void TransitionLayout(VkImageLayout newLayout);
        void TransitionLayout(VkCommandBuffer commandBuffer, VkImageLayout newLayout);

        VkDescriptorImageInfo GetDescriptorInfo() const;

        VkImage GetImage() const { return m_Image; }
        VkImageView GetImageView() const { return m_ImageView; }
        VkSampler GetSampler() const { return m_Sampler; }
        VkFormat GetFormat() const { return m_Config.format; }
        VkImageLayout GetLayout() const { return m_CurrentLayout; }
        VkExtent3D GetExtent() const { return m_Extent; }
        uint32_t GetWidth() const { return m_Extent.width; }
        uint32_t GetHeight() const { return m_Extent.height; }
        uint32_t GetDepth() const { return m_Extent.depth; }
        uint32_t GetMipLevels() const { return m_MipLevels; }

        static uint32_t CalculateMipLevels(uint32_t width, uint32_t height, uint32_t depth);

    private:
        void CreateImage();
        void CreateImageView();
        void CreateSampler();
        void Destroy();

    private:
        VulkanContext* m_Context{ nullptr };
        TextureConfig m_Config;

        VkImage m_Image{ VK_NULL_HANDLE };
        VmaAllocation m_Allocation{ VK_NULL_HANDLE };
        VkImageView m_ImageView{ VK_NULL_HANDLE };
        VkSampler m_Sampler{ VK_NULL_HANDLE };

        VkExtent3D m_Extent{ 1, 1, 1 };
        VkImageType m_ImageType{ VK_IMAGE_TYPE_2D };
        VkImageViewType m_ViewType{ VK_IMAGE_VIEW_TYPE_2D };
        uint32_t m_MipLevels{ 1 };
        VkImageLayout m_CurrentLayout{ VK_IMAGE_LAYOUT_UNDEFINED };
    };

} // namespace Core