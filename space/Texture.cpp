#include "Texture.h"

#include "Buffer.h"
#include "VulkanContext.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace Core
{

    namespace
    {
        void CheckResult(VkResult result, const char* message)
        {
            if (result != VK_SUCCESS)
            {
                throw std::runtime_error(message);
            }
        }

        void BarrierImage(
            VkCommandBuffer commandBuffer,
            VkImage image,
            VkImageLayout oldLayout,
            VkImageLayout newLayout,
            VkImageAspectFlags aspectMask,
            uint32_t baseMipLevel,
            uint32_t levelCount)
        {
            VkImageMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.oldLayout = oldLayout;
            barrier.newLayout = newLayout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange.aspectMask = aspectMask;
            barrier.subresourceRange.baseMipLevel = baseMipLevel;
            barrier.subresourceRange.levelCount = levelCount;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;

            barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;

            if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED)
            {
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                barrier.srcAccessMask = VK_ACCESS_2_NONE;
            }

            if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
            {
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            }

            if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
            {
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            }

            if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
            {
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            }

            if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
            {
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            }

            if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            {
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            }

            VkDependencyInfo dependency{};
            dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dependency.imageMemoryBarrierCount = 1;
            dependency.pImageMemoryBarriers = &barrier;

            vkCmdPipelineBarrier2(commandBuffer, &dependency);
        }
    }

    uint32_t Texture::CalculateMipLevels(uint32_t width, uint32_t height, uint32_t depth)
    {
        uint32_t largest = width;

        if (height > largest)
        {
            largest = height;
        }

        if (depth > largest)
        {
            largest = depth;
        }

        uint32_t levels = 1;

        while (largest > 1)
        {
            largest = largest / 2;
            levels++;
        }

        return levels;
    }

    Texture::Texture(VulkanContext& context, const TextureConfig& config)
        : m_Context(&context)
        , m_Config(config)
    {
        if (config.width == 0 || config.height == 0 || config.depth == 0)
        {
            throw std::runtime_error("Texture dimensions must be non zero");
        }

        m_Extent.width = config.width;
        m_Extent.height = config.height;
        m_Extent.depth = config.depth;

        if (config.depth > 1)
        {
            m_ImageType = VK_IMAGE_TYPE_3D;
            m_ViewType = VK_IMAGE_VIEW_TYPE_3D;
        }
        else
        {
            m_ImageType = VK_IMAGE_TYPE_2D;
            m_ViewType = VK_IMAGE_VIEW_TYPE_2D;
        }

        m_MipLevels = config.mipLevels;

        if (config.generateMipmaps)
        {
            m_MipLevels = CalculateMipLevels(config.width, config.height, config.depth);
        }

        if (m_MipLevels == 0)
        {
            m_MipLevels = 1;
        }

        CreateImage();
        CreateImageView();

        if (config.createSampler)
        {
            CreateSampler();
        }
    }

    Texture::Texture(VulkanContext& context, const TextureConfig& config, const void* pixels, VkDeviceSize dataSize)
        : Texture(context, config)
    {
        SetData(pixels, dataSize);
    }

    Texture::~Texture()
    {
        Destroy();
    }

    Texture::Texture(Texture&& other) noexcept
        : m_Context(other.m_Context)
        , m_Config(other.m_Config)
        , m_Image(other.m_Image)
        , m_Allocation(other.m_Allocation)
        , m_ImageView(other.m_ImageView)
        , m_Sampler(other.m_Sampler)
        , m_Extent(other.m_Extent)
        , m_ImageType(other.m_ImageType)
        , m_ViewType(other.m_ViewType)
        , m_MipLevels(other.m_MipLevels)
        , m_CurrentLayout(other.m_CurrentLayout)
    {
        other.m_Context = nullptr;
        other.m_Image = VK_NULL_HANDLE;
        other.m_Allocation = VK_NULL_HANDLE;
        other.m_ImageView = VK_NULL_HANDLE;
        other.m_Sampler = VK_NULL_HANDLE;
    }

    Texture& Texture::operator=(Texture&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        Destroy();

        m_Context = other.m_Context;
        m_Config = other.m_Config;
        m_Image = other.m_Image;
        m_Allocation = other.m_Allocation;
        m_ImageView = other.m_ImageView;
        m_Sampler = other.m_Sampler;
        m_Extent = other.m_Extent;
        m_ImageType = other.m_ImageType;
        m_ViewType = other.m_ViewType;
        m_MipLevels = other.m_MipLevels;
        m_CurrentLayout = other.m_CurrentLayout;

        other.m_Context = nullptr;
        other.m_Image = VK_NULL_HANDLE;
        other.m_Allocation = VK_NULL_HANDLE;
        other.m_ImageView = VK_NULL_HANDLE;
        other.m_Sampler = VK_NULL_HANDLE;

        return *this;
    }

    void Texture::Destroy()
    {
        if (m_Context == nullptr)
        {
            return;
        }

        if (m_Sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(m_Context->GetDevice(), m_Sampler, nullptr);
            m_Sampler = VK_NULL_HANDLE;
        }

        if (m_ImageView != VK_NULL_HANDLE)
        {
            vkDestroyImageView(m_Context->GetDevice(), m_ImageView, nullptr);
            m_ImageView = VK_NULL_HANDLE;
        }

        if (m_Image != VK_NULL_HANDLE)
        {
            vmaDestroyImage(m_Context->GetAllocator(), m_Image, m_Allocation);
            m_Image = VK_NULL_HANDLE;
            m_Allocation = VK_NULL_HANDLE;
        }
    }

    void Texture::CreateImage()
    {
        VkImageUsageFlags usage = m_Config.usage;

        if (m_MipLevels > 1)
        {
            usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        }

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = m_ImageType;
        imageInfo.format = m_Config.format;
        imageInfo.extent = m_Extent;
        imageInfo.mipLevels = m_MipLevels;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = m_Config.sampleCount;
        imageInfo.tiling = m_Config.tiling;
        imageInfo.usage = usage;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocationInfo{};
        allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;

        CheckResult(
            vmaCreateImage(m_Context->GetAllocator(), &imageInfo, &allocationInfo, &m_Image, &m_Allocation, nullptr),
            "Failed to create texture image");

        m_CurrentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    void Texture::CreateImageView()
    {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = m_Image;
        createInfo.viewType = m_ViewType;
        createInfo.format = m_Config.format;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = m_Config.aspectMask;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = m_MipLevels;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        CheckResult(vkCreateImageView(m_Context->GetDevice(), &createInfo, nullptr, &m_ImageView), "Failed to create texture image view");
    }

    void Texture::CreateSampler()
    {
        const SamplerConfig& sampler = m_Config.sampler;

        float maxAnisotropy = sampler.maxAnisotropy;
        const float deviceLimit = m_Context->GetDeviceProperties().limits.maxSamplerAnisotropy;

        if (maxAnisotropy > deviceLimit)
        {
            maxAnisotropy = deviceLimit;
        }

        VkSamplerCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        createInfo.magFilter = sampler.magFilter;
        createInfo.minFilter = sampler.minFilter;
        createInfo.mipmapMode = sampler.mipmapMode;
        createInfo.addressModeU = sampler.addressModeU;
        createInfo.addressModeV = sampler.addressModeV;
        createInfo.addressModeW = sampler.addressModeW;
        createInfo.mipLodBias = 0.0f;
        createInfo.minLod = 0.0f;
        createInfo.maxLod = static_cast<float>(m_MipLevels);
        createInfo.borderColor = sampler.borderColor;
        createInfo.unnormalizedCoordinates = VK_FALSE;
        createInfo.anisotropyEnable = VK_FALSE;
        createInfo.maxAnisotropy = 1.0f;
        createInfo.compareEnable = VK_FALSE;
        createInfo.compareOp = sampler.compareOp;

        if (sampler.anisotropyEnable && m_Context->GetEnabledFeatures().samplerAnisotropy == VK_TRUE)
        {
            createInfo.anisotropyEnable = VK_TRUE;
            createInfo.maxAnisotropy = maxAnisotropy;
        }

        if (sampler.compareEnable)
        {
            createInfo.compareEnable = VK_TRUE;
        }

        CheckResult(vkCreateSampler(m_Context->GetDevice(), &createInfo, nullptr, &m_Sampler), "Failed to create texture sampler");
    }

    void Texture::SetData(const void* pixels, VkDeviceSize dataSize)
    {
        if (pixels == nullptr)
        {
            throw std::runtime_error("Cannot upload texture data from a null pointer");
        }

        Buffer staging(*m_Context, dataSize, BufferType::Staging, MemoryUsage::HostVisible);
        staging.SetData(pixels, dataSize);

        VkCommandBuffer commandBuffer = m_Context->BeginSingleTimeCommands();

        BarrierImage(
            commandBuffer,
            m_Image,
            m_CurrentLayout,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            m_Config.aspectMask,
            0,
            m_MipLevels);

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = m_Config.aspectMask;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = m_Extent;

        vkCmdCopyBufferToImage(commandBuffer, staging.GetHandle(), m_Image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        m_CurrentLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        if (m_MipLevels == 1)
        {
            BarrierImage(
                commandBuffer,
                m_Image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                m_Config.aspectMask,
                0,
                m_MipLevels);

            m_CurrentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        m_Context->EndSingleTimeCommands(commandBuffer);

        if (m_MipLevels > 1)
        {
            GenerateMipmaps();
        }
    }

    void Texture::GenerateMipmaps()
    {
        if (m_MipLevels <= 1)
        {
            return;
        }

        VkFormatProperties formatProperties{};
        vkGetPhysicalDeviceFormatProperties(m_Context->GetPhysicalDevice(), m_Config.format, &formatProperties);

        if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
        {
            throw std::runtime_error("Texture format does not support linear blitting for mipmap generation");
        }

        VkCommandBuffer commandBuffer = m_Context->BeginSingleTimeCommands();

        int32_t mipWidth = static_cast<int32_t>(m_Extent.width);
        int32_t mipHeight = static_cast<int32_t>(m_Extent.height);
        int32_t mipDepth = static_cast<int32_t>(m_Extent.depth);

        for (uint32_t level = 1; level < m_MipLevels; level++)
        {
            BarrierImage(
                commandBuffer,
                m_Image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                m_Config.aspectMask,
                level - 1,
                1);

            int32_t nextWidth = std::max(mipWidth / 2, 1);
            int32_t nextHeight = std::max(mipHeight / 2, 1);
            int32_t nextDepth = std::max(mipDepth / 2, 1);

            VkImageBlit blit{};
            blit.srcOffsets[0] = { 0, 0, 0 };
            blit.srcOffsets[1] = { mipWidth, mipHeight, mipDepth };
            blit.srcSubresource.aspectMask = m_Config.aspectMask;
            blit.srcSubresource.mipLevel = level - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            blit.dstOffsets[0] = { 0, 0, 0 };
            blit.dstOffsets[1] = { nextWidth, nextHeight, nextDepth };
            blit.dstSubresource.aspectMask = m_Config.aspectMask;
            blit.dstSubresource.mipLevel = level;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;

            vkCmdBlitImage(
                commandBuffer,
                m_Image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                m_Image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &blit,
                VK_FILTER_LINEAR);

            BarrierImage(
                commandBuffer,
                m_Image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                m_Config.aspectMask,
                level - 1,
                1);

            mipWidth = nextWidth;
            mipHeight = nextHeight;
            mipDepth = nextDepth;
        }

        BarrierImage(
            commandBuffer,
            m_Image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            m_Config.aspectMask,
            m_MipLevels - 1,
            1);

        m_Context->EndSingleTimeCommands(commandBuffer);

        m_CurrentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    void Texture::TransitionLayout(VkImageLayout newLayout)
    {
        VkCommandBuffer commandBuffer = m_Context->BeginSingleTimeCommands();
        TransitionLayout(commandBuffer, newLayout);
        m_Context->EndSingleTimeCommands(commandBuffer);
    }

    void Texture::TransitionLayout(VkCommandBuffer commandBuffer, VkImageLayout newLayout)
    {
        if (m_CurrentLayout == newLayout)
        {
            return;
        }

        BarrierImage(commandBuffer, m_Image, m_CurrentLayout, newLayout, m_Config.aspectMask, 0, m_MipLevels);

        m_CurrentLayout = newLayout;
    }

    VkDescriptorImageInfo Texture::GetDescriptorInfo() const
    {
        VkDescriptorImageInfo info{};
        info.sampler = m_Sampler;
        info.imageView = m_ImageView;
        info.imageLayout = m_CurrentLayout;
        return info;
    }

} // namespace Core