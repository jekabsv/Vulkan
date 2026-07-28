#include "Swapchain.h"

#include "VulkanContext.h"
#include "Window.h"

#include <algorithm>
#include <limits>
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

        bool ContainsPresentMode(const std::vector<VkPresentModeKHR>& available, VkPresentModeKHR mode)
        {
            for (VkPresentModeKHR candidate : available)
            {
                if (candidate == mode)
                {
                    return true;
                }
            }

            return false;
        }
    }

    Swapchain::Swapchain(VulkanContext& context, Window& window, const SwapchainConfig& config)
        : m_Context(&context)
        , m_Window(&window)
        , m_Config(config)
    {
        Initialize();
    }

    Swapchain::Swapchain(VulkanContext& context, Window& window)
        : m_Context(&context)
        , m_Window(&window)
    {
        Initialize();
    }

    Swapchain::~Swapchain()
    {
        Destroy();
    }

    Swapchain::Swapchain(Swapchain&& other) noexcept
        : m_Context(other.m_Context)
        , m_Window(other.m_Window)
        , m_Config(other.m_Config)
        , m_Swapchain(other.m_Swapchain)
        , m_ImageFormat(other.m_ImageFormat)
        , m_ColorSpace(other.m_ColorSpace)
        , m_PresentMode(other.m_PresentMode)
        , m_Extent(other.m_Extent)
        , m_Images(std::move(other.m_Images))
        , m_ImageViews(std::move(other.m_ImageViews))
        , m_ImageAvailableSemaphores(std::move(other.m_ImageAvailableSemaphores))
        , m_RenderFinishedSemaphores(std::move(other.m_RenderFinishedSemaphores))
        , m_InFlightFences(std::move(other.m_InFlightFences))
        , m_ImagesInFlight(std::move(other.m_ImagesInFlight))
        , m_CurrentFrame(other.m_CurrentFrame)
    {
        other.m_Context = nullptr;
        other.m_Window = nullptr;
        other.m_Swapchain = VK_NULL_HANDLE;
    }

    Swapchain& Swapchain::operator=(Swapchain&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        Destroy();

        m_Context = other.m_Context;
        m_Window = other.m_Window;
        m_Config = other.m_Config;
        m_Swapchain = other.m_Swapchain;
        m_ImageFormat = other.m_ImageFormat;
        m_ColorSpace = other.m_ColorSpace;
        m_PresentMode = other.m_PresentMode;
        m_Extent = other.m_Extent;
        m_Images = std::move(other.m_Images);
        m_ImageViews = std::move(other.m_ImageViews);
        m_ImageAvailableSemaphores = std::move(other.m_ImageAvailableSemaphores);
        m_RenderFinishedSemaphores = std::move(other.m_RenderFinishedSemaphores);
        m_InFlightFences = std::move(other.m_InFlightFences);
        m_ImagesInFlight = std::move(other.m_ImagesInFlight);
        m_CurrentFrame = other.m_CurrentFrame;

        other.m_Context = nullptr;
        other.m_Window = nullptr;
        other.m_Swapchain = VK_NULL_HANDLE;

        return *this;
    }

    void Swapchain::Initialize()
    {
        if (m_Config.maxFramesInFlight == 0)
        {
            m_Config.maxFramesInFlight = 1;
        }

        CreateSwapchain(VK_NULL_HANDLE);
        CreateImageViews();
        CreateSyncObjects();
    }

    void Swapchain::Destroy()
    {
        if (m_Context == nullptr)
        {
            return;
        }

        DestroySyncObjects();
        DestroyImageViews();

        if (m_Swapchain != VK_NULL_HANDLE)
        {
            vkDestroySwapchainKHR(m_Context->GetDevice(), m_Swapchain, nullptr);
            m_Swapchain = VK_NULL_HANDLE;
        }

        m_Images.clear();
    }

    VkSurfaceFormatKHR Swapchain::ChooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available) const
    {
        for (const VkSurfaceFormatKHR& format : available)
        {
            if (format.format == m_Config.preferredFormat && format.colorSpace == m_Config.preferredColorSpace)
            {
                return format;
            }
        }

        for (const VkSurfaceFormatKHR& format : available)
        {
            if (format.format == VK_FORMAT_R8G8B8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return format;
            }
        }

        return available[0];
    }

    VkPresentModeKHR Swapchain::ChoosePresentMode(const std::vector<VkPresentModeKHR>& available) const
    {
        if (m_Config.vsync)
        {
            return VK_PRESENT_MODE_FIFO_KHR;
        }

        if (ContainsPresentMode(available, VK_PRESENT_MODE_MAILBOX_KHR))
        {
            return VK_PRESENT_MODE_MAILBOX_KHR;
        }

        if (ContainsPresentMode(available, VK_PRESENT_MODE_IMMEDIATE_KHR))
        {
            return VK_PRESENT_MODE_IMMEDIATE_KHR;
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D Swapchain::ChooseExtent(const VkSurfaceCapabilitiesKHR& capabilities) const
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        {
            return capabilities.currentExtent;
        }

        VkExtent2D extent{};
        extent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, m_Window->GetWidth()));
        extent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, m_Window->GetHeight()));
        return extent;
    }

    uint32_t Swapchain::ChooseImageCount(const VkSurfaceCapabilitiesKHR& capabilities) const
    {
        uint32_t imageCount = m_Config.desiredImageCount;

        if (imageCount < capabilities.minImageCount)
        {
            imageCount = capabilities.minImageCount;
        }

        if (capabilities.maxImageCount > 0)
        {
            if (imageCount > capabilities.maxImageCount)
            {
                imageCount = capabilities.maxImageCount;
            }
        }

        return imageCount;
    }

    void Swapchain::CreateSwapchain(VkSwapchainKHR oldSwapchain)
    {
        const SwapchainSupportDetails support = m_Context->QuerySwapchainSupport();

        if (!support.IsAdequate())
        {
            throw std::runtime_error("Surface does not support any formats or present modes");
        }

        const VkSurfaceFormatKHR surfaceFormat = ChooseSurfaceFormat(support.formats);
        const VkPresentModeKHR presentMode = ChoosePresentMode(support.presentModes);
        const VkExtent2D extent = ChooseExtent(support.capabilities);
        const uint32_t imageCount = ChooseImageCount(support.capabilities);

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = m_Context->GetSurface();
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = m_Config.imageUsage;
        createInfo.preTransform = support.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = oldSwapchain;

        const QueueFamilyIndices& indices = m_Context->GetQueueFamilyIndices();
        const uint32_t families[2] = { indices.graphicsFamily, indices.presentFamily };

        if (indices.graphicsFamily != indices.presentFamily)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = families;
        }
        else
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        }

        CheckResult(vkCreateSwapchainKHR(m_Context->GetDevice(), &createInfo, nullptr, &m_Swapchain), "Failed to create swapchain");

        uint32_t actualImageCount = 0;
        vkGetSwapchainImagesKHR(m_Context->GetDevice(), m_Swapchain, &actualImageCount, nullptr);

        m_Images.resize(actualImageCount);
        vkGetSwapchainImagesKHR(m_Context->GetDevice(), m_Swapchain, &actualImageCount, m_Images.data());

        m_ImageFormat = surfaceFormat.format;
        m_ColorSpace = surfaceFormat.colorSpace;
        m_PresentMode = presentMode;
        m_Extent = extent;
    }

    void Swapchain::CreateImageViews()
    {
        m_ImageViews.resize(m_Images.size());

        for (size_t i = 0; i < m_Images.size(); i++)
        {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = m_Images[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = m_ImageFormat;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            CheckResult(vkCreateImageView(m_Context->GetDevice(), &createInfo, nullptr, &m_ImageViews[i]), "Failed to create swapchain image view");
        }
    }

    void Swapchain::DestroyImageViews()
    {
        for (VkImageView view : m_ImageViews)
        {
            if (view != VK_NULL_HANDLE)
            {
                vkDestroyImageView(m_Context->GetDevice(), view, nullptr);
            }
        }

        m_ImageViews.clear();
    }

    void Swapchain::CreateSyncObjects()
    {
        m_ImageAvailableSemaphores.resize(m_Config.maxFramesInFlight);
        m_InFlightFences.resize(m_Config.maxFramesInFlight);
        m_RenderFinishedSemaphores.resize(m_Images.size());
        m_ImagesInFlight.assign(m_Images.size(), VK_NULL_HANDLE);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (uint32_t i = 0; i < m_Config.maxFramesInFlight; i++)
        {
            CheckResult(vkCreateSemaphore(m_Context->GetDevice(), &semaphoreInfo, nullptr, &m_ImageAvailableSemaphores[i]), "Failed to create image available semaphore");
            CheckResult(vkCreateFence(m_Context->GetDevice(), &fenceInfo, nullptr, &m_InFlightFences[i]), "Failed to create in flight fence");
        }

        for (size_t i = 0; i < m_Images.size(); i++)
        {
            CheckResult(vkCreateSemaphore(m_Context->GetDevice(), &semaphoreInfo, nullptr, &m_RenderFinishedSemaphores[i]), "Failed to create render finished semaphore");
        }

        m_CurrentFrame = 0;
    }

    void Swapchain::DestroySyncObjects()
    {
        for (VkSemaphore semaphore : m_ImageAvailableSemaphores)
        {
            if (semaphore != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(m_Context->GetDevice(), semaphore, nullptr);
            }
        }

        for (VkSemaphore semaphore : m_RenderFinishedSemaphores)
        {
            if (semaphore != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(m_Context->GetDevice(), semaphore, nullptr);
            }
        }

        for (VkFence fence : m_InFlightFences)
        {
            if (fence != VK_NULL_HANDLE)
            {
                vkDestroyFence(m_Context->GetDevice(), fence, nullptr);
            }
        }

        m_ImageAvailableSemaphores.clear();
        m_RenderFinishedSemaphores.clear();
        m_InFlightFences.clear();
        m_ImagesInFlight.clear();
    }

    void Swapchain::Recreate()
    {
        while (m_Window->IsMinimized())
        {
            m_Window->WaitEvents();
        }

        m_Context->WaitIdle();

        const size_t previousImageCount = m_Images.size();

        VkSwapchainKHR oldSwapchain = m_Swapchain;
        m_Swapchain = VK_NULL_HANDLE;

        DestroyImageViews();
        CreateSwapchain(oldSwapchain);

        if (oldSwapchain != VK_NULL_HANDLE)
        {
            vkDestroySwapchainKHR(m_Context->GetDevice(), oldSwapchain, nullptr);
        }

        CreateImageViews();

        if (m_Images.size() != previousImageCount)
        {
            DestroySyncObjects();
            CreateSyncObjects();
        }
        else
        {
            m_ImagesInFlight.assign(m_Images.size(), VK_NULL_HANDLE);
        }

        m_Window->ResetResizeFlag();
    }

    void Swapchain::SetVSync(bool enabled)
    {
        if (m_Config.vsync == enabled)
        {
            return;
        }

        m_Config.vsync = enabled;
        Recreate();
    }

    float Swapchain::GetAspectRatio() const
    {
        if (m_Extent.height == 0)
        {
            return 1.0f;
        }

        return static_cast<float>(m_Extent.width) / static_cast<float>(m_Extent.height);
    }

    SwapchainStatus Swapchain::AcquireNextImage(uint32_t& outImageIndex)
    {
        vkWaitForFences(m_Context->GetDevice(), 1, &m_InFlightFences[m_CurrentFrame], VK_TRUE, UINT64_MAX);

        const VkResult result = vkAcquireNextImageKHR(
            m_Context->GetDevice(),
            m_Swapchain,
            UINT64_MAX,
            m_ImageAvailableSemaphores[m_CurrentFrame],
            VK_NULL_HANDLE,
            &outImageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            return SwapchainStatus::OutOfDate;
        }

        if (result == VK_SUBOPTIMAL_KHR)
        {
            return SwapchainStatus::Suboptimal;
        }

        CheckResult(result, "Failed to acquire swapchain image");

        return SwapchainStatus::Success;
    }

    SwapchainStatus Swapchain::SubmitAndPresent(VkCommandBuffer commandBuffer, uint32_t imageIndex)
    {
        if (m_ImagesInFlight[imageIndex] != VK_NULL_HANDLE)
        {
            vkWaitForFences(m_Context->GetDevice(), 1, &m_ImagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        }

        m_ImagesInFlight[imageIndex] = m_InFlightFences[m_CurrentFrame];

        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &m_ImageAvailableSemaphores[m_CurrentFrame];
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &m_RenderFinishedSemaphores[imageIndex];

        vkResetFences(m_Context->GetDevice(), 1, &m_InFlightFences[m_CurrentFrame]);

        CheckResult(
            vkQueueSubmit(m_Context->GetGraphicsQueue(), 1, &submitInfo, m_InFlightFences[m_CurrentFrame]),
            "Failed to submit draw command buffer");

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &m_RenderFinishedSemaphores[imageIndex];
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &m_Swapchain;
        presentInfo.pImageIndices = &imageIndex;

        const VkResult result = vkQueuePresentKHR(m_Context->GetPresentQueue(), &presentInfo);

        m_CurrentFrame = (m_CurrentFrame + 1) % m_Config.maxFramesInFlight;

        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            return SwapchainStatus::OutOfDate;
        }

        if (result == VK_SUBOPTIMAL_KHR)
        {
            return SwapchainStatus::Suboptimal;
        }

        CheckResult(result, "Failed to present swapchain image");

        return SwapchainStatus::Success;
    }

} // namespace Core