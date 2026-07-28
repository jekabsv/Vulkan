#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

namespace Core
{

    class Window;
    class VulkanContext;

    enum class SwapchainStatus : uint8_t
    {
        Success = 0,
        Suboptimal,
        OutOfDate
    };

    struct SwapchainConfig
    {
        bool vsync{ true };
        uint32_t desiredImageCount{ 3 };
        uint32_t maxFramesInFlight{ 2 };
        VkFormat preferredFormat{ VK_FORMAT_B8G8R8A8_SRGB };
        VkColorSpaceKHR preferredColorSpace{ VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
        VkImageUsageFlags imageUsage{ VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT };
    };

    class Swapchain
    {
    public:
        Swapchain(VulkanContext& context, Window& window, const SwapchainConfig& config);
        Swapchain(VulkanContext& context, Window& window);
        ~Swapchain();

        Swapchain(const Swapchain&) = delete;
        Swapchain& operator=(const Swapchain&) = delete;
        Swapchain(Swapchain&& other) noexcept;
        Swapchain& operator=(Swapchain&& other) noexcept;

        SwapchainStatus AcquireNextImage(uint32_t& outImageIndex);
        SwapchainStatus SubmitAndPresent(VkCommandBuffer commandBuffer, uint32_t imageIndex);
        void Recreate();

        void SetVSync(bool enabled);
        bool IsVSyncEnabled() const { return m_Config.vsync; }

        VkSwapchainKHR GetHandle() const { return m_Swapchain; }
        VkFormat GetImageFormat() const { return m_ImageFormat; }
        VkColorSpaceKHR GetColorSpace() const { return m_ColorSpace; }
        VkPresentModeKHR GetPresentMode() const { return m_PresentMode; }
        VkExtent2D GetExtent() const { return m_Extent; }
        uint32_t GetWidth() const { return m_Extent.width; }
        uint32_t GetHeight() const { return m_Extent.height; }
        float GetAspectRatio() const;

        uint32_t GetImageCount() const { return static_cast<uint32_t>(m_Images.size()); }
        const std::vector<VkImage>& GetImages() const { return m_Images; }
        const std::vector<VkImageView>& GetImageViews() const { return m_ImageViews; }
        VkImage GetImage(uint32_t index) const { return m_Images[index]; }
        VkImageView GetImageView(uint32_t index) const { return m_ImageViews[index]; }

        uint32_t GetMaxFramesInFlight() const { return m_Config.maxFramesInFlight; }
        uint32_t GetCurrentFrame() const { return m_CurrentFrame; }
        VkSemaphore GetImageAvailableSemaphore() const { return m_ImageAvailableSemaphores[m_CurrentFrame]; }
        VkSemaphore GetRenderFinishedSemaphore(uint32_t imageIndex) const { return m_RenderFinishedSemaphores[imageIndex]; }
        VkFence GetInFlightFence() const { return m_InFlightFences[m_CurrentFrame]; }

    private:
        void Initialize();
        void CreateSwapchain(VkSwapchainKHR oldSwapchain);
        void CreateImageViews();
        void CreateSyncObjects();
        void DestroyImageViews();
        void DestroySyncObjects();
        void Destroy();

        VkSurfaceFormatKHR ChooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available) const;
        VkPresentModeKHR ChoosePresentMode(const std::vector<VkPresentModeKHR>& available) const;
        VkExtent2D ChooseExtent(const VkSurfaceCapabilitiesKHR& capabilities) const;
        uint32_t ChooseImageCount(const VkSurfaceCapabilitiesKHR& capabilities) const;

    private:
        VulkanContext* m_Context{ nullptr };
        Window* m_Window{ nullptr };
        SwapchainConfig m_Config;

        VkSwapchainKHR m_Swapchain{ VK_NULL_HANDLE };
        VkFormat m_ImageFormat{ VK_FORMAT_UNDEFINED };
        VkColorSpaceKHR m_ColorSpace{ VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
        VkPresentModeKHR m_PresentMode{ VK_PRESENT_MODE_FIFO_KHR };
        VkExtent2D m_Extent{ 0, 0 };

        std::vector<VkImage> m_Images;
        std::vector<VkImageView> m_ImageViews;

        std::vector<VkSemaphore> m_ImageAvailableSemaphores;
        std::vector<VkSemaphore> m_RenderFinishedSemaphores;
        std::vector<VkFence> m_InFlightFences;
        std::vector<VkFence> m_ImagesInFlight;

        uint32_t m_CurrentFrame{ 0 };
    };

} // namespace Core