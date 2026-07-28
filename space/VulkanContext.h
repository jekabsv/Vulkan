#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"

namespace Core
{
    class Window;

    constexpr uint32_t InvalidQueueFamily = UINT32_MAX;

    struct QueueFamilyIndices
    {
        uint32_t graphicsFamily{ InvalidQueueFamily };
        uint32_t presentFamily{ InvalidQueueFamily };
        uint32_t transferFamily{ InvalidQueueFamily };

        bool IsComplete() const
        {
            return graphicsFamily != InvalidQueueFamily && presentFamily != InvalidQueueFamily;
        }

        bool HasDedicatedTransfer() const
        {
            return transferFamily != InvalidQueueFamily && transferFamily != graphicsFamily;
        }
    };

    struct SwapchainSupportDetails
    {
        VkSurfaceCapabilitiesKHR capabilities{};
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;

        bool IsAdequate() const
        {
            return !formats.empty() && !presentModes.empty();
        }
    };

    struct VulkanContextConfig
    {
        std::string applicationName{ "Application" };
        std::string engineName{ "Engine" };
        uint32_t applicationVersion{ VK_MAKE_API_VERSION(0, 1, 0, 0) };
        uint32_t engineVersion{ VK_MAKE_API_VERSION(0, 1, 0, 0) };
        uint32_t apiVersion{ VK_API_VERSION_1_3 };

        bool enableValidationLayers{ true };
        bool preferDiscreteGpu{ true };
        bool requestDedicatedTransferQueue{ true };

        std::vector<const char*> requiredDeviceExtensions{ VK_KHR_SWAPCHAIN_EXTENSION_NAME };
        std::vector<const char*> optionalDeviceExtensions;
    };

    class VulkanContext
    {
    public:
        VulkanContext(Window& window, const VulkanContextConfig& config);
        explicit VulkanContext(Window& window);
        ~VulkanContext();

        VulkanContext(const VulkanContext&) = delete;
        VulkanContext& operator=(const VulkanContext&) = delete;
        VulkanContext(VulkanContext&& other) noexcept;
        VulkanContext& operator=(VulkanContext&& other) noexcept;

        VkInstance GetInstance() const { return m_Instance; }
        VkSurfaceKHR GetSurface() const { return m_Surface; }
        VkPhysicalDevice GetPhysicalDevice() const { return m_PhysicalDevice; }
        VkDevice GetDevice() const { return m_Device; }

        VkQueue GetGraphicsQueue() const { return m_GraphicsQueue; }
        VkQueue GetPresentQueue() const { return m_PresentQueue; }
        VkQueue GetTransferQueue() const { return m_TransferQueue; }

        VmaAllocator GetAllocator() const { return m_Allocator; }

        VkCommandBuffer BeginSingleTimeCommands() const;
        void EndSingleTimeCommands(VkCommandBuffer commandBuffer) const;

        const QueueFamilyIndices& GetQueueFamilyIndices() const { return m_QueueFamilyIndices; }
        const VkPhysicalDeviceProperties& GetDeviceProperties() const { return m_DeviceProperties; }
        const VkPhysicalDeviceFeatures& GetEnabledFeatures() const { return m_EnabledFeatures; }
        const VkPhysicalDeviceMemoryProperties& GetMemoryProperties() const { return m_MemoryProperties; }

        bool IsValidationEnabled() const { return m_ValidationEnabled; }
        bool IsDeviceExtensionEnabled(const char* extensionName) const;

        SwapchainSupportDetails QuerySwapchainSupport() const;
        VkSampleCountFlagBits GetMaxUsableSampleCount() const;
        VkFormat FindSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) const;
        VkFormat FindDepthFormat() const;
        uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;

        void WaitIdle() const;

    private:
        void Initialize();
        void CreateInstance();
        void SetupDebugMessenger();
        void CreateSurface();
        void PickPhysicalDevice();
        void CreateLogicalDevice();
        void CreateAllocator();
        void CreateSingleTimeCommandPool();
        void Destroy();

        bool CheckValidationLayerSupport() const;
        std::vector<const char*> GatherInstanceExtensions() const;
        uint32_t RateDevice(VkPhysicalDevice device) const;
        bool IsDeviceSuitable(VkPhysicalDevice device) const;
        bool CheckDeviceExtensionSupport(VkPhysicalDevice device) const;
        QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device) const;
        SwapchainSupportDetails QuerySwapchainSupport(VkPhysicalDevice device) const;

    private:
        Window* m_Window{ nullptr };
        VulkanContextConfig m_Config;

        VkInstance m_Instance{ VK_NULL_HANDLE };
        VkDebugUtilsMessengerEXT m_DebugMessenger{ VK_NULL_HANDLE };
        VkSurfaceKHR m_Surface{ VK_NULL_HANDLE };
        VkPhysicalDevice m_PhysicalDevice{ VK_NULL_HANDLE };
        VkDevice m_Device{ VK_NULL_HANDLE };
        VmaAllocator m_Allocator{ VK_NULL_HANDLE };
        VkCommandPool m_SingleTimeCommandPool{ VK_NULL_HANDLE };

        QueueFamilyIndices m_QueueFamilyIndices;
        VkQueue m_GraphicsQueue{ VK_NULL_HANDLE };
        VkQueue m_PresentQueue{ VK_NULL_HANDLE };
        VkQueue m_TransferQueue{ VK_NULL_HANDLE };

        VkPhysicalDeviceProperties m_DeviceProperties{};
        VkPhysicalDeviceFeatures m_EnabledFeatures{};
        VkPhysicalDeviceMemoryProperties m_MemoryProperties{};

        std::vector<const char*> m_EnabledDeviceExtensions;
        bool m_ValidationEnabled{ false };
    };

} // namespace Core