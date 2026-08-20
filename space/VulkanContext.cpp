#include "VulkanContext.h"

#include "Window.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace Core
{

    namespace
    {
        const char* const ValidationLayerName = "VK_LAYER_KHRONOS_validation";

        void CheckResult(VkResult result, const char* message)
        {
            if (result != VK_SUCCESS)
            {
                throw std::runtime_error(message);
            }
        }

        bool ContainsExtension(const std::vector<VkExtensionProperties>& available, const char* name)
        {
            for (const VkExtensionProperties& extension : available)
            {
                if (std::strcmp(extension.extensionName, name) == 0)
                {
                    return true;
                }
            }

            return false;
        }

        VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT severity,
            VkDebugUtilsMessageTypeFlagsEXT,
            const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
            void*)
        {
            if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
            {
                std::cerr << "[Vulkan][error] " << callbackData->pMessage << std::endl;
            }
            else if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
            {
                std::cerr << "[Vulkan][warning] " << callbackData->pMessage << std::endl;
            }

            return VK_FALSE;
        }

        void PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
        {
            createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            createInfo.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            createInfo.messageType =
                VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            createInfo.pfnUserCallback = DebugCallback;
        }

        VkResult CreateDebugMessenger(
            VkInstance instance,
            const VkDebugUtilsMessengerCreateInfoEXT* createInfo,
            VkDebugUtilsMessengerEXT* messenger)
        {
            PFN_vkCreateDebugUtilsMessengerEXT function =
                reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                    vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));

            if (function == nullptr)
            {
                return VK_ERROR_EXTENSION_NOT_PRESENT;
            }

            return function(instance, createInfo, nullptr, messenger);
        }

        void DestroyDebugMessenger(VkInstance instance, VkDebugUtilsMessengerEXT messenger)
        {
            PFN_vkDestroyDebugUtilsMessengerEXT function =
                reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                    vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));

            if (function == nullptr)
            {
                return;
            }

            function(instance, messenger, nullptr);
        }
    }

    VulkanContext::VulkanContext(Window& window, const VulkanContextConfig& config)
        : m_Window(&window)
        , m_Config(config)
    {
        Initialize();
    }

    VulkanContext::VulkanContext(Window& window)
        : m_Window(&window)
    {
        Initialize();
    }

    VulkanContext::~VulkanContext()
    {
        Destroy();
    }

    VulkanContext::VulkanContext(VulkanContext&& other) noexcept
        : m_Window(other.m_Window)
        , m_Config(std::move(other.m_Config))
        , m_Instance(other.m_Instance)
        , m_DebugMessenger(other.m_DebugMessenger)
        , m_Surface(other.m_Surface)
        , m_PhysicalDevice(other.m_PhysicalDevice)
        , m_Device(other.m_Device)
        , m_Allocator(other.m_Allocator)
        , m_SingleTimeCommandPool(other.m_SingleTimeCommandPool)
        , m_QueueFamilyIndices(other.m_QueueFamilyIndices)
        , m_GraphicsQueue(other.m_GraphicsQueue)
        , m_PresentQueue(other.m_PresentQueue)
        , m_TransferQueue(other.m_TransferQueue)
        , m_ComputeQueue(other.m_ComputeQueue)
        , m_DeviceProperties(other.m_DeviceProperties)
        , m_EnabledFeatures(other.m_EnabledFeatures)
        , m_MemoryProperties(other.m_MemoryProperties)
        , m_EnabledDeviceExtensions(std::move(other.m_EnabledDeviceExtensions))
        , m_ValidationEnabled(other.m_ValidationEnabled)
    {
        other.m_Window = nullptr;
        other.m_Instance = VK_NULL_HANDLE;
        other.m_DebugMessenger = VK_NULL_HANDLE;
        other.m_Surface = VK_NULL_HANDLE;
        other.m_PhysicalDevice = VK_NULL_HANDLE;
        other.m_Device = VK_NULL_HANDLE;
        other.m_Allocator = VK_NULL_HANDLE;
        other.m_SingleTimeCommandPool = VK_NULL_HANDLE;
        other.m_GraphicsQueue = VK_NULL_HANDLE;
        other.m_PresentQueue = VK_NULL_HANDLE;
        other.m_TransferQueue = VK_NULL_HANDLE;
        other.m_ComputeQueue = VK_NULL_HANDLE;
    }

    VulkanContext& VulkanContext::operator=(VulkanContext&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        Destroy();

        m_Window = other.m_Window;
        m_Config = std::move(other.m_Config);
        m_Instance = other.m_Instance;
        m_DebugMessenger = other.m_DebugMessenger;
        m_Surface = other.m_Surface;
        m_PhysicalDevice = other.m_PhysicalDevice;
        m_Device = other.m_Device;
        m_Allocator = other.m_Allocator;
        m_SingleTimeCommandPool = other.m_SingleTimeCommandPool;
        m_QueueFamilyIndices = other.m_QueueFamilyIndices;
        m_GraphicsQueue = other.m_GraphicsQueue;
        m_PresentQueue = other.m_PresentQueue;
        m_TransferQueue = other.m_TransferQueue;
        m_ComputeQueue = other.m_ComputeQueue;
        m_DeviceProperties = other.m_DeviceProperties;
        m_EnabledFeatures = other.m_EnabledFeatures;
        m_MemoryProperties = other.m_MemoryProperties;
        m_EnabledDeviceExtensions = std::move(other.m_EnabledDeviceExtensions);
        m_ValidationEnabled = other.m_ValidationEnabled;

        other.m_Window = nullptr;
        other.m_Instance = VK_NULL_HANDLE;
        other.m_DebugMessenger = VK_NULL_HANDLE;
        other.m_Surface = VK_NULL_HANDLE;
        other.m_PhysicalDevice = VK_NULL_HANDLE;
        other.m_Device = VK_NULL_HANDLE;
        other.m_Allocator = VK_NULL_HANDLE;
        other.m_SingleTimeCommandPool = VK_NULL_HANDLE;
        other.m_GraphicsQueue = VK_NULL_HANDLE;
        other.m_PresentQueue = VK_NULL_HANDLE;
        other.m_TransferQueue = VK_NULL_HANDLE;
        other.m_ComputeQueue = VK_NULL_HANDLE;

        return *this;
    }

    void VulkanContext::Initialize()
    {
        m_ValidationEnabled = m_Config.enableValidationLayers;

        if (m_ValidationEnabled)
        {
            if (!CheckValidationLayerSupport())
            {
                std::cerr << "[Vulkan][warning] Validation layers requested but not available, continuing without them" << std::endl;
                m_ValidationEnabled = false;
            }
        }

        CreateInstance();
        SetupDebugMessenger();
        CreateSurface();
        PickPhysicalDevice();
        CreateLogicalDevice();
        CreateAllocator();
        CreateSingleTimeCommandPool();
    }

    void VulkanContext::Destroy()
    {
        if (m_SingleTimeCommandPool != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(m_Device, m_SingleTimeCommandPool, nullptr);
            m_SingleTimeCommandPool = VK_NULL_HANDLE;
        }

        if (m_Allocator != VK_NULL_HANDLE)
        {
            vmaDestroyAllocator(m_Allocator);
            m_Allocator = VK_NULL_HANDLE;
        }

        if (m_Device != VK_NULL_HANDLE)
        {
            vkDestroyDevice(m_Device, nullptr);
            m_Device = VK_NULL_HANDLE;
        }

        if (m_Surface != VK_NULL_HANDLE)
        {
            vkDestroySurfaceKHR(m_Instance, m_Surface, nullptr);
            m_Surface = VK_NULL_HANDLE;
        }

        if (m_DebugMessenger != VK_NULL_HANDLE)
        {
            DestroyDebugMessenger(m_Instance, m_DebugMessenger);
            m_DebugMessenger = VK_NULL_HANDLE;
        }

        if (m_Instance != VK_NULL_HANDLE)
        {
            vkDestroyInstance(m_Instance, nullptr);
            m_Instance = VK_NULL_HANDLE;
        }

        m_PhysicalDevice = VK_NULL_HANDLE;
        m_GraphicsQueue = VK_NULL_HANDLE;
        m_PresentQueue = VK_NULL_HANDLE;
        m_TransferQueue = VK_NULL_HANDLE;
        m_ComputeQueue = VK_NULL_HANDLE;
    }

    bool VulkanContext::CheckValidationLayerSupport() const
    {
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const VkLayerProperties& layer : availableLayers)
        {
            if (std::strcmp(layer.layerName, ValidationLayerName) == 0)
            {
                return true;
            }
        }

        return false;
    }

    std::vector<const char*> VulkanContext::GatherInstanceExtensions() const
    {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        if (glfwExtensions == nullptr)
        {
            throw std::runtime_error("GLFW reports that Vulkan surface creation is not supported on this platform");
        }

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (m_ValidationEnabled)
        {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    void VulkanContext::CreateInstance()
    {
        VkApplicationInfo applicationInfo{};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pApplicationName = m_Config.applicationName.c_str();
        applicationInfo.applicationVersion = m_Config.applicationVersion;
        applicationInfo.pEngineName = m_Config.engineName.c_str();
        applicationInfo.engineVersion = m_Config.engineVersion;
        applicationInfo.apiVersion = m_Config.apiVersion;

        const std::vector<const char*> extensions = GatherInstanceExtensions();

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &applicationInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};

        if (m_ValidationEnabled)
        {
            PopulateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.enabledLayerCount = 1;
            createInfo.ppEnabledLayerNames = &ValidationLayerName;
            createInfo.pNext = &debugCreateInfo;
        }
        else
        {
            createInfo.enabledLayerCount = 0;
            createInfo.ppEnabledLayerNames = nullptr;
            createInfo.pNext = nullptr;
        }

        CheckResult(vkCreateInstance(&createInfo, nullptr, &m_Instance), "Failed to create Vulkan instance");
    }

    void VulkanContext::SetupDebugMessenger()
    {
        if (!m_ValidationEnabled)
        {
            return;
        }

        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        PopulateDebugMessengerCreateInfo(createInfo);

        CheckResult(CreateDebugMessenger(m_Instance, &createInfo, &m_DebugMessenger), "Failed to create debug messenger");
    }

    void VulkanContext::CreateSurface()
    {
        CheckResult(
            glfwCreateWindowSurface(m_Instance, m_Window->GetNativeHandle(), nullptr, &m_Surface),
            "Failed to create window surface");
    }

    void VulkanContext::PickPhysicalDevice()
    {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(m_Instance, &deviceCount, nullptr);

        if (deviceCount == 0)
        {
            throw std::runtime_error("Failed to find a GPU with Vulkan support");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(m_Instance, &deviceCount, devices.data());

        VkPhysicalDevice bestDevice = VK_NULL_HANDLE;
        uint32_t bestScore = 0;

        for (VkPhysicalDevice device : devices)
        {
            if (!IsDeviceSuitable(device))
            {
                continue;
            }

            const uint32_t score = RateDevice(device);

            if (score > bestScore || bestDevice == VK_NULL_HANDLE)
            {
                bestDevice = device;
                bestScore = score;
            }
        }

        if (bestDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("Failed to find a suitable GPU");
        }

        m_PhysicalDevice = bestDevice;
        m_QueueFamilyIndices = FindQueueFamilies(m_PhysicalDevice);

        vkGetPhysicalDeviceProperties(m_PhysicalDevice, &m_DeviceProperties);
        vkGetPhysicalDeviceMemoryProperties(m_PhysicalDevice, &m_MemoryProperties);
    }

    bool VulkanContext::IsDeviceSuitable(VkPhysicalDevice device) const
    {
        const QueueFamilyIndices indices = FindQueueFamilies(device);

        if (!indices.IsComplete())
        {
            return false;
        }

        if (!CheckDeviceExtensionSupport(device))
        {
            return false;
        }

        const SwapchainSupportDetails support = QuerySwapchainSupport(device);

        if (!support.IsAdequate())
        {
            return false;
        }

        return true;
    }

    uint32_t VulkanContext::RateDevice(VkPhysicalDevice device) const
    {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);

        uint32_t score = 1;

        if (m_Config.preferDiscreteGpu)
        {
            if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            {
                score += 100000;
            }
        }

        score += properties.limits.maxImageDimension2D;

        const QueueFamilyIndices indices = FindQueueFamilies(device);

        if (indices.HasDedicatedTransfer())
        {
            score += 1000;
        }

        return score;
    }

    bool VulkanContext::CheckDeviceExtensionSupport(VkPhysicalDevice device) const
    {
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> available(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, available.data());

        for (const char* required : m_Config.requiredDeviceExtensions)
        {
            if (!ContainsExtension(available, required))
            {
                return false;
            }
        }

        return true;
    }

    QueueFamilyIndices VulkanContext::FindQueueFamilies(VkPhysicalDevice device) const
    {
        QueueFamilyIndices indices;

        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);

        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, families.data());

        // Compute-capable family with no graphics support: usable for async compute, and the
        // only fallback if the graphics family somehow lacks VK_QUEUE_COMPUTE_BIT.
        uint32_t asyncComputeFamily = InvalidQueueFamily;
        bool graphicsSupportsCompute = false;

        for (uint32_t i = 0; i < familyCount; i++)
        {
            const VkQueueFamilyProperties& family = families[i];

            if (family.queueCount == 0)
            {
                continue;
            }

            const bool supportsCompute = (family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;

            if (family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                // Vulkan only guarantees that *some* family exposes graphics and compute together,
                // not that the first graphics family does, so prefer one that exposes both.
                if (indices.graphicsFamily == InvalidQueueFamily || (supportsCompute && !graphicsSupportsCompute))
                {
                    indices.graphicsFamily = i;
                    graphicsSupportsCompute = supportsCompute;
                }
            }
            else if (supportsCompute)
            {
                if (asyncComputeFamily == InvalidQueueFamily)
                {
                    asyncComputeFamily = i;
                }
            }

            if (family.queueFlags & VK_QUEUE_TRANSFER_BIT)
            {
                if (!(family.queueFlags & VK_QUEUE_GRAPHICS_BIT))
                {
                    if (indices.transferFamily == InvalidQueueFamily)
                    {
                        indices.transferFamily = i;
                    }
                }
            }

            VkBool32 presentSupported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_Surface, &presentSupported);

            if (presentSupported == VK_TRUE)
            {
                if (indices.presentFamily == InvalidQueueFamily)
                {
                    indices.presentFamily = i;
                }
            }
        }

        if (!m_Config.requestDedicatedTransferQueue)
        {
            indices.transferFamily = indices.graphicsFamily;
        }

        if (indices.transferFamily == InvalidQueueFamily)
        {
            indices.transferFamily = indices.graphicsFamily;
        }

        if (m_Config.requestDedicatedComputeQueue && asyncComputeFamily != InvalidQueueFamily)
        {
            indices.computeFamily = asyncComputeFamily;
        }
        else if (graphicsSupportsCompute)
        {
            indices.computeFamily = indices.graphicsFamily;
        }
        else
        {
            indices.computeFamily = asyncComputeFamily;
        }

        return indices;
    }

    void VulkanContext::CreateLogicalDevice()
    {
        std::set<uint32_t> uniqueFamilies;
        uniqueFamilies.insert(m_QueueFamilyIndices.graphicsFamily);
        uniqueFamilies.insert(m_QueueFamilyIndices.presentFamily);

        if (m_QueueFamilyIndices.HasDedicatedTransfer())
        {
            uniqueFamilies.insert(m_QueueFamilyIndices.transferFamily);
        }

        if (m_QueueFamilyIndices.HasDedicatedCompute())
        {
            uniqueFamilies.insert(m_QueueFamilyIndices.computeFamily);
        }

        const float queuePriority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

        for (uint32_t family : uniqueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = family;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures supportedFeatures{};
        vkGetPhysicalDeviceFeatures(m_PhysicalDevice, &supportedFeatures);

        m_EnabledFeatures = {};

        if (supportedFeatures.samplerAnisotropy == VK_TRUE)
        {
            m_EnabledFeatures.samplerAnisotropy = VK_TRUE;
        }

        if (supportedFeatures.fillModeNonSolid == VK_TRUE)
        {
            m_EnabledFeatures.fillModeNonSolid = VK_TRUE;
        }

        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(m_PhysicalDevice, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> available(extensionCount);
        vkEnumerateDeviceExtensionProperties(m_PhysicalDevice, nullptr, &extensionCount, available.data());

        m_EnabledDeviceExtensions = m_Config.requiredDeviceExtensions;

        for (const char* optional : m_Config.optionalDeviceExtensions)
        {
            if (ContainsExtension(available, optional))
            {
                m_EnabledDeviceExtensions.push_back(optional);
            }
        }

        VkPhysicalDeviceVulkan13Features features13{};
        features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        features13.dynamicRendering = VK_TRUE;
        features13.synchronization2 = VK_TRUE;

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pNext = &features13;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &m_EnabledFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(m_EnabledDeviceExtensions.size());
        createInfo.ppEnabledExtensionNames = m_EnabledDeviceExtensions.data();

        CheckResult(vkCreateDevice(m_PhysicalDevice, &createInfo, nullptr, &m_Device), "Failed to create logical device");

        vkGetDeviceQueue(m_Device, m_QueueFamilyIndices.graphicsFamily, 0, &m_GraphicsQueue);
        vkGetDeviceQueue(m_Device, m_QueueFamilyIndices.presentFamily, 0, &m_PresentQueue);

        if (m_QueueFamilyIndices.HasDedicatedTransfer())
        {
            vkGetDeviceQueue(m_Device, m_QueueFamilyIndices.transferFamily, 0, &m_TransferQueue);
        }
        else
        {
            m_TransferQueue = m_GraphicsQueue;
        }

        if (m_QueueFamilyIndices.HasDedicatedCompute())
        {
            vkGetDeviceQueue(m_Device, m_QueueFamilyIndices.computeFamily, 0, &m_ComputeQueue);
        }
        else
        {
            m_ComputeQueue = m_GraphicsQueue;
        }
    }

    void VulkanContext::CreateAllocator()
    {
        VmaAllocatorCreateInfo createInfo{};
        createInfo.instance = m_Instance;
        createInfo.physicalDevice = m_PhysicalDevice;
        createInfo.device = m_Device;
        createInfo.vulkanApiVersion = m_Config.apiVersion;

        if (IsDeviceExtensionEnabled(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME))
        {
            createInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
        }

        CheckResult(vmaCreateAllocator(&createInfo, &m_Allocator), "Failed to create memory allocator");
    }

    void VulkanContext::CreateSingleTimeCommandPool()
    {
        VkCommandPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        createInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        createInfo.queueFamilyIndex = m_QueueFamilyIndices.graphicsFamily;

        CheckResult(
            vkCreateCommandPool(m_Device, &createInfo, nullptr, &m_SingleTimeCommandPool),
            "Failed to create single time command pool");
    }

    VkCommandBuffer VulkanContext::BeginSingleTimeCommands() const
    {
        VkCommandBufferAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocateInfo.commandPool = m_SingleTimeCommandPool;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        CheckResult(vkAllocateCommandBuffers(m_Device, &allocateInfo, &commandBuffer), "Failed to allocate transfer command buffer");

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        CheckResult(vkBeginCommandBuffer(commandBuffer, &beginInfo), "Failed to begin transfer command buffer");

        return commandBuffer;
    }

    void VulkanContext::EndSingleTimeCommands(VkCommandBuffer commandBuffer) const
    {
        CheckResult(vkEndCommandBuffer(commandBuffer), "Failed to end transfer command buffer");

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        CheckResult(vkQueueSubmit(m_GraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE), "Failed to submit transfer command buffer");
        CheckResult(vkQueueWaitIdle(m_GraphicsQueue), "Failed to wait for transfer to finish");

        vkFreeCommandBuffers(m_Device, m_SingleTimeCommandPool, 1, &commandBuffer);
    }

    bool VulkanContext::IsDeviceExtensionEnabled(const char* extensionName) const {
        for (const char* enabled : m_EnabledDeviceExtensions)
        {
            if (std::strcmp(enabled, extensionName) == 0)
            {
                return true;
            }
        }

        return false;
    }

    SwapchainSupportDetails VulkanContext::QuerySwapchainSupport() const
    {
        return QuerySwapchainSupport(m_PhysicalDevice);
    }

    SwapchainSupportDetails VulkanContext::QuerySwapchainSupport(VkPhysicalDevice device) const
    {
        SwapchainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, m_Surface, &details.capabilities);

        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_Surface, &formatCount, nullptr);

        if (formatCount > 0)
        {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_Surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_Surface, &presentModeCount, nullptr);

        if (presentModeCount > 0)
        {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_Surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    VkSampleCountFlagBits VulkanContext::GetMaxUsableSampleCount() const
    {
        const VkSampleCountFlags counts =
            m_DeviceProperties.limits.framebufferColorSampleCounts &
            m_DeviceProperties.limits.framebufferDepthSampleCounts;

        if (counts & VK_SAMPLE_COUNT_64_BIT)
        {
            return VK_SAMPLE_COUNT_64_BIT;
        }

        if (counts & VK_SAMPLE_COUNT_32_BIT)
        {
            return VK_SAMPLE_COUNT_32_BIT;
        }

        if (counts & VK_SAMPLE_COUNT_16_BIT)
        {
            return VK_SAMPLE_COUNT_16_BIT;
        }

        if (counts & VK_SAMPLE_COUNT_8_BIT)
        {
            return VK_SAMPLE_COUNT_8_BIT;
        }

        if (counts & VK_SAMPLE_COUNT_4_BIT)
        {
            return VK_SAMPLE_COUNT_4_BIT;
        }

        if (counts & VK_SAMPLE_COUNT_2_BIT)
        {
            return VK_SAMPLE_COUNT_2_BIT;
        }

        return VK_SAMPLE_COUNT_1_BIT;
    }

    VkFormat VulkanContext::FindSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) const
    {
        for (VkFormat format : candidates)
        {
            VkFormatProperties properties{};
            vkGetPhysicalDeviceFormatProperties(m_PhysicalDevice, format, &properties);

            if (tiling == VK_IMAGE_TILING_LINEAR)
            {
                if ((properties.linearTilingFeatures & features) == features)
                {
                    return format;
                }
            }

            if (tiling == VK_IMAGE_TILING_OPTIMAL)
            {
                if ((properties.optimalTilingFeatures & features) == features)
                {
                    return format;
                }
            }
        }

        throw std::runtime_error("Failed to find a supported format");
    }

    VkFormat VulkanContext::FindDepthFormat() const
    {
        const std::vector<VkFormat> candidates
        {
            VK_FORMAT_D32_SFLOAT,
            VK_FORMAT_D32_SFLOAT_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT
        };

        return FindSupportedFormat(candidates, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }

    uint32_t VulkanContext::FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const
    {
        for (uint32_t i = 0; i < m_MemoryProperties.memoryTypeCount; i++)
        {
            const bool typeAllowed = (typeFilter & (1u << i)) != 0;
            const bool propertiesMatch = (m_MemoryProperties.memoryTypes[i].propertyFlags & properties) == properties;

            if (typeAllowed && propertiesMatch)
            {
                return i;
            }
        }

        throw std::runtime_error("Failed to find a suitable memory type");
    }

    void VulkanContext::WaitIdle() const
    {
        if (m_Device == VK_NULL_HANDLE)
        {
            return;
        }

        vkDeviceWaitIdle(m_Device);
    }

} // namespace Core