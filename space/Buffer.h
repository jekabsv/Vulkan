#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"

namespace Core
{

    class VulkanContext;

    enum class BufferType : uint8_t
    {
        Vertex = 0,
        Index,
        Uniform,
        Storage,
        Staging
    };

    enum class MemoryUsage : uint8_t
    {
        DeviceLocal = 0,
        HostVisible,
        HostReadback
    };

    class Buffer
    {
    public:
        Buffer(VulkanContext& context, VkDeviceSize size, BufferType type, MemoryUsage memoryUsage);
        ~Buffer();

        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        Buffer(Buffer&& other) noexcept;
        Buffer& operator=(Buffer&& other) noexcept;

        void SetData(const void* data, VkDeviceSize size, VkDeviceSize offset);
        void SetData(const void* data, VkDeviceSize size);

        void CopyFrom(const Buffer& source, VkDeviceSize size, VkDeviceSize sourceOffset, VkDeviceSize destinationOffset);

        void Flush(VkDeviceSize size, VkDeviceSize offset);
        void Invalidate(VkDeviceSize size, VkDeviceSize offset);

        void BindAsVertexBuffer(VkCommandBuffer commandBuffer, uint32_t binding) const;
        void BindAsIndexBuffer(VkCommandBuffer commandBuffer, VkIndexType indexType) const;

        VkDescriptorBufferInfo GetDescriptorInfo(VkDeviceSize size, VkDeviceSize offset) const;
        VkDescriptorBufferInfo GetDescriptorInfo() const;

        VkBuffer GetHandle() const { return m_Buffer; }
        VmaAllocation GetAllocation() const { return m_Allocation; }
        VkDeviceSize GetSize() const { return m_Size; }
        BufferType GetType() const { return m_Type; }
        MemoryUsage GetMemoryUsage() const { return m_MemoryUsage; }
        bool IsMapped() const { return m_MappedData != nullptr; }
        void* GetMappedData() const { return m_MappedData; }

    private:
        void Destroy();

        static VkBufferUsageFlags GetUsageFlags(BufferType type);

    private:
        VulkanContext* m_Context{ nullptr };
        VkBuffer m_Buffer{ VK_NULL_HANDLE };
        VmaAllocation m_Allocation{ VK_NULL_HANDLE };
        void* m_MappedData{ nullptr };

        VkDeviceSize m_Size{ 0 };
        BufferType m_Type{ BufferType::Vertex };
        MemoryUsage m_MemoryUsage{ MemoryUsage::DeviceLocal };
    };

} // namespace Core