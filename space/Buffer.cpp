#include "Buffer.h"

#include "VulkanContext.h"

#include <cstring>
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
    }

    VkBufferUsageFlags Buffer::GetUsageFlags(BufferType type)
    {
        if (type == BufferType::Vertex)
        {
            return VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        }

        if (type == BufferType::Index)
        {
            return VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        }

        if (type == BufferType::Uniform)
        {
            return VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        }

        if (type == BufferType::Storage)
        {
            return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        }

        return VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    }

    Buffer::Buffer(VulkanContext& context, VkDeviceSize size, BufferType type, MemoryUsage memoryUsage)
        : m_Context(&context)
        , m_Size(size)
        , m_Type(type)
        , m_MemoryUsage(memoryUsage)
    {
        if (size == 0)
        {
            throw std::runtime_error("Cannot create a buffer of size zero");
        }

        if (type == BufferType::Staging && memoryUsage == MemoryUsage::DeviceLocal)
        {
            m_MemoryUsage = MemoryUsage::HostVisible;
        }

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = GetUsageFlags(type);
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocationInfo{};
        allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;

        if (m_MemoryUsage == MemoryUsage::HostVisible)
        {
            allocationInfo.flags =
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }

        if (m_MemoryUsage == MemoryUsage::HostReadback)
        {
            allocationInfo.flags =
                VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }

        VmaAllocationInfo result{};

        CheckResult(
            vmaCreateBuffer(m_Context->GetAllocator(), &bufferInfo, &allocationInfo, &m_Buffer, &m_Allocation, &result),
            "Failed to create buffer");

        m_MappedData = result.pMappedData;
    }

    Buffer::~Buffer()
    {
        Destroy();
    }

    Buffer::Buffer(Buffer&& other) noexcept
        : m_Context(other.m_Context)
        , m_Buffer(other.m_Buffer)
        , m_Allocation(other.m_Allocation)
        , m_MappedData(other.m_MappedData)
        , m_Size(other.m_Size)
        , m_Type(other.m_Type)
        , m_MemoryUsage(other.m_MemoryUsage)
    {
        other.m_Context = nullptr;
        other.m_Buffer = VK_NULL_HANDLE;
        other.m_Allocation = VK_NULL_HANDLE;
        other.m_MappedData = nullptr;
        other.m_Size = 0;
    }

    Buffer& Buffer::operator=(Buffer&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        Destroy();

        m_Context = other.m_Context;
        m_Buffer = other.m_Buffer;
        m_Allocation = other.m_Allocation;
        m_MappedData = other.m_MappedData;
        m_Size = other.m_Size;
        m_Type = other.m_Type;
        m_MemoryUsage = other.m_MemoryUsage;

        other.m_Context = nullptr;
        other.m_Buffer = VK_NULL_HANDLE;
        other.m_Allocation = VK_NULL_HANDLE;
        other.m_MappedData = nullptr;
        other.m_Size = 0;

        return *this;
    }

    void Buffer::Destroy()
    {
        if (m_Context == nullptr)
        {
            return;
        }

        if (m_Buffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(m_Context->GetAllocator(), m_Buffer, m_Allocation);
            m_Buffer = VK_NULL_HANDLE;
            m_Allocation = VK_NULL_HANDLE;
            m_MappedData = nullptr;
        }
    }

    void Buffer::SetData(const void* data, VkDeviceSize size)
    {
        SetData(data, size, 0);
    }

    void Buffer::SetData(const void* data, VkDeviceSize size, VkDeviceSize offset)
    {
        if (data == nullptr)
        {
            throw std::runtime_error("Cannot upload data from a null pointer");
        }

        if (offset + size > m_Size)
        {
            throw std::runtime_error("Upload range exceeds buffer size");
        }

        if (IsMapped())
        {
            uint8_t* destination = static_cast<uint8_t*>(m_MappedData) + offset;
            std::memcpy(destination, data, static_cast<size_t>(size));
            Flush(size, offset);
            return;
        }

        Buffer staging(*m_Context, size, BufferType::Staging, MemoryUsage::HostVisible);
        staging.SetData(data, size, 0);

        CopyFrom(staging, size, 0, offset);
    }

    void Buffer::CopyFrom(const Buffer& source, VkDeviceSize size, VkDeviceSize sourceOffset, VkDeviceSize destinationOffset)
    {
        if (destinationOffset + size > m_Size)
        {
            throw std::runtime_error("Copy range exceeds destination buffer size");
        }

        if (sourceOffset + size > source.GetSize())
        {
            throw std::runtime_error("Copy range exceeds source buffer size");
        }

        VkCommandBuffer commandBuffer = m_Context->BeginSingleTimeCommands();

        VkBufferCopy region{};
        region.srcOffset = sourceOffset;
        region.dstOffset = destinationOffset;
        region.size = size;

        vkCmdCopyBuffer(commandBuffer, source.GetHandle(), m_Buffer, 1, &region);

        m_Context->EndSingleTimeCommands(commandBuffer);
    }

    void Buffer::Flush(VkDeviceSize size, VkDeviceSize offset)
    {
        if (!IsMapped())
        {
            return;
        }

        CheckResult(vmaFlushAllocation(m_Context->GetAllocator(), m_Allocation, offset, size), "Failed to flush buffer memory");
    }

    void Buffer::Invalidate(VkDeviceSize size, VkDeviceSize offset)
    {
        if (!IsMapped())
        {
            return;
        }

        CheckResult(vmaInvalidateAllocation(m_Context->GetAllocator(), m_Allocation, offset, size), "Failed to invalidate buffer memory");
    }

    void Buffer::BindAsVertexBuffer(VkCommandBuffer commandBuffer, uint32_t binding) const
    {
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(commandBuffer, binding, 1, &m_Buffer, &offset);
    }

    void Buffer::BindAsIndexBuffer(VkCommandBuffer commandBuffer, VkIndexType indexType) const
    {
        vkCmdBindIndexBuffer(commandBuffer, m_Buffer, 0, indexType);
    }

    VkDescriptorBufferInfo Buffer::GetDescriptorInfo(VkDeviceSize size, VkDeviceSize offset) const
    {
        VkDescriptorBufferInfo info{};
        info.buffer = m_Buffer;
        info.offset = offset;
        info.range = size;
        return info;
    }

    VkDescriptorBufferInfo Buffer::GetDescriptorInfo() const
    {
        return GetDescriptorInfo(m_Size, 0);
    }

} // namespace Core