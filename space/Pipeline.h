#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "Shader.h"

namespace Core
{

    class VulkanContext;

    enum class BlendMode : uint8_t
    {
        None = 0,
        AlphaBlend,
        Additive,
        Multiply
    };

    struct VertexInputLayout
    {
        std::vector<VkVertexInputBindingDescription> bindings;
        std::vector<VkVertexInputAttributeDescription> attributes;

        VertexInputLayout& AddBinding(uint32_t binding, uint32_t stride, VkVertexInputRate inputRate);
        VertexInputLayout& AddAttribute(uint32_t location, uint32_t binding, VkFormat format, uint32_t offset);
    };

    struct PipelineConfig
    {
        VertexInputLayout vertexInput;

        VkPrimitiveTopology topology{ VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
        VkPolygonMode polygonMode{ VK_POLYGON_MODE_FILL };
        VkCullModeFlags cullMode{ VK_CULL_MODE_BACK_BIT };
        VkFrontFace frontFace{ VK_FRONT_FACE_COUNTER_CLOCKWISE };
        float lineWidth{ 1.0f };

        bool depthTestEnable{ true };
        bool depthWriteEnable{ true };
        VkCompareOp depthCompareOp{ VK_COMPARE_OP_LESS };
        bool stencilTestEnable{ false };
        VkStencilOpState stencilFront{};
        VkStencilOpState stencilBack{};

        BlendMode blendMode{ BlendMode::None };
        VkSampleCountFlagBits sampleCount{ VK_SAMPLE_COUNT_1_BIT };

        std::vector<VkFormat> colorAttachmentFormats;
        VkFormat depthAttachmentFormat{ VK_FORMAT_UNDEFINED };
        VkFormat stencilAttachmentFormat{ VK_FORMAT_UNDEFINED };

        std::vector<VkDynamicState> dynamicStates{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    };

    class GraphicsPipeline
    {
    public:
        GraphicsPipeline(VulkanContext& context, const std::vector<const Shader*>& shaders, const PipelineConfig& config);
        ~GraphicsPipeline();

        GraphicsPipeline(const GraphicsPipeline&) = delete;
        GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
        GraphicsPipeline(GraphicsPipeline&& other) noexcept;
        GraphicsPipeline& operator=(GraphicsPipeline&& other) noexcept;

        void Bind(VkCommandBuffer commandBuffer) const;
        void BindDescriptorSets(VkCommandBuffer commandBuffer, uint32_t firstSet, const std::vector<VkDescriptorSet>& sets) const;
        void PushConstants(VkCommandBuffer commandBuffer, uint32_t offset, uint32_t size, const void* data) const;

        VkPipeline GetHandle() const { return m_Pipeline; }
        VkPipelineLayout GetLayout() const { return m_Layout; }
        const std::vector<VkDescriptorSetLayout>& GetDescriptorSetLayouts() const { return m_DescriptorSetLayouts; }
        VkDescriptorSetLayout GetDescriptorSetLayout(uint32_t set) const;
        const std::vector<ShaderBinding>& GetBindings() const { return m_Bindings; }
        bool FindDescriptorType(uint32_t set, uint32_t binding, VkDescriptorType& outType) const;
        const ShaderBinding* FindBinding(uint32_t set, uint32_t binding) const;
        const ShaderBinding* FindBindingByName(uint32_t set, const std::string& name) const;
        const ShaderBinding* FindUniformBlock(uint32_t set) const;
        bool FindMember(uint32_t set, const std::string& memberName, uint32_t& outBinding, uint32_t& outOffset, uint32_t& outSize) const;
        bool HasPushConstants() const { return m_PushConstantRange.size > 0; }

    private:
        void CreateLayout(const std::vector<const Shader*>& shaders);
        void CreatePipeline(const std::vector<const Shader*>& shaders, const PipelineConfig& config);
        void Destroy();

        static VkPipelineColorBlendAttachmentState MakeBlendAttachment(BlendMode mode);

    private:
        VulkanContext* m_Context{ nullptr };
        VkPipeline m_Pipeline{ VK_NULL_HANDLE };
        VkPipelineLayout m_Layout{ VK_NULL_HANDLE };
        std::vector<VkDescriptorSetLayout> m_DescriptorSetLayouts;
        std::vector<ShaderBinding> m_Bindings;
        VkPushConstantRange m_PushConstantRange{};
    };

} // namespace Core