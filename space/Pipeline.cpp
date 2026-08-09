#include "Pipeline.h"

#include "VulkanContext.h"

#include <map>
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

    VertexInputLayout& VertexInputLayout::AddBinding(uint32_t binding, uint32_t stride, VkVertexInputRate inputRate)
    {
        VkVertexInputBindingDescription description{};
        description.binding = binding;
        description.stride = stride;
        description.inputRate = inputRate;
        bindings.push_back(description);
        return *this;
    }

    VertexInputLayout& VertexInputLayout::AddAttribute(uint32_t location, uint32_t binding, VkFormat format, uint32_t offset)
    {
        VkVertexInputAttributeDescription description{};
        description.location = location;
        description.binding = binding;
        description.format = format;
        description.offset = offset;
        attributes.push_back(description);
        return *this;
    }

    Pipeline::Pipeline(VulkanContext& context)
        : m_Context(&context)
    {
    }

    Pipeline::Pipeline(Pipeline&& other) noexcept
        : m_Context(other.m_Context)
        , m_Pipeline(other.m_Pipeline)
        , m_Layout(other.m_Layout)
        , m_DescriptorSetLayouts(std::move(other.m_DescriptorSetLayouts))
        , m_Bindings(std::move(other.m_Bindings))
        , m_PushConstantRange(other.m_PushConstantRange)
    {
        other.m_Context = nullptr;
        other.m_Pipeline = VK_NULL_HANDLE;
        other.m_Layout = VK_NULL_HANDLE;
    }

    Pipeline& Pipeline::operator=(Pipeline&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        Destroy();

        m_Context = other.m_Context;
        m_Pipeline = other.m_Pipeline;
        m_Layout = other.m_Layout;
        m_DescriptorSetLayouts = std::move(other.m_DescriptorSetLayouts);
        m_Bindings = std::move(other.m_Bindings);
        m_PushConstantRange = other.m_PushConstantRange;

        other.m_Context = nullptr;
        other.m_Pipeline = VK_NULL_HANDLE;
        other.m_Layout = VK_NULL_HANDLE;

        return *this;
    }

    void Pipeline::Destroy()
    {
        if (m_Context == nullptr)
        {
            return;
        }

        if (m_Pipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(m_Context->GetDevice(), m_Pipeline, nullptr);
            m_Pipeline = VK_NULL_HANDLE;
        }

        if (m_Layout != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(m_Context->GetDevice(), m_Layout, nullptr);
            m_Layout = VK_NULL_HANDLE;
        }

        for (VkDescriptorSetLayout layout : m_DescriptorSetLayouts)
        {
            if (layout != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(m_Context->GetDevice(), layout, nullptr);
            }
        }

        m_DescriptorSetLayouts.clear();
    }

    void Pipeline::CreateLayout(const std::vector<const Shader*>& shaders)
    {
        std::map<uint32_t, std::map<uint32_t, VkDescriptorSetLayoutBinding>> merged;

        uint32_t pushConstantStages = 0;
        uint32_t pushConstantBegin = UINT32_MAX;
        uint32_t pushConstantEnd = 0;

        for (const Shader* shader : shaders)
        {
            const ShaderReflection& reflection = shader->GetReflection();

            for (const ShaderBinding& binding : reflection.bindings)
            {
                std::map<uint32_t, VkDescriptorSetLayoutBinding>& setBindings = merged[binding.set];

                if (setBindings.find(binding.binding) == setBindings.end())
                {
                    VkDescriptorSetLayoutBinding layoutBinding{};
                    layoutBinding.binding = binding.binding;
                    layoutBinding.descriptorType = binding.descriptorType;
                    layoutBinding.descriptorCount = binding.count;
                    layoutBinding.stageFlags = binding.stageFlags;
                    setBindings[binding.binding] = layoutBinding;
                    continue;
                }

                VkDescriptorSetLayoutBinding& existing = setBindings[binding.binding];

                if (existing.descriptorType != binding.descriptorType)
                {
                    throw std::runtime_error("Shader stages declare conflicting descriptor types for the same set and binding");
                }

                if (existing.descriptorCount != binding.count)
                {
                    throw std::runtime_error("Shader stages declare conflicting descriptor counts for the same set and binding");
                }

                existing.stageFlags |= binding.stageFlags;
            }

            for (const ShaderBinding& binding : reflection.bindings)
            {
                bool alreadyKnown = false;

                for (ShaderBinding& known : m_Bindings)
                {
                    if (known.set == binding.set && known.binding == binding.binding)
                    {
                        known.stageFlags |= binding.stageFlags;
                        alreadyKnown = true;
                        break;
                    }
                }

                if (!alreadyKnown)
                {
                    m_Bindings.push_back(binding);
                }
            }

            if (reflection.hasPushConstants)
            {
                pushConstantStages |= static_cast<uint32_t>(shader->GetStage());

                if (reflection.pushConstantOffset < pushConstantBegin)
                {
                    pushConstantBegin = reflection.pushConstantOffset;
                }

                const uint32_t end = reflection.pushConstantOffset + reflection.pushConstantSize;

                if (end > pushConstantEnd)
                {
                    pushConstantEnd = end;
                }
            }
        }

        uint32_t highestSet = 0;

        for (const auto& entry : merged)
        {
            if (entry.first > highestSet)
            {
                highestSet = entry.first;
            }
        }

        if (!merged.empty())
        {
            m_DescriptorSetLayouts.resize(highestSet + 1, VK_NULL_HANDLE);
        }

        for (uint32_t set = 0; set < m_DescriptorSetLayouts.size(); set++)
        {
            std::vector<VkDescriptorSetLayoutBinding> bindings;

            if (merged.find(set) != merged.end())
            {
                for (const auto& entry : merged[set])
                {
                    bindings.push_back(entry.second);
                }
            }

            VkDescriptorSetLayoutCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            createInfo.bindingCount = static_cast<uint32_t>(bindings.size());
            createInfo.pBindings = bindings.data();

            CheckResult(
                vkCreateDescriptorSetLayout(m_Context->GetDevice(), &createInfo, nullptr, &m_DescriptorSetLayouts[set]),
                "Failed to create descriptor set layout");
        }

        m_PushConstantRange = {};

        if (pushConstantStages != 0)
        {
            m_PushConstantRange.stageFlags = pushConstantStages;
            m_PushConstantRange.offset = pushConstantBegin;
            m_PushConstantRange.size = pushConstantEnd - pushConstantBegin;
        }

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = static_cast<uint32_t>(m_DescriptorSetLayouts.size());
        layoutInfo.pSetLayouts = m_DescriptorSetLayouts.data();

        if (m_PushConstantRange.size > 0)
        {
            layoutInfo.pushConstantRangeCount = 1;
            layoutInfo.pPushConstantRanges = &m_PushConstantRange;
        }
        else
        {
            layoutInfo.pushConstantRangeCount = 0;
            layoutInfo.pPushConstantRanges = nullptr;
        }

        CheckResult(vkCreatePipelineLayout(m_Context->GetDevice(), &layoutInfo, nullptr, &m_Layout), "Failed to create pipeline layout");
    }

    void Pipeline::Bind(VkCommandBuffer commandBuffer) const
    {
        vkCmdBindPipeline(commandBuffer, GetBindPoint(), m_Pipeline);
    }

    void Pipeline::BindDescriptorSets(VkCommandBuffer commandBuffer, uint32_t firstSet, const std::vector<VkDescriptorSet>& sets) const
    {
        if (sets.empty())
        {
            return;
        }

        vkCmdBindDescriptorSets(
            commandBuffer,
            GetBindPoint(),
            m_Layout,
            firstSet,
            static_cast<uint32_t>(sets.size()),
            sets.data(),
            0,
            nullptr);
    }

    void Pipeline::PushConstants(VkCommandBuffer commandBuffer, uint32_t offset, uint32_t size, const void* data) const
    {
        if (m_PushConstantRange.size == 0)
        {
            throw std::runtime_error("Pipeline has no push constant range");
        }

        vkCmdPushConstants(commandBuffer, m_Layout, m_PushConstantRange.stageFlags, offset, size, data);
    }

    bool Pipeline::FindDescriptorType(uint32_t set, uint32_t binding, VkDescriptorType& outType) const
    {
        for (const ShaderBinding& known : m_Bindings)
        {
            if (known.set == set && known.binding == binding)
            {
                outType = known.descriptorType;
                return true;
            }
        }

        return false;
    }

    const ShaderBinding* Pipeline::FindBinding(uint32_t set, uint32_t binding) const
    {
        for (const ShaderBinding& known : m_Bindings)
        {
            if (known.set == set && known.binding == binding)
            {
                return &known;
            }
        }

        return nullptr;
    }

    const ShaderBinding* Pipeline::FindBindingByName(uint32_t set, const std::string& name) const
    {
        for (const ShaderBinding& known : m_Bindings)
        {
            if (known.set == set && known.name == name)
            {
                return &known;
            }
        }

        return nullptr;
    }

    const ShaderBinding* Pipeline::FindUniformBlock(uint32_t set) const
    {
        for (const ShaderBinding& known : m_Bindings)
        {
            if (known.set != set)
            {
                continue;
            }

            if (known.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
            {
                return &known;
            }
        }

        return nullptr;
    }

    bool Pipeline::FindMember(uint32_t set, const std::string& memberName, uint32_t& outBinding, uint32_t& outOffset, uint32_t& outSize) const
    {
        for (const ShaderBinding& known : m_Bindings)
        {
            if (known.set != set)
            {
                continue;
            }

            for (const ShaderBlockMember& member : known.members)
            {
                if (member.name == memberName)
                {
                    outBinding = known.binding;
                    outOffset = member.offset;
                    outSize = member.size;
                    return true;
                }
            }
        }

        return false;
    }

    VkDescriptorSetLayout Pipeline::GetDescriptorSetLayout(uint32_t set) const
    {
        if (set >= m_DescriptorSetLayouts.size())
        {
            return VK_NULL_HANDLE;
        }

        return m_DescriptorSetLayouts[set];
    }

    GraphicsPipeline::GraphicsPipeline(VulkanContext& context, const std::vector<const Shader*>& shaders, const PipelineConfig& config)
        : Pipeline(context)
    {
        if (shaders.empty())
        {
            throw std::runtime_error("A graphics pipeline needs at least one shader stage");
        }

        CreateLayout(shaders);
        CreatePipeline(shaders, config);
    }

    GraphicsPipeline::~GraphicsPipeline()
    {
        Destroy();
    }

    GraphicsPipeline::GraphicsPipeline(GraphicsPipeline&& other) noexcept
        : Pipeline(std::move(other))
    {
    }

    GraphicsPipeline& GraphicsPipeline::operator=(GraphicsPipeline&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        Pipeline::operator=(std::move(other));

        return *this;
    }

    VkPipelineColorBlendAttachmentState GraphicsPipeline::MakeBlendAttachment(BlendMode mode)
    {
        VkPipelineColorBlendAttachmentState attachment{};
        attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;

        if (mode == BlendMode::None)
        {
            attachment.blendEnable = VK_FALSE;
            return attachment;
        }

        attachment.blendEnable = VK_TRUE;
        attachment.colorBlendOp = VK_BLEND_OP_ADD;
        attachment.alphaBlendOp = VK_BLEND_OP_ADD;
        attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;

        if (mode == BlendMode::AlphaBlend)
        {
            attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            return attachment;
        }

        if (mode == BlendMode::Additive)
        {
            attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            return attachment;
        }

        attachment.srcColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR;
        attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        return attachment;
    }

    void GraphicsPipeline::CreatePipeline(const std::vector<const Shader*>& shaders, const PipelineConfig& config)
    {
        std::vector<VkPipelineShaderStageCreateInfo> stages;

        for (const Shader* shader : shaders)
        {
            stages.push_back(shader->GetStageCreateInfo());
        }

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount = static_cast<uint32_t>(config.vertexInput.bindings.size());
        vertexInput.pVertexBindingDescriptions = config.vertexInput.bindings.data();
        vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(config.vertexInput.attributes.size());
        vertexInput.pVertexAttributeDescriptions = config.vertexInput.attributes.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = config.topology;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterization{};
        rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterization.depthClampEnable = VK_FALSE;
        rasterization.rasterizerDiscardEnable = VK_FALSE;
        rasterization.polygonMode = config.polygonMode;
        rasterization.cullMode = config.cullMode;
        rasterization.frontFace = config.frontFace;
        rasterization.depthBiasEnable = VK_FALSE;
        rasterization.lineWidth = config.lineWidth;

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = config.sampleCount;
        multisample.sampleShadingEnable = VK_FALSE;
        multisample.minSampleShading = 1.0f;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_FALSE;
        depthStencil.depthWriteEnable = VK_FALSE;

        if (config.depthTestEnable)
        {
            depthStencil.depthTestEnable = VK_TRUE;
        }

        if (config.depthWriteEnable)
        {
            depthStencil.depthWriteEnable = VK_TRUE;
        }

        depthStencil.depthCompareOp = config.depthCompareOp;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_FALSE;

        if (config.stencilTestEnable)
        {
            depthStencil.stencilTestEnable = VK_TRUE;
            depthStencil.front = config.stencilFront;
            depthStencil.back = config.stencilBack;
        }

        std::vector<VkPipelineColorBlendAttachmentState> blendAttachments;

        for (size_t i = 0; i < config.colorAttachmentFormats.size(); i++)
        {
            blendAttachments.push_back(MakeBlendAttachment(config.blendMode));
        }

        VkPipelineColorBlendStateCreateInfo colorBlend{};
        colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlend.logicOpEnable = VK_FALSE;
        colorBlend.logicOp = VK_LOGIC_OP_COPY;
        colorBlend.attachmentCount = static_cast<uint32_t>(blendAttachments.size());
        colorBlend.pAttachments = blendAttachments.data();

        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(config.dynamicStates.size());
        dynamicState.pDynamicStates = config.dynamicStates.data();

        VkPipelineRenderingCreateInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        renderingInfo.colorAttachmentCount = static_cast<uint32_t>(config.colorAttachmentFormats.size());
        renderingInfo.pColorAttachmentFormats = config.colorAttachmentFormats.data();
        renderingInfo.depthAttachmentFormat = config.depthAttachmentFormat;
        renderingInfo.stencilAttachmentFormat = config.stencilAttachmentFormat;

        VkGraphicsPipelineCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        createInfo.pNext = &renderingInfo;
        createInfo.stageCount = static_cast<uint32_t>(stages.size());
        createInfo.pStages = stages.data();
        createInfo.pVertexInputState = &vertexInput;
        createInfo.pInputAssemblyState = &inputAssembly;
        createInfo.pViewportState = &viewportState;
        createInfo.pRasterizationState = &rasterization;
        createInfo.pMultisampleState = &multisample;
        createInfo.pDepthStencilState = &depthStencil;
        createInfo.pColorBlendState = &colorBlend;
        createInfo.pDynamicState = &dynamicState;
        createInfo.layout = m_Layout;
        createInfo.renderPass = VK_NULL_HANDLE;
        createInfo.subpass = 0;

        CheckResult(
            vkCreateGraphicsPipelines(m_Context->GetDevice(), VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_Pipeline),
            "Failed to create graphics pipeline");
    }

} // namespace Core