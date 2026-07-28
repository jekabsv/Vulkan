#include "Shader.h"

#include "VulkanContext.h"

#include "spirv_reflect.h"

#include <fstream>
#include <stdexcept>
#include <utility>

namespace Core
{

    namespace
    {
        const uint32_t SpirvMagicNumber = 0x07230203;

        void CheckResult(VkResult result, const char* message)
        {
            if (result != VK_SUCCESS)
            {
                throw std::runtime_error(message);
            }
        }

        void CheckReflectResult(SpvReflectResult result, const char* message)
        {
            if (result != SPV_REFLECT_RESULT_SUCCESS)
            {
                throw std::runtime_error(message);
            }
        }
    }

    Shader::Shader(VulkanContext& context, const std::string& filePath)
        : m_Context(&context)
        , m_SourcePath(filePath)
    {
        const std::vector<uint32_t> spirv = ReadSpirvFile(filePath);
        Initialize(spirv);
    }

    Shader::Shader(VulkanContext& context, const std::vector<uint32_t>& spirv)
        : m_Context(&context)
    {
        Initialize(spirv);
    }

    Shader::~Shader()
    {
        Destroy();
    }

    Shader::Shader(Shader&& other) noexcept
        : m_Context(other.m_Context)
        , m_Module(other.m_Module)
        , m_Stage(other.m_Stage)
        , m_EntryPoint(std::move(other.m_EntryPoint))
        , m_SourcePath(std::move(other.m_SourcePath))
        , m_Reflection(std::move(other.m_Reflection))
    {
        other.m_Context = nullptr;
        other.m_Module = VK_NULL_HANDLE;
    }

    Shader& Shader::operator=(Shader&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        Destroy();

        m_Context = other.m_Context;
        m_Module = other.m_Module;
        m_Stage = other.m_Stage;
        m_EntryPoint = std::move(other.m_EntryPoint);
        m_SourcePath = std::move(other.m_SourcePath);
        m_Reflection = std::move(other.m_Reflection);

        other.m_Context = nullptr;
        other.m_Module = VK_NULL_HANDLE;

        return *this;
    }

    void Shader::Initialize(const std::vector<uint32_t>& spirv)
    {
        if (spirv.empty())
        {
            throw std::runtime_error("SPIR-V module is empty");
        }

        if (spirv[0] != SpirvMagicNumber)
        {
            throw std::runtime_error("Data is not a valid SPIR-V module");
        }

        Reflect(spirv);
        CreateModule(spirv);
    }

    void Shader::Destroy()
    {
        if (m_Context == nullptr)
        {
            return;
        }

        if (m_Module != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(m_Context->GetDevice(), m_Module, nullptr);
            m_Module = VK_NULL_HANDLE;
        }
    }

    std::vector<uint32_t> Shader::ReadSpirvFile(const std::string& filePath)
    {
        std::ifstream file(filePath, std::ios::binary | std::ios::ate);

        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open shader file: " + filePath);
        }

        const std::streamsize byteSize = file.tellg();

        if (byteSize <= 0)
        {
            throw std::runtime_error("Shader file is empty: " + filePath);
        }

        if (byteSize % 4 != 0)
        {
            throw std::runtime_error("Shader file size is not a multiple of four: " + filePath);
        }

        std::vector<uint32_t> buffer(static_cast<size_t>(byteSize) / 4);

        file.seekg(0);
        file.read(reinterpret_cast<char*>(buffer.data()), byteSize);
        file.close();

        return buffer;
    }

    void Shader::CreateModule(const std::vector<uint32_t>& spirv)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = spirv.size() * sizeof(uint32_t);
        createInfo.pCode = spirv.data();

        CheckResult(vkCreateShaderModule(m_Context->GetDevice(), &createInfo, nullptr, &m_Module), "Failed to create shader module");
    }

    void Shader::Reflect(const std::vector<uint32_t>& spirv)
    {
        SpvReflectShaderModule module{};

        CheckReflectResult(
            spvReflectCreateShaderModule(spirv.size() * sizeof(uint32_t), spirv.data(), &module),
            "Failed to reflect SPIR-V module");

        m_Stage = static_cast<VkShaderStageFlagBits>(module.shader_stage);

        if (module.entry_point_name != nullptr)
        {
            m_EntryPoint = module.entry_point_name;
        }

        uint32_t setCount = 0;
        CheckReflectResult(spvReflectEnumerateDescriptorSets(&module, &setCount, nullptr), "Failed to enumerate descriptor sets");

        std::vector<SpvReflectDescriptorSet*> sets(setCount);
        CheckReflectResult(spvReflectEnumerateDescriptorSets(&module, &setCount, sets.data()), "Failed to enumerate descriptor sets");

        for (const SpvReflectDescriptorSet* set : sets)
        {
            for (uint32_t i = 0; i < set->binding_count; i++)
            {
                const SpvReflectDescriptorBinding* reflected = set->bindings[i];

                ShaderBinding binding;

                if (reflected->name != nullptr && reflected->name[0] != '\0')
                {
                    binding.name = reflected->name;
                }
                else if (reflected->type_description != nullptr && reflected->type_description->type_name != nullptr)
                {
                    binding.name = reflected->type_description->type_name;
                }

                binding.set = set->set;
                binding.binding = reflected->binding;
                binding.descriptorType = static_cast<VkDescriptorType>(reflected->descriptor_type);
                binding.stageFlags = static_cast<VkShaderStageFlags>(m_Stage);

                uint32_t elementCount = 1;

                for (uint32_t dimension = 0; dimension < reflected->array.dims_count; dimension++)
                {
                    elementCount *= reflected->array.dims[dimension];
                }

                binding.count = elementCount;
                binding.blockSize = reflected->block.size;

                for (uint32_t member = 0; member < reflected->block.member_count; member++)
                {
                    const SpvReflectBlockVariable& reflectedMember = reflected->block.members[member];

                    ShaderBlockMember blockMember;

                    if (reflectedMember.name != nullptr)
                    {
                        blockMember.name = reflectedMember.name;
                    }

                    blockMember.offset = reflectedMember.offset;
                    blockMember.size = reflectedMember.size;

                    binding.members.push_back(blockMember);
                }

                m_Reflection.bindings.push_back(binding);
            }
        }

        uint32_t blockCount = 0;
        CheckReflectResult(spvReflectEnumeratePushConstantBlocks(&module, &blockCount, nullptr), "Failed to enumerate push constants");

        std::vector<SpvReflectBlockVariable*> blocks(blockCount);
        CheckReflectResult(spvReflectEnumeratePushConstantBlocks(&module, &blockCount, blocks.data()), "Failed to enumerate push constants");

        for (const SpvReflectBlockVariable* block : blocks)
        {
            const uint32_t end = block->offset + block->size;

            if (!m_Reflection.hasPushConstants)
            {
                m_Reflection.hasPushConstants = true;
                m_Reflection.pushConstantOffset = block->offset;
                m_Reflection.pushConstantSize = block->size;
                continue;
            }

            if (block->offset < m_Reflection.pushConstantOffset)
            {
                m_Reflection.pushConstantOffset = block->offset;
            }

            if (end > m_Reflection.pushConstantOffset + m_Reflection.pushConstantSize)
            {
                m_Reflection.pushConstantSize = end - m_Reflection.pushConstantOffset;
            }
        }

        spvReflectDestroyShaderModule(&module);
    }

    VkPipelineShaderStageCreateInfo Shader::GetStageCreateInfo() const
    {
        VkPipelineShaderStageCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        createInfo.stage = m_Stage;
        createInfo.module = m_Module;
        createInfo.pName = m_EntryPoint.c_str();
        return createInfo;
    }

} // namespace Core