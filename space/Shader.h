#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace Core
{

    class VulkanContext;

    struct ShaderBlockMember
    {
        std::string name;
        uint32_t offset{ 0 };
        uint32_t size{ 0 };
    };

    struct ShaderBinding
    {
        std::string name;
        uint32_t set{ 0 };
        uint32_t binding{ 0 };
        uint32_t count{ 1 };
        VkDescriptorType descriptorType{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
        VkShaderStageFlags stageFlags{ 0 };
        uint32_t blockSize{ 0 };
        std::vector<ShaderBlockMember> members;
    };

    struct ShaderReflection
    {
        std::vector<ShaderBinding> bindings;
        bool hasPushConstants{ false };
        uint32_t pushConstantOffset{ 0 };
        uint32_t pushConstantSize{ 0 };
    };

    class Shader
    {
    public:
        Shader(VulkanContext& context, const std::string& filePath);
        Shader(VulkanContext& context, const std::vector<uint32_t>& spirv);
        ~Shader();

        Shader(const Shader&) = delete;
        Shader& operator=(const Shader&) = delete;
        Shader(Shader&& other) noexcept;
        Shader& operator=(Shader&& other) noexcept;

        VkShaderModule GetModule() const { return m_Module; }
        VkShaderStageFlagBits GetStage() const { return m_Stage; }
        const std::string& GetEntryPoint() const { return m_EntryPoint; }
        const std::string& GetSourcePath() const { return m_SourcePath; }
        const ShaderReflection& GetReflection() const { return m_Reflection; }

        VkPipelineShaderStageCreateInfo GetStageCreateInfo() const;

        static std::vector<uint32_t> ReadSpirvFile(const std::string& filePath);

    private:
        void Initialize(const std::vector<uint32_t>& spirv);
        void CreateModule(const std::vector<uint32_t>& spirv);
        void Reflect(const std::vector<uint32_t>& spirv);
        void Destroy();

    private:
        VulkanContext* m_Context{ nullptr };
        VkShaderModule m_Module{ VK_NULL_HANDLE };
        VkShaderStageFlagBits m_Stage{ VK_SHADER_STAGE_VERTEX_BIT };
        std::string m_EntryPoint{ "main" };
        std::string m_SourcePath;
        ShaderReflection m_Reflection;
    };

} // namespace Core