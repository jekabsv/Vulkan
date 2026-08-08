#pragma once

#include <string>
#include <unordered_map>

#include <glm/glm.hpp>

#include "Material.h"
#include "Mesh.h"

namespace Core
{

    class VulkanContext;
    class Renderer;
    class GraphicsPipeline;
    class Texture;
    struct VertexInputLayout;

    struct Glyph
    {
        glm::vec2 uvOffset{ 0.0f, 0.0f };
        glm::vec2 uvSize{ 0.0f, 0.0f };
        glm::vec2 size{ 0.0f, 0.0f };
        glm::vec2 bearing{ 0.0f, 0.0f };
        float advance{ 0.0f };
    };

    struct TextPushConstants
    {
        glm::mat4 transform{ 1.0f };
        glm::vec2 uvOffset{ 0.0f, 0.0f };
        glm::vec2 uvSize{ 0.0f, 0.0f };
    };


    struct GlyphInstance
    {
        glm::mat4 transform{ 1.0f };
        glm::vec2 uvOffset{ 0.0f, 0.0f };
        glm::vec2 uvSize{ 0.0f, 0.0f };
        glm::vec4 color{ 1.0f, 1.0f, 1.0f, 1.0f };
    };

    class Font
    {
    public:
        Font(VulkanContext& context, Renderer& renderer, const GraphicsPipeline& pipeline, Texture& atlas);

        Font(const Font&) = delete;
        Font& operator=(const Font&) = delete;
        Font(Font&& other) noexcept = default;
        Font& operator=(Font&& other) noexcept = default;

        void SetGlyph(char character, const Glyph& glyph);
        const Glyph* FindGlyph(char character) const;
        void SetDefaultAdvance(float advance) { m_DefaultAdvance = advance; }

        void DrawText(Renderer& renderer, const std::string& text, const glm::vec3& origin, float scale, const glm::vec4& color);
        void SubmitText(Renderer& renderer, const std::string& text, const glm::vec3& origin, float scale, const glm::vec4& color);


        static Font CreateMonospaceGrid(
            VulkanContext& context,
            Renderer& renderer,
            const GraphicsPipeline& pipeline,
            Texture& atlas,
            uint32_t columns,
            uint32_t rows,
            const std::string& characters,
            const glm::vec2& glyphSize);

        static void AppendInstanceLayout(VertexInputLayout& layout, uint32_t binding, uint32_t startLocation);

    private:
        static glm::mat4 ComputeGlyphTransform(const glm::vec3& pen, const Glyph& glyph, float scale);

    private:
        VulkanContext* m_Context{ nullptr };
        Texture* m_Atlas{ nullptr };
        Mesh m_QuadMesh;
        Material m_Material;
        std::unordered_map<char, Glyph> m_Glyphs;
        float m_DefaultAdvance{ 0.0f };
    };

} // namespace Core
