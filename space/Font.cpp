#include "Font.h"

#include "Pipeline.h"
#include "Renderer.h"
#include "Texture.h"

#include <cstddef>
#include <glm/gtc/matrix_transform.hpp>

namespace Core
{

    Font::Font(VulkanContext& context, Renderer& renderer, const GraphicsPipeline& pipeline, Texture& atlas)
        : m_Context(&context)
        , m_Atlas(&atlas)
        , m_QuadMesh(Mesh::CreateQuad(context))
        , m_Material(renderer.CreateMaterial(pipeline))
    {
        m_Material.SetTexture("u_Atlas", atlas);
    }

    void Font::SetGlyph(char character, const Glyph& glyph)
    {
        m_Glyphs[character] = glyph;
    }

    const Glyph* Font::FindGlyph(char character) const
    {
        auto it = m_Glyphs.find(character);
        return it != m_Glyphs.end() ? &it->second : nullptr;
    }

    glm::mat4 Font::ComputeGlyphTransform(const glm::vec3& pen, const Glyph& glyph, float scale)
    {
        const glm::vec3 glyphCenter = pen + glm::vec3(
            glyph.bearing.x + glyph.size.x * 0.5f,
            glyph.bearing.y - glyph.size.y * 0.5f,
            0.0f) * scale;

        return glm::translate(glm::mat4(1.0f), glyphCenter) *
            glm::scale(glm::mat4(1.0f), glm::vec3(glyph.size * scale, 1.0f));
    }

    void Font::DrawText(Renderer& renderer, const std::string& text, const glm::vec3& origin, float scale, const glm::vec4& color)
    {
        m_Material.SetVec4("u_Color", color);

        glm::vec3 pen = origin;

        for (char character : text)
        {
            const Glyph* glyph = FindGlyph(character);

            if (glyph == nullptr)
            {
                pen.x += m_DefaultAdvance * scale;
                continue;
            }

            TextPushConstants pushConstants;
            pushConstants.transform = ComputeGlyphTransform(pen, *glyph, scale);
            pushConstants.uvOffset = glyph->uvOffset;
            pushConstants.uvSize = glyph->uvSize;

            renderer.DrawMesh(m_QuadMesh, m_Material, &pushConstants, sizeof(TextPushConstants));

            pen.x += glyph->advance * scale;
        }
    }

    Font Font::CreateMonospaceGrid(
        VulkanContext& context,
        Renderer& renderer,
        const GraphicsPipeline& pipeline,
        Texture& atlas,
        uint32_t columns,
        uint32_t rows,
        const std::string& characters,
        const glm::vec2& glyphSize)
    {
        Font font(context, renderer, pipeline, atlas);
        font.SetDefaultAdvance(glyphSize.x);

        const glm::vec2 cellUV(1.0f / static_cast<float>(columns), 1.0f / static_cast<float>(rows));

        for (size_t i = 0; i < characters.size(); i++)
        {
            const uint32_t column = static_cast<uint32_t>(i) % columns;
            const uint32_t row = static_cast<uint32_t>(i) / columns;

            if (row >= rows)
            {
                break;
            }

            Glyph glyph;
            glyph.uvOffset = glm::vec2(column * cellUV.x, row * cellUV.y);
            glyph.uvSize = cellUV;
            glyph.size = glyphSize;
            glyph.bearing = glm::vec2(0.0f, 0.0f);
            glyph.advance = glyphSize.x;

            font.SetGlyph(characters[i], glyph);
        }

        return font;
    }

    void Font::AppendInstanceLayout(VertexInputLayout& layout, uint32_t binding, uint32_t startLocation)
    {
        layout.AddBinding(binding, sizeof(GlyphInstance), VK_VERTEX_INPUT_RATE_INSTANCE);

        for (uint32_t column = 0; column < 4; column++)
        {
            layout.AddAttribute(startLocation + column, binding, VK_FORMAT_R32G32B32A32_SFLOAT,
                static_cast<uint32_t>(offsetof(GlyphInstance, transform) + sizeof(glm::vec4) * column));
        }

        layout.AddAttribute(startLocation + 4, binding, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(GlyphInstance, uvOffset)));
        layout.AddAttribute(startLocation + 5, binding, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(GlyphInstance, uvSize)));
        layout.AddAttribute(startLocation + 6, binding, VK_FORMAT_R32G32B32A32_SFLOAT, static_cast<uint32_t>(offsetof(GlyphInstance, color)));
    }

} // namespace Core
