#include "AssetManager.h"

#include "Renderer.h"
#include "VulkanContext.h"

#include "stb_truetype.h"

#include <fstream>
#include <stdexcept>

namespace Core
{

    namespace
    {
        constexpr int kFontFirstChar = 32;
        constexpr int kFontCharCount = 95; // ASCII 32..126 (printable range)
        constexpr int kFontAtlasWidth = 512;
        constexpr int kFontAtlasHeight = 512;

        std::vector<unsigned char> ReadFile(const std::string& path)
        {
            std::ifstream file(path, std::ios::binary | std::ios::ate);

            if (!file.is_open())
            {
                throw std::runtime_error("Failed to open file: " + path);
            }

            const std::streamsize size = file.tellg();
            file.seekg(0);

            std::vector<unsigned char> data(static_cast<size_t>(size));
            file.read(reinterpret_cast<char*>(data.data()), size);

            return data;
        }
    }

    AssetManager::AssetManager(VulkanContext& context, Renderer& renderer)
        : m_Context(&context)
        , m_Renderer(&renderer)
    {
    }

    // --- Mesh ---

    const Mesh& AssetManager::GetMesh(const std::string& name) const
    {
        auto it = m_Meshes.find(name);

        if (it == m_Meshes.end())
        {
            throw std::runtime_error("Mesh asset not found: " + name);
        }

        return it->second;
    }

    Mesh& AssetManager::SetMesh(const std::string& name, Mesh&& mesh)
    {
        return m_Meshes.insert_or_assign(name, std::move(mesh)).first->second;
    }

    bool AssetManager::HasMesh(const std::string& name) const
    {
        return m_Meshes.find(name) != m_Meshes.end();
    }

    bool AssetManager::RemoveMesh(const std::string& name)
    {
        return m_Meshes.erase(name) > 0;
    }

    // --- Texture ---

    const Texture& AssetManager::GetTexture(const std::string& name) const
    {
        auto it = m_Textures.find(name);

        if (it == m_Textures.end())
        {
            throw std::runtime_error("Texture asset not found: " + name);
        }

        return it->second;
    }

    Texture& AssetManager::SetTexture(const std::string& name, Texture&& texture)
    {
        return m_Textures.insert_or_assign(name, std::move(texture)).first->second;
    }

    bool AssetManager::HasTexture(const std::string& name) const
    {
        return m_Textures.find(name) != m_Textures.end();
    }

    bool AssetManager::RemoveTexture(const std::string& name)
    {
        return m_Textures.erase(name) > 0;
    }

    // --- Font ---

    const Font& AssetManager::GetFont(const std::string& name) const
    {
        auto it = m_Fonts.find(name);

        if (it == m_Fonts.end())
        {
            throw std::runtime_error("Font asset not found: " + name);
        }

        return it->second;
    }

    Font& AssetManager::SetFont(const std::string& name, Font&& font)
    {
        return m_Fonts.insert_or_assign(name, std::move(font)).first->second;
    }

    Font& AssetManager::LoadFont(const std::string& name, const GraphicsPipeline& pipeline, const std::string& path, float pixelHeight)
    {
        const std::vector<unsigned char> fontFile = ReadFile(path);

        std::vector<unsigned char> atlasCoverage(static_cast<size_t>(kFontAtlasWidth) * kFontAtlasHeight);
        std::vector<stbtt_bakedchar> bakedChars(kFontCharCount);

        const int bakeResult = stbtt_BakeFontBitmap(
            fontFile.data(), 0, pixelHeight,
            atlasCoverage.data(), kFontAtlasWidth, kFontAtlasHeight,
            kFontFirstChar, kFontCharCount, bakedChars.data());

        if (bakeResult <= 0)
        {
            throw std::runtime_error("Failed to bake font atlas (atlas too small for this pixel height?): " + path);
        }

        // Texture only accepts RGBA8 pixel data here, so the single-channel coverage bitmap is
        // replicated into RGB and used as-is for A; text_instanced.frag/text.frag only read .r.
        std::vector<uint8_t> atlasPixels(atlasCoverage.size() * 4);

        for (size_t i = 0; i < atlasCoverage.size(); i++)
        {
            const uint8_t value = atlasCoverage[i];
            atlasPixels[i * 4 + 0] = value;
            atlasPixels[i * 4 + 1] = value;
            atlasPixels[i * 4 + 2] = value;
            atlasPixels[i * 4 + 3] = value;
        }

        TextureConfig atlasConfig;
        atlasConfig.width = kFontAtlasWidth;
        atlasConfig.height = kFontAtlasHeight;
        atlasConfig.format = VK_FORMAT_R8G8B8A8_UNORM;
        atlasConfig.generateMipmaps = false;

        Texture& atlasTexture = SetTexture(
            name + "_Atlas",
            Texture(*m_Context, atlasConfig, atlasPixels.data(), static_cast<VkDeviceSize>(atlasPixels.size())));

        Font font(*m_Context, *m_Renderer, pipeline, atlasTexture);
        font.SetDefaultAdvance(pixelHeight * 0.5f);

        const float invWidth = 1.0f / static_cast<float>(kFontAtlasWidth);
        const float invHeight = 1.0f / static_cast<float>(kFontAtlasHeight);

        for (int i = 0; i < kFontCharCount; i++)
        {
            const stbtt_bakedchar& baked = bakedChars[i];

            Glyph glyph;
            glyph.uvOffset = glm::vec2(baked.x0 * invWidth, baked.y0 * invHeight);
            glyph.uvSize = glm::vec2((baked.x1 - baked.x0) * invWidth, (baked.y1 - baked.y0) * invHeight);
            glyph.size = glm::vec2(static_cast<float>(baked.x1 - baked.x0), static_cast<float>(baked.y1 - baked.y0));
            // stb_truetype's yoff is a downward offset from the baseline to the glyph's top edge
            // (+Y down); Glyph/Font::ComputeGlyphTransform assume +Y up, hence the negation. Not
            // verified against a real render yet - flip this sign if glyphs come out vertically
            // offset/upside down.
            glyph.bearing = glm::vec2(baked.xoff, -baked.yoff);
            glyph.advance = baked.xadvance;

            font.SetGlyph(static_cast<char>(kFontFirstChar + i), glyph);
        }

        return SetFont(name, std::move(font));
    }

    bool AssetManager::HasFont(const std::string& name) const
    {
        return m_Fonts.find(name) != m_Fonts.end();
    }

    bool AssetManager::RemoveFont(const std::string& name)
    {
        return m_Fonts.erase(name) > 0;
    }

    // --- Material ---

    const Material& AssetManager::GetMaterial(const std::string& name) const
    {
        auto it = m_Materials.find(name);

        if (it == m_Materials.end())
        {
            throw std::runtime_error("Material asset not found: " + name);
        }

        return it->second;
    }

    Material& AssetManager::SetMaterial(const std::string& name, Material&& material)
    {
        return m_Materials.insert_or_assign(name, std::move(material)).first->second;
    }

    bool AssetManager::HasMaterial(const std::string& name) const
    {
        return m_Materials.find(name) != m_Materials.end();
    }

    bool AssetManager::RemoveMaterial(const std::string& name)
    {
        return m_Materials.erase(name) > 0;
    }

    // --- Shader ---

    const Shader& AssetManager::GetShader(const std::string& name) const
    {
        auto it = m_Shaders.find(name);

        if (it == m_Shaders.end())
        {
            throw std::runtime_error("Shader asset not found: " + name);
        }

        return it->second;
    }

    Shader& AssetManager::SetShader(const std::string& name, Shader&& shader)
    {
        return m_Shaders.insert_or_assign(name, std::move(shader)).first->second;
    }

    Shader& AssetManager::LoadShader(const std::string& name, const std::string& path)
    {
        return SetShader(name, Shader(*m_Context, path));
    }

    bool AssetManager::HasShader(const std::string& name) const
    {
        return m_Shaders.find(name) != m_Shaders.end();
    }

    bool AssetManager::RemoveShader(const std::string& name)
    {
        return m_Shaders.erase(name) > 0;
    }

    // --- GraphicsPipeline ---

    const GraphicsPipeline& AssetManager::GetPipeline(const std::string& name) const
    {
        auto it = m_Pipelines.find(name);

        if (it == m_Pipelines.end())
        {
            throw std::runtime_error("Pipeline asset not found: " + name);
        }

        return it->second;
    }

    GraphicsPipeline& AssetManager::SetPipeline(const std::string& name, GraphicsPipeline&& pipeline)
    {
        return m_Pipelines.insert_or_assign(name, std::move(pipeline)).first->second;
    }

    bool AssetManager::HasPipeline(const std::string& name) const
    {
        return m_Pipelines.find(name) != m_Pipelines.end();
    }

    bool AssetManager::RemovePipeline(const std::string& name)
    {
        return m_Pipelines.erase(name) > 0;
    }

    // --- Sound ---
    // TODO: uncomment once a Sound class exists (see AssetManager.h) - no implementation until then.

} // namespace Core
