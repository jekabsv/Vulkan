#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "Font.h"
#include "Material.h"
#include "Mesh.h"
#include "Pipeline.h"
#include "Shader.h"
#include "Texture.h"

namespace Core
{

    class VulkanContext;
    class Renderer;

    // TODO: uncomment once a Sound class exists.
    // class Sound;

    // Owns every asset by name (Mesh, Texture, Font, Material, Shader, GraphicsPipeline, ...) and
    // hands out references to callers for the rest of the asset's lifetime. Assets are constructed
    // elsewhere - directly by the caller for Set (e.g. `Mesh::CreateCube(context)`), or by
    // AssetManager itself for Load - then handed to AssetManager, which becomes the sole owner;
    // everyone else only holds references/pointers into it, the same non-owning-pointer convention
    // already used throughout Core (e.g. Material only holds a `const Texture*`, never owning it).
    //
    // Getters return `const T&` and throw if the name isn't found, matching how the rest of Core
    // reports misuse (Buffer::SetData, Material::WriteRaw, etc. all throw rather than return a null/
    // empty result). Use HasX() first if a missing asset is an expected, non-exceptional case.
    class AssetManager
    {
    public:
        AssetManager(VulkanContext& context, Renderer& renderer);

        AssetManager(const AssetManager&) = delete;
        AssetManager& operator=(const AssetManager&) = delete;
        AssetManager(AssetManager&&) = delete;
        AssetManager& operator=(AssetManager&&) = delete;

        // --- Mesh ---
        const Mesh& GetMesh(const std::string& name) const;
        Mesh& SetMesh(const std::string& name, Mesh&& mesh);
        // Mesh& LoadMesh(const std::string& name, const std::string& path);
        // TODO: requires a model loader (e.g. tinyobjloader/cgltf) - not yet added, so this can't
        // be implemented yet. Uncomment once one is vendored in, following the same pattern as
        // LoadFont below (decode file -> build Vertex/index data -> SetMesh).
        bool HasMesh(const std::string& name) const;
        bool RemoveMesh(const std::string& name);

        // --- Texture ---
        const Texture& GetTexture(const std::string& name) const;
        Texture& SetTexture(const std::string& name, Texture&& texture);
        // Texture& LoadTexture(const std::string& name, const std::string& path);
        // TODO: requires an image decoder (e.g. stb_image.h) - not yet added. Uncomment once one
        // is vendored in (decode file -> pixels -> SetTexture).
        bool HasTexture(const std::string& name) const;
        bool RemoveTexture(const std::string& name);

        // --- Font ---
        const Font& GetFont(const std::string& name) const;
        Font& SetFont(const std::string& name, Font&& font);
        // Bakes `path` (a .ttf/.otf file) into a bitmap glyph atlas via stb_truetype and builds a
        // Font from it. The atlas Texture is owned by AssetManager too, stored under `name + "_Atlas"`.
        Font& LoadFont(const std::string& name, const GraphicsPipeline& pipeline, const std::string& path, float pixelHeight);
        bool HasFont(const std::string& name) const;
        bool RemoveFont(const std::string& name);

        // --- Material ---
        const Material& GetMaterial(const std::string& name) const;
        Material& SetMaterial(const std::string& name, Material&& material);
        // No LoadMaterial: there's no serialized material format, only construct-from-code.
        bool HasMaterial(const std::string& name) const;
        bool RemoveMaterial(const std::string& name);

        // --- Shader ---
        const Shader& GetShader(const std::string& name) const;
        Shader& SetShader(const std::string& name, Shader&& shader);
        Shader& LoadShader(const std::string& name, const std::string& path);
        bool HasShader(const std::string& name) const;
        bool RemoveShader(const std::string& name);

        // --- GraphicsPipeline ---
        // No LoadPipeline: pipelines are built from Shaders + a PipelineConfig; there's no
        // serialized pipeline format, only construct-from-code.
        const GraphicsPipeline& GetPipeline(const std::string& name) const;
        GraphicsPipeline& SetPipeline(const std::string& name, GraphicsPipeline&& pipeline);
        bool HasPipeline(const std::string& name) const;
        bool RemovePipeline(const std::string& name);

        // --- Sound ---
        // TODO: uncomment once a Sound class exists. Deliberately left unimplemented (no .cpp
        // body at all) until then - only the intended API shape is sketched here.
        // const Sound& GetSound(const std::string& name) const;
        // Sound& SetSound(const std::string& name, Sound&& sound);
        // Sound& LoadSound(const std::string& name, const std::string& path);
        // bool HasSound(const std::string& name) const;
        // bool RemoveSound(const std::string& name);

    private:
        VulkanContext* m_Context{ nullptr };
        Renderer* m_Renderer{ nullptr };

        std::unordered_map<std::string, Mesh> m_Meshes;
        std::unordered_map<std::string, Texture> m_Textures;
        std::unordered_map<std::string, Font> m_Fonts;
        std::unordered_map<std::string, Material> m_Materials;
        std::unordered_map<std::string, Shader> m_Shaders;
        std::unordered_map<std::string, GraphicsPipeline> m_Pipelines;

        // std::unordered_map<std::string, Sound> m_Sounds; // TODO: uncomment once Sound exists.
    };

} // namespace Core
