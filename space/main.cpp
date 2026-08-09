#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "VulkanAbstraction.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <vector>
#include <string>

namespace
{
    std::vector<uint8_t> MakeCheckerPixels(uint32_t size, uint32_t squares)
    {
        std::vector<uint8_t> pixels(static_cast<size_t>(size) * size * 4);

        for (uint32_t y = 0; y < size; y++)
        {
            for (uint32_t x = 0; x < size; x++)
            {
                const uint32_t cellX = x * squares / size;
                const uint32_t cellY = y * squares / size;

                uint8_t value = 60;

                if ((cellX + cellY) % 2 == 0)
                {
                    value = 220;
                }

                const size_t index = (static_cast<size_t>(y) * size + x) * 4;
                pixels[index + 0] = value;
                pixels[index + 1] = value;
                pixels[index + 2] = value;
                pixels[index + 3] = 255;
            }
        }

        return pixels;
    }
}

int main()
{
    try
    {
        Core::Window window(1280, 720, "3D Engine");
        Core::VulkanContext context(window);
        Core::Renderer renderer(context, window);
        Core::AssetManager assets(context, renderer);

        // --- Shaders ---
        assets.LoadShader("triangle_vert", "vert.spv");
        assets.LoadShader("triangle_frag", "frag.spv");
        assets.LoadShader("triangle_instanced_vert", "triangle_instanced_vert.spv");
        assets.LoadShader("text_instanced_vert", "text_instanced_vert.spv");
        assets.LoadShader("text_instanced_frag", "text_instanced_frag.spv");

        // --- Pipelines ---
        Core::PipelineConfig pipelineConfig;
        pipelineConfig.vertexInput = Core::Vertex::GetLayout();
        pipelineConfig.cullMode = VK_CULL_MODE_BACK_BIT;
        pipelineConfig.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        pipelineConfig.depthTestEnable = true;
        pipelineConfig.depthWriteEnable = true;
        pipelineConfig.colorAttachmentFormats.push_back(renderer.GetColorFormat());
        pipelineConfig.depthAttachmentFormat = renderer.GetDepthFormat();

        std::vector<const Core::Shader*> pbrShaders;
        pbrShaders.push_back(&assets.GetShader("triangle_vert"));
        pbrShaders.push_back(&assets.GetShader("triangle_frag"));

        assets.SetPipeline("pbr", Core::GraphicsPipeline(context, pbrShaders, pipelineConfig));

        // Mesh batching: reads the model matrix from a per-instance vertex buffer instead of a
        // push constant, so it shares triangle.frag/frag.spv with "pbr" above unchanged.
        Core::PipelineConfig instancedPipelineConfig;
        instancedPipelineConfig.vertexInput = Core::Vertex::GetLayout();
        Core::InstanceBatch::AppendInstanceLayout(instancedPipelineConfig.vertexInput, 1, 3);
        instancedPipelineConfig.cullMode = VK_CULL_MODE_BACK_BIT;
        instancedPipelineConfig.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        instancedPipelineConfig.depthTestEnable = true;
        instancedPipelineConfig.depthWriteEnable = true;
        instancedPipelineConfig.colorAttachmentFormats.push_back(renderer.GetColorFormat());
        instancedPipelineConfig.depthAttachmentFormat = renderer.GetDepthFormat();

        std::vector<const Core::Shader*> instancedShaders;
        instancedShaders.push_back(&assets.GetShader("triangle_instanced_vert"));
        instancedShaders.push_back(&assets.GetShader("triangle_frag"));

        assets.SetPipeline("instanced", Core::GraphicsPipeline(context, instancedShaders, instancedPipelineConfig));

        // Instanced text: per-letter transform, atlas UV rect and color all come from per-instance
        // vertex data (see Font::AppendInstanceLayout), not push constants/Material.
        Core::PipelineConfig textPipelineConfig;
        textPipelineConfig.vertexInput = Core::Vertex::GetLayout();
        Core::Font::AppendInstanceLayout(textPipelineConfig.vertexInput, 1, 3);
        textPipelineConfig.cullMode = VK_CULL_MODE_NONE;
        textPipelineConfig.depthTestEnable = false;
        textPipelineConfig.depthWriteEnable = false;
        textPipelineConfig.blendMode = Core::BlendMode::AlphaBlend;
        textPipelineConfig.colorAttachmentFormats.push_back(renderer.GetColorFormat());
        textPipelineConfig.depthAttachmentFormat = renderer.GetDepthFormat();

        std::vector<const Core::Shader*> textShaders;
        textShaders.push_back(&assets.GetShader("text_instanced_vert"));
        textShaders.push_back(&assets.GetShader("text_instanced_frag"));

        assets.SetPipeline("text", Core::GraphicsPipeline(context, textShaders, textPipelineConfig));

        // --- Mesh ---
        assets.SetMesh("cube", Core::Mesh::CreateCube(context));

        // --- Texture ---
        const uint32_t textureSize = 256;
        const std::vector<uint8_t> pixels = MakeCheckerPixels(textureSize, 8);

        Core::TextureConfig textureConfig;
        textureConfig.width = textureSize;
        textureConfig.height = textureSize;
        textureConfig.format = VK_FORMAT_R8G8B8A8_SRGB;
        textureConfig.generateMipmaps = true;

        assets.SetTexture("wood", Core::Texture(context, textureConfig, pixels.data(), static_cast<VkDeviceSize>(pixels.size())));

        // --- Materials ---
        Core::Material& woodMaterial = assets.SetMaterial("wood", renderer.CreateMaterial(assets.GetPipeline("pbr")));
        woodMaterial.SetVec3("u_AlbedoColor", glm::vec3(1.0f, 0.8f, 0.6f));
        woodMaterial.SetFloat("u_Roughness", 1.0f);
        woodMaterial.SetFloat("u_Metallic", 0.0f);
        woodMaterial.SetVec2("u_Tiling", glm::vec2(1.0f, 1.0f));
        woodMaterial.SetTexture("u_AlbedoMap", assets.GetTexture("wood"));

        Core::Material& instancedWoodMaterial = assets.SetMaterial("wood_instanced", renderer.CreateMaterial(assets.GetPipeline("instanced")));
        instancedWoodMaterial.SetVec3("u_AlbedoColor", glm::vec3(1.0f, 0.8f, 0.6f));
        instancedWoodMaterial.SetFloat("u_Roughness", 1.0f);
        instancedWoodMaterial.SetFloat("u_Metallic", 0.0f);
        instancedWoodMaterial.SetVec2("u_Tiling", glm::vec2(1.0f, 1.0f));
        instancedWoodMaterial.SetTexture("u_AlbedoMap", assets.GetTexture("wood"));


        constexpr uint32_t particleCount = 0;
        constexpr uint32_t particleLocalSizeX = 128;
        constexpr float particleScale = 0.05f;

        struct ParticlePushConstants
        {
            float time;
            uint32_t count;
        };

        bool particlesEnabled = false;
        std::optional<Core::Buffer> particleBuffer;
        std::optional<Core::Buffer> particleReadback;
        std::optional<Core::DescriptorManager> particleDescriptors;
        VkDescriptorSet particleDescriptorSet = VK_NULL_HANDLE;
        Core::Material* particleMaterial = nullptr;

        try
        {
            assets.LoadShader("particles_comp", "particles.spv");
            assets.SetComputePipeline("particles", Core::ComputePipeline(context, assets.GetShader("particles_comp")));

            particleBuffer.emplace(context, sizeof(glm::vec4) * particleCount, Core::BufferType::Storage, Core::MemoryUsage::DeviceLocal);
            particleReadback.emplace(context, particleBuffer->GetSize(), Core::BufferType::Staging, Core::MemoryUsage::HostReadback);
            particleDescriptors.emplace(context, 1);

            Core::DescriptorWriter particleWriter = particleDescriptors->Begin(assets.GetComputePipeline("particles"), 0);
            particleWriter.WriteBuffer(0, *particleBuffer);
            particleDescriptorSet = particleWriter.Build();

            particleMaterial = &assets.SetMaterial("particles", renderer.CreateMaterial(assets.GetPipeline("instanced")));
            particleMaterial->SetVec3("u_AlbedoColor", glm::vec3(1.0f, 0.35f, 0.15f));
            particleMaterial->SetFloat("u_Roughness", 1.0f);
            particleMaterial->SetFloat("u_Metallic", 0.0f);
            particleMaterial->SetVec2("u_Tiling", glm::vec2(1.0f, 1.0f));
            particleMaterial->SetTexture("u_AlbedoMap", assets.GetTexture("wood"));

            particlesEnabled = true;
        }
        catch (const std::exception& error)
        {
            std::cerr << "[Particles] Skipped (compile particles.comp to particles.spv with glslc to enable): "
                << error.what() << std::endl;
        }

        // --- Font ---
        const float fontPixelHeight = 48.0f;
        Core::Font& textFont = assets.LoadFont("main", assets.GetPipeline("text"), "font.ttf", fontPixelHeight);


        const float textScale = 1.0f / fontPixelHeight;

        Core::Camera mainCamera;
        mainCamera.SetPosition(glm::vec3(2.4f, 1.8f, 2.4f));
        mainCamera.LookAt(glm::vec3(0.0f));

        const auto startTime = std::chrono::high_resolution_clock::now();

        float prevFrameTime = 0.0f;
        glm::vec2 lastMousePos = window.GetMousePos();

        while (!window.ShouldClose())
        {
            auto FrameStart = std::chrono::high_resolution_clock::now();

            window.PollEvents();

            if (window.IsKeyPressed(Core::KeyCode::Escape))
            {
                window.Close();
                continue;
            }
            if (window.IsKeyDown(Core::KeyCode::W))
            {
                mainCamera.MoveForward(0.1f);
            }
            if (window.IsKeyDown(Core::KeyCode::S))
            {
                mainCamera.MoveForward(-0.1f);
            }
            if (window.IsKeyDown(Core::KeyCode::A))
            {
                mainCamera.MoveRight(-0.1f);
            }
            if (window.IsKeyDown(Core::KeyCode::D))
            {
                mainCamera.MoveRight(0.1f);
            }
            if (window.IsKeyDown(Core::KeyCode::Space))
            {
                mainCamera.MoveWorld(glm::vec3(0.0f, 0.1f, 0.0f));
            }
            if (window.IsKeyDown(Core::KeyCode::Q))
            {
                mainCamera.MoveWorld(glm::vec3(0.0f, -0.1f, 0.0f));
            }

            const glm::vec2 mousePos = window.GetMousePos();
            const glm::vec2 mouseDelta = mousePos - lastMousePos;
            lastMousePos = mousePos;

            if (window.IsMouseButtonDown(Core::MouseButton::Right))
            {
                constexpr float mouseSensitivity = 0.0025f;
                mainCamera.Yaw(-mouseDelta.x * mouseSensitivity);
                mainCamera.Pitch(-mouseDelta.y * mouseSensitivity);
            }

            auto now = std::chrono::high_resolution_clock::now();

            float elapsed = std::chrono::duration<float>(now - startTime).count();

            mainCamera.SetPerspective(glm::radians(50.0f), renderer.GetAspectRatio(), 0.1f, 100.0f);

            const glm::mat4 transformMatrix = glm::rotate(glm::mat4(1.0f), elapsed * 0.6f, glm::vec3(0.0f, 1.0f, 0.0f));

            std::vector<glm::vec4> particlePositions;

            if (particlesEnabled)
            {
                ParticlePushConstants particlePushConstants{ elapsed, particleCount };

                VkCommandBuffer computeCommandBuffer = context.BeginSingleTimeCommands();

                const Core::ComputePipeline& particlePipeline = assets.GetComputePipeline("particles");
                particlePipeline.Bind(computeCommandBuffer);
                particlePipeline.BindDescriptorSets(computeCommandBuffer, 0, { particleDescriptorSet });
                particlePipeline.PushConstants(computeCommandBuffer, 0, sizeof(particlePushConstants), &particlePushConstants);
                particlePipeline.Dispatch(computeCommandBuffer, (particleCount + particleLocalSizeX - 1) / particleLocalSizeX);

                Core::ComputePipeline::BufferBarrier(
                    computeCommandBuffer,
                    *particleBuffer,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

                VkBufferCopy copyRegion{};
                copyRegion.size = particleBuffer->GetSize();
                vkCmdCopyBuffer(computeCommandBuffer, particleBuffer->GetHandle(), particleReadback->GetHandle(), 1, &copyRegion);

                context.EndSingleTimeCommands(computeCommandBuffer);

                particleReadback->Invalidate(particleReadback->GetSize(), 0);

                const glm::vec4* positions = static_cast<const glm::vec4*>(particleReadback->GetMappedData());
                particlePositions.assign(positions, positions + particleCount);
            }

            if (renderer.BeginFrame())
            {
                renderer.BeginRenderPass(glm::vec4(0.1f, 0.1f, 0.12f, 1.0f));

                renderer.SetCamera(mainCamera);
                renderer.DrawMesh(assets.GetMesh("cube"), woodMaterial, transformMatrix);


                renderer.BeginBatch();


                constexpr int a = 200;

                for (int gridX = -1 * a; gridX <= a; gridX++)
                {
                    for (int gridZ = -1 * a; gridZ <= a; gridZ++)
                    {
                        const glm::vec3 offset(static_cast<float>(gridX) * 1.5f, 0.0f, static_cast<float>(gridZ) * 1.5f);
                        const glm::mat4 instanceTransform = glm::translate(glm::mat4(1.0f), offset) * transformMatrix;
                        renderer.Submit(assets.GetMesh("cube"), instancedWoodMaterial, instanceTransform);
                    }
                }

                if (particlesEnabled)
                {
                    for (const glm::vec4& position : particlePositions)
                    {
                        const glm::mat4 particleTransform = glm::scale(
                            glm::translate(glm::mat4(1.0f), glm::vec3(position)),
                            glm::vec3(particleScale));

                        renderer.Submit(assets.GetMesh("cube"), *particleMaterial, particleTransform);
                    }
                }

                renderer.SubmitText(textFont, "Hello " + std::to_string(4 * a * a) + " cubes", glm::vec3(-2.0f, 2.5f, 0.0f), textScale, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
                renderer.SubmitText(textFont, "FPS: " + std::to_string(1.0f / std::max(0.0001f, prevFrameTime)) + " FPS", glm::vec3(-2.0f, 1.5f, 0.0f), textScale, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));

                renderer.FlushBatch();

                renderer.EndRenderPass();
                renderer.EndFrame();

                now = std::chrono::high_resolution_clock::now();
                prevFrameTime = std::chrono::duration<float>(now - FrameStart).count();

            }
        }

        context.WaitIdle();
    }
    catch (const std::exception& error)
    {
        std::cerr << "Fatal: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
