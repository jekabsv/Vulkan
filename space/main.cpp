#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "Window.h"
#include "VulkanContext.h"
#include "Renderer.h"
#include "Shader.h"
#include "Pipeline.h"
#include "Mesh.h"
#include "Material.h"
#include "Texture.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

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

int main2()
{
    try
    {
        Core::Window window(1280, 720, "3D Engine");
        Core::VulkanContext context(window);
        Core::Renderer renderer(context, window);

        Core::Shader vertexShader(context, "vert.spv");
        Core::Shader fragmentShader(context, "frag.spv");

        std::vector<const Core::Shader*> shaders;
        shaders.push_back(&vertexShader);
        shaders.push_back(&fragmentShader);

        Core::PipelineConfig pipelineConfig;
        pipelineConfig.vertexInput = Core::Vertex::GetLayout();
        pipelineConfig.cullMode = VK_CULL_MODE_BACK_BIT;
        pipelineConfig.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        pipelineConfig.depthTestEnable = true;
        pipelineConfig.depthWriteEnable = true;
        pipelineConfig.colorAttachmentFormats.push_back(renderer.GetColorFormat());
        pipelineConfig.depthAttachmentFormat = renderer.GetDepthFormat();

        Core::GraphicsPipeline pbrPipeline(context, shaders, pipelineConfig);

        Core::Mesh containerMesh = Core::Mesh::CreateCube(context);

        const uint32_t textureSize = 256;
        const std::vector<uint8_t> pixels = MakeCheckerPixels(textureSize, 8);

        Core::TextureConfig textureConfig;
        textureConfig.width = textureSize;
        textureConfig.height = textureSize;
        textureConfig.format = VK_FORMAT_R8G8B8A8_SRGB;
        textureConfig.generateMipmaps = true;

        Core::Texture woodTexture(context, textureConfig, pixels.data(), static_cast<VkDeviceSize>(pixels.size()));

        Core::Material woodMaterial = renderer.CreateMaterial(pbrPipeline);
        woodMaterial.SetVec3("u_AlbedoColor", glm::vec3(1.0f, 0.8f, 0.6f));
        woodMaterial.SetFloat("u_Roughness", 1.0f);
        woodMaterial.SetFloat("u_Metallic", 0.f);
        woodMaterial.SetVec2("u_Tiling", glm::vec2(1.0f, 1.0f));
        woodMaterial.SetTexture("u_AlbedoMap", woodTexture);

        Core::Camera mainCamera;
        mainCamera.position = glm::vec3(2.4f, 1.8f, 2.4f);

        const auto startTime = std::chrono::high_resolution_clock::now();

        while (!window.ShouldClose())
        {
            window.PollEvents();

            if (window.IsKeyPressed(Core::KeyCode::Escape))
            {
                window.Close();
                continue;
            }
            if (window.IsKeyDown(Core::KeyCode::W))
            {
				mainCamera.position.z -= 0.1f;
            }
			if (window.IsKeyDown(Core::KeyCode::S))
			{
				mainCamera.position.z += 0.1f;
			}


            const auto now = std::chrono::high_resolution_clock::now();
            const float elapsed = std::chrono::duration<float>(now - startTime).count();

            mainCamera.view = glm::lookAt(mainCamera.position, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            mainCamera.projection = glm::perspective(glm::radians(50.0f), renderer.GetAspectRatio(), 0.1f, 100.0f);
            mainCamera.projection[1][1] *= -1.0f;

            const glm::mat4 transformMatrix = glm::rotate(glm::mat4(1.0f), elapsed * 0.6f, glm::vec3(0.0f, 1.0f, 0.0f));

            if (renderer.BeginFrame())
            {
                renderer.BeginRenderPass(glm::vec4(0.1f, 0.1f, 0.12f, 1.0f));

                renderer.SetCamera(mainCamera);
                renderer.DrawMesh(containerMesh, woodMaterial, transformMatrix);

                renderer.EndRenderPass();
                renderer.EndFrame();
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