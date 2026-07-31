#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "Window.h"
#include "VulkanContext.h"
#include "Renderer.h"
#include "Shader.h"
#include "Pipeline.h"
#include "BeamEditor.h"

#include <glm/glm.hpp>

#include <cstdlib>
#include <iostream>
#include <vector>

int main()
{
    try
    {
        Core::Window window(1600, 900, "Hull Beams");
        Core::VulkanContext context(window);
        Core::Renderer renderer(context, window);

        Core::Shader vertexShader(context, "vert.spv");
        Core::Shader fragmentShader(context, "frag.spv");

        std::vector<const Core::Shader*> shaders;
        shaders.push_back(&vertexShader);
        shaders.push_back(&fragmentShader);

        Core::PipelineConfig pipelineConfig;
        pipelineConfig.vertexInput = Core::Vertex::GetLayout();
        pipelineConfig.cullMode = VK_CULL_MODE_NONE;
        pipelineConfig.depthTestEnable = true;
        pipelineConfig.depthWriteEnable = true;
        pipelineConfig.colorAttachmentFormats.push_back(renderer.GetColorFormat());
        pipelineConfig.depthAttachmentFormat = renderer.GetDepthFormat();

        Core::GraphicsPipeline beamPipeline(context, shaders, pipelineConfig);

        Editor::BeamEditor editor(context, window, renderer, beamPipeline);

        while (!window.ShouldClose())
        {
            window.PollEvents();

            //if (window.IsKeyPressed(Core::KeyCode::Escape))
            //{
            //    window.Close();
            //    continue;
            //}

            if (window.IsMinimized())
            {
                window.WaitEvents();
                continue;
            }

            editor.Update();

            if (renderer.BeginFrame())
            {
                renderer.BeginRenderPass(glm::vec4(0.05f, 0.06f, 0.08f, 1.0f));

                editor.Render();

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