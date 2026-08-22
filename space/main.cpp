#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "VulkanAbstraction.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "ParticleManager.h"
#include "ParticleSimulation.h"

namespace
{
    constexpr uint32_t OrbiterCount = 4096;
    constexpr float CentralMass = 500.0f;
    constexpr float Gravity = 1.0f;

    constexpr float InnerRadius = 1.5f;
    constexpr float OuterRadius = 8.0f;

    // The N^2 kernel is stable well past this, but a large step turns a circular orbit into an
    // outward spiral through plain integration error, so the step is capped rather than tracking
    // a slow frame.
    constexpr float MaxTimeStep = 1.0f / 120.0f;

    std::string FormatFloat(float value, int decimals)
    {
        std::string text = std::to_string(value);
        const size_t dot = text.find('.');

        if (dot != std::string::npos && dot + static_cast<size_t>(decimals) + 1 < text.size())
        {
            text.erase(dot + static_cast<size_t>(decimals) + 1);
        }

        return text;
    }

    // A heavy body at the origin with a disc of orbiters around it. Each orbiter gets the circular
    // velocity for its radius, sqrt(G * M / r), so the disc holds together instead of collapsing
    // on the first frame — which makes it obvious at a glance whether the integrator is working.
    void SeedGalaxy(Core::ParticleManager& particles, uint32_t orbiterCount)
    {
        particles.Clear();
        particles.Reserve(orbiterCount + 1);

        particles.CreateParticle(Core::ParticleState{ glm::vec3(0.0f), glm::vec3(0.0f), CentralMass });

        std::mt19937 rng(1337);
        std::uniform_real_distribution<float> angleDist(0.0f, 6.2831853f);
        std::uniform_real_distribution<float> radiusDist(InnerRadius, OuterRadius);
        std::uniform_real_distribution<float> heightDist(-0.15f, 0.15f);
        std::uniform_real_distribution<float> massDist(0.5f, 1.5f);

        for (uint32_t i = 0; i < orbiterCount; i++)
        {
            const float angle = angleDist(rng);
            const float radius = radiusDist(rng);

            const glm::vec3 position(radius * std::cos(angle), heightDist(rng), radius * std::sin(angle));

            // Tangent in the xz plane, so the whole disc circulates the same way.
            const glm::vec3 tangent(-std::sin(angle), 0.0f, std::cos(angle));
            const float orbitalSpeed = std::sqrt(Gravity * CentralMass / radius);

            particles.CreateParticle(Core::ParticleState{ position, tangent * orbitalSpeed, massDist(rng) });
        }
    }
}

int main()
{
    try
    {
        Core::Window window(1280, 720, "N-Body");
        Core::VulkanContext context(window);
        Core::Renderer renderer(context, window);
        Core::AssetManager assets(context, renderer);

        // --- Shaders ---
        assets.LoadShader("particle_vert", "particle_vert.spv");
        assets.LoadShader("particle_frag", "particle_frag.spv");
        assets.LoadShader("nbody_comp", "nbody.spv");
        assets.LoadShader("text_instanced_vert", "text_instanced_vert.spv");
        assets.LoadShader("text_instanced_frag", "text_instanced_frag.spv");

        // --- Pipelines ---
        // No per-instance vertex layout: the vertex shader reads the particle storage buffer by
        // gl_InstanceIndex, so there is nothing per-instance for the host to supply.
        Core::PipelineConfig particlePipelineConfig;
        particlePipelineConfig.vertexInput = Core::Vertex::GetLayout();
        particlePipelineConfig.cullMode = VK_CULL_MODE_BACK_BIT;
        particlePipelineConfig.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        particlePipelineConfig.depthTestEnable = true;
        particlePipelineConfig.depthWriteEnable = true;
        particlePipelineConfig.colorAttachmentFormats.push_back(renderer.GetColorFormat());
        particlePipelineConfig.depthAttachmentFormat = renderer.GetDepthFormat();

        std::vector<const Core::Shader*> particleShaders;
        particleShaders.push_back(&assets.GetShader("particle_vert"));
        particleShaders.push_back(&assets.GetShader("particle_frag"));

        assets.SetPipeline("particles", Core::GraphicsPipeline(context, particleShaders, particlePipelineConfig));

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

        assets.SetComputePipeline("nbody", Core::ComputePipeline(context, assets.GetShader("nbody_comp")));

        // --- Mesh ---
        // Radius 1.0: particle.vert scales by u_ParticleScale, and any radius baked into the mesh
        // would multiply into that. Low tessellation because this is drawn thousands of times.
        assets.SetMesh("particle", Core::Mesh::CreateSphere(context, 1.0f, 12, 8));

        // --- Material ---
        Core::Material& particleMaterial = assets.SetMaterial("particles", renderer.CreateMaterial(assets.GetPipeline("particles")));
        particleMaterial.SetVec3("u_AlbedoColor", glm::vec3(0.35f, 0.45f, 0.85f));
        particleMaterial.SetFloat("u_ParticleScale", 0.05f);
        particleMaterial.SetVec3("u_FastColor", glm::vec3(1.0f, 0.75f, 0.3f));
        particleMaterial.SetFloat("u_SpeedRange", 20.0f);

        Core::Font& textFont = assets.LoadFont("main", assets.GetPipeline("text"), "font.ttf", 48.0f);
        const float textScale = 1.0f / 48.0f;

        // --- Simulation ---
        Core::ParticleManager particles;
        SeedGalaxy(particles, OrbiterCount);

        Core::ParticleSimulationConfig simulationConfig;
        simulationConfig.capacity = OrbiterCount * 2;
        simulationConfig.gravity = Gravity;
        simulationConfig.softening = 0.05f;

        Core::ParticleSimulation simulation(context, assets.GetComputePipeline("nbody"), simulationConfig);

        particles.AttachSyncSource(&simulation);
        simulation.Upload(particles);

        Core::Camera mainCamera;
        mainCamera.SetPosition(glm::vec3(0.0f, 9.0f, 16.0f));
        mainCamera.LookAt(glm::vec3(0.0f));

        auto frameStart = std::chrono::high_resolution_clock::now();
        float previousFrameTime = 1.0f / 60.0f;
        glm::vec2 lastMousePos = window.GetMousePos();

        std::string syncMessage = "running";

        while (!window.ShouldClose())
        {
            window.PollEvents();

            if (window.IsKeyPressed(Core::KeyCode::Escape))
            {
                window.Close();
                continue;
            }

            const float cameraSpeed = 12.0f * previousFrameTime;

            if (window.IsKeyDown(Core::KeyCode::W)) { mainCamera.MoveForward(cameraSpeed); }
            if (window.IsKeyDown(Core::KeyCode::S)) { mainCamera.MoveForward(-cameraSpeed); }
            if (window.IsKeyDown(Core::KeyCode::A)) { mainCamera.MoveRight(-cameraSpeed); }
            if (window.IsKeyDown(Core::KeyCode::D)) { mainCamera.MoveRight(cameraSpeed); }
            if (window.IsKeyDown(Core::KeyCode::Space)) { mainCamera.MoveWorld(glm::vec3(0.0f, cameraSpeed, 0.0f)); }
            if (window.IsKeyDown(Core::KeyCode::Q)) { mainCamera.MoveWorld(glm::vec3(0.0f, -cameraSpeed, 0.0f)); }

            const glm::vec2 mousePos = window.GetMousePos();
            const glm::vec2 mouseDelta = mousePos - lastMousePos;
            lastMousePos = mousePos;

            if (window.IsMouseButtonDown(Core::MouseButton::Right))
            {
                constexpr float mouseSensitivity = 0.0025f;
                mainCamera.Yaw(-mouseDelta.x * mouseSensitivity);
                mainCamera.Pitch(-mouseDelta.y * mouseSensitivity);
            }

            // --- Host/device sync ---
            // Both of these run outside the BeginFrame/EndFrame pair, which AcquireLatest requires
            // and PeekLatest needs so it reads the ring slot before this frame overwrites it.

            // G: the full read-modify-write cycle. Halts the device, edits exact state, resumes.
            if (window.IsKeyPressed(Core::KeyCode::G))
            {
                if (particles.GetLatest())
                {
                    // Kick every orbiter outward a little, and add a second heavy body so the
                    // structural path (realloc + re-upload) gets exercised too.
                    const std::vector<Core::ParticleId>& ids = particles.GetIds();

                    for (Core::ParticleId id : ids)
                    {
                        glm::vec3 position;
                        glm::vec3 velocity;

                        if (particles.GetPosition(id, position) && particles.GetVelocity(id, velocity))
                        {
                            const float distance = glm::length(position);

                            if (distance > 0.001f)
                            {
                                particles.SetVelocity(id, velocity + (position / distance) * 0.5f);
                            }
                        }
                    }

                    particles.CreateParticle(Core::ParticleState{ glm::vec3(0.0f, 4.0f, 0.0f), glm::vec3(6.0f, 0.0f, 0.0f), 120.0f });
                    particles.Resume();

                    syncMessage = "kicked + spawned, resumed";
                }
            }

            // P: read-only snapshot. Never halts, and the data is a few frames stale.
            if (window.IsKeyPressed(Core::KeyCode::P))
            {
                if (particles.PeekLatest())
                {
                    float fastest = 0.0f;

                    for (const glm::vec3& velocity : particles.GetVelocities())
                    {
                        fastest = std::max(fastest, glm::length(velocity));
                    }

                    syncMessage = "peek: fastest " + FormatFloat(fastest, 2);
                }
                else
                {
                    syncMessage = "peek: no snapshot yet";
                }
            }

            // R: rebuild the whole system from scratch.
            if (window.IsKeyPressed(Core::KeyCode::R))
            {
                if (particles.GetLatest())
                {
                    SeedGalaxy(particles, OrbiterCount);
                    particles.Resume();

                    syncMessage = "reset";
                }
            }

            mainCamera.SetPerspective(glm::radians(50.0f), renderer.GetAspectRatio(), 0.1f, 500.0f);

            const float timeStep = std::min(previousFrameTime, MaxTimeStep);

            if (renderer.BeginFrame())
            {
                // The frame command buffer is open but the render pass has not begun, so compute
                // goes in here — no second submit, no queue stall.
                VkCommandBuffer commandBuffer = renderer.GetCommandBuffer();

                simulation.RecordCompute(commandBuffer, timeStep);
                simulation.RecordPeekCopy(commandBuffer);

                renderer.BeginRenderPass(glm::vec4(0.02f, 0.02f, 0.05f, 1.0f));
                renderer.SetCamera(mainCamera);

                // Rebound every frame because the ping-pong pair alternates which buffer holds the
                // newest state; Material skips the descriptor rewrite when the pointer is unchanged.
                particleMaterial.SetStorageBuffer("u_Particles", simulation.GetRenderBuffer());

                renderer.DrawMeshInstanced(assets.GetMesh("particle"), particleMaterial, simulation.GetParticleCount());

                renderer.BeginBatch();

                const float fps = 1.0f / std::max(0.0001f, previousFrameTime);

                renderer.SubmitText(textFont, std::to_string(simulation.GetParticleCount()) + " particles",
                    glm::vec3(-3.2f, 2.5f, 0.0f), textScale, glm::vec4(1.0f));
                renderer.SubmitText(textFont, FormatFloat(fps, 1) + " fps",
                    glm::vec3(-3.2f, 2.1f, 0.0f), textScale, glm::vec4(1.0f));
                renderer.SubmitText(textFont, simulation.IsHalted() ? "HALTED" : "running",
                    glm::vec3(-3.2f, 1.7f, 0.0f), textScale, glm::vec4(1.0f));
                renderer.SubmitText(textFont, syncMessage,
                    glm::vec3(-3.2f, 1.3f, 0.0f), textScale, glm::vec4(0.7f, 0.8f, 1.0f, 1.0f));
                renderer.SubmitText(textFont, "G kick+spawn   P peek   R reset",
                    glm::vec3(-3.2f, -2.5f, 0.0f), textScale, glm::vec4(0.6f, 0.6f, 0.7f, 1.0f));

                renderer.FlushBatch();

                renderer.EndRenderPass();
                renderer.EndFrame();

                // Only for a frame that was actually submitted: the ping-pong flip and the peek
                // ring both assume one call per submission.
                simulation.AdvanceFrame();
            }

            const auto now = std::chrono::high_resolution_clock::now();
            previousFrameTime = std::chrono::duration<float>(now - frameStart).count();
            frameStart = now;
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
