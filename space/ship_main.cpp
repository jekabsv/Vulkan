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

#include "Assembly.h"
#include "HullGenerator.h"
#include "MeshBuilder.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <vector>

namespace
{
    // The analysis panel: mass properties are read straight off the assembly's
    // incrementally-maintained running sums (no recompute here), and the BOM is
    // the live Mark -> count map. This is meant to stand in for a real UI panel
    // that would update every keystroke; the console print is a stand-in for
    // "on-screen text renderer", which does not exist in this engine yet.
    void PrintAnalysisPanel(const Ship::Assembly& assembly)
    {
        const Ship::MassProperties& mp = assembly.GetMassProperties();

        std::printf("\n=== Analysis Panel ===\n");
        std::printf("Mass:            %.1f kg (%.2f t)\n", mp.mass, mp.mass / 1000.0);
        std::printf("Centre of mass:  (%.3f, %.3f, %.3f) m\n", mp.centerOfMass.x, mp.centerOfMass.y, mp.centerOfMass.z);
        std::printf("Inertia about COM (kg*m^2), diagonal:\n");
        std::printf("  Ixx=%.1f  Iyy=%.1f  Izz=%.1f\n", mp.inertiaAboutCOM[0][0], mp.inertiaAboutCOM[1][1], mp.inertiaAboutCOM[2][2]);
        std::printf("Inertia about COM, off-diagonal (products):\n");
        std::printf("  Ixy=%.1f  Ixz=%.1f  Iyz=%.1f\n", mp.inertiaAboutCOM[0][1], mp.inertiaAboutCOM[0][2], mp.inertiaAboutCOM[1][2]);

        const Ship::BillOfMaterials& bom = assembly.GetBillOfMaterials();
        std::printf("\nBill of materials: %zu unique marks, %zu total parts\n",
            bom.UniqueMarkCount(), assembly.Members().size() + assembly.Panels().size());

        std::vector<std::pair<Ship::MarkId, Ship::MarkEntry>> rows(bom.Entries().begin(), bom.Entries().end());
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.count > b.second.count; });

        for (const auto& [mark, entry] : rows)
        {
            std::printf("  x%-3u  %-45s  %.2f kg each\n", entry.count, entry.description.c_str(), entry.unitMassKg);
        }
        std::printf("=======================\n\n");
    }

    Ship::Assembly BuildDemoAssembly()
    {
        Ship::GridFrame grid;
        grid.unitMetres = 0.5;

        Ship::Assembly assembly(grid, Ship::MaterialCatalog());
        const Ship::MaterialId aluminum = assembly.Materials().Aluminum();
        const Ship::MaterialId steel = assembly.Materials().Steel();

        Ship::HullGeneratorParams hull;
        hull.startPoint = glm::dvec3(0.0, 0.0, 0.0);
        hull.axis = glm::dvec3(1.0, 0.0, 0.0);
        hull.length = 20.0;
        hull.stationCount = 14;
        hull.preset = Ship::CrossSectionPreset::Circular;
        hull.facetedSegments = 12;
        hull.startRadius = 2.2;
        hull.endRadius = 1.1; // stay well clear of the 0.5m grid's resolution limit for a 12-gon (see Assembly::AddMember guard)
        hull.longeronProfile = Ship::MakeBoxProfile(0.06, 0.04, 0.004);
        hull.ringProfile = Ship::MakeAngleProfile(0.05, 0.05, 0.004);
        hull.structureMaterial = aluminum;
        hull.skinThickness = 0.003;
        hull.skinMaterial = aluminum;

        const Ship::HullGeneratorResult result = Ship::GenerateHullSection(assembly, hull);
        std::printf("Generated hull: %zu ring members, %zu longerons, %zu skin panels\n",
            result.rings.size(), result.longerons.size(), result.skinPanels.size());

        // A standalone bulkhead panel with a rectangular opening, clear of the
        // hull, to exercise the "openings are a panel feature" path end to end:
        // the hole is just a second node loop, and mass/render both fall out of
        // the same polygon-with-holes math as the solid case.
        auto node = [&](int x, int y, int z) { return assembly.FindOrAddNode(Ship::GridCoord(x, y, z)); };

        Ship::Panel bulkhead;
        bulkhead.loop = { node(4, -14, 0), node(10, -14, 0), node(10, -10, 0), node(4, -10, 0) }; // 3m x 2m hatch cover
        bulkhead.openings.push_back({ node(6, -13, 0), node(8, -13, 0), node(8, -11, 0), node(6, -11, 0) }); // 1m x 1m port
        bulkhead.thickness = 0.010;
        bulkhead.material = steel;
        assembly.AddPanel(bulkhead);

        return assembly;
    }

    struct OrbitCamera
    {
        glm::dvec3 target{ 0.0 };
        double yaw{ -0.6 };
        double pitch{ 0.35 };
        double distance{ 30.0 };

        glm::vec3 Position() const
        {
            const double cp = std::cos(pitch);
            const glm::dvec3 dir(std::cos(yaw) * cp, std::sin(pitch), std::sin(yaw) * cp);
            return glm::vec3(target + dir * distance);
        }
    };
}

int main()
{
    try
    {
        Core::Window window(1600, 900, "Ship Builder - Prototype");
        Core::VulkanContext context(window);
        Core::Renderer renderer(context, window);

        Core::Shader vertexShader(context, "vert.spv");
        Core::Shader fragmentShader(context, "frag.spv");

        std::vector<const Core::Shader*> shaders{ &vertexShader, &fragmentShader };

        Core::PipelineConfig pipelineConfig;
        pipelineConfig.vertexInput = Core::Vertex::GetLayout();
        pipelineConfig.cullMode = VK_CULL_MODE_NONE; // generator winding isn't guaranteed consistent yet; see MeshBuilder.cpp
        pipelineConfig.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        pipelineConfig.depthTestEnable = true;
        pipelineConfig.depthWriteEnable = true;
        pipelineConfig.colorAttachmentFormats.push_back(renderer.GetColorFormat());
        pipelineConfig.depthAttachmentFormat = renderer.GetDepthFormat();

        Core::GraphicsPipeline pipeline(context, shaders, pipelineConfig);

        const uint8_t whitePixel[4] = { 255, 255, 255, 255 };
        Core::TextureConfig textureConfig;
        textureConfig.width = 1;
        textureConfig.height = 1;
        textureConfig.format = VK_FORMAT_R8G8B8A8_SRGB;
        textureConfig.generateMipmaps = false;
        Core::Texture whiteTexture(context, textureConfig, whitePixel, sizeof(whitePixel));

        Ship::Assembly assembly = BuildDemoAssembly();
        PrintAnalysisPanel(assembly);

        const std::vector<Ship::RenderBatch> batches = Ship::BuildRenderBatches(assembly);

        std::vector<Core::Mesh> meshes;
        std::vector<Core::Material> materials;
        meshes.reserve(batches.size());
        materials.reserve(batches.size());

        for (const Ship::RenderBatch& batch : batches)
        {
            meshes.push_back(Core::Mesh::CreateCustom(context, batch.vertices, batch.indices));

            Core::Material material = renderer.CreateMaterial(pipeline);
            const Ship::PartMaterial& partMaterial = assembly.Materials().Get(batch.material);
            material.SetVec3("u_AlbedoColor", partMaterial.renderColor);
            material.SetFloat("u_Roughness", 0.55f);
            material.SetFloat("u_Metallic", 0.85f);
            material.SetVec2("u_Tiling", glm::vec2(1.0f, 1.0f));
            material.SetTexture("u_AlbedoMap", whiteTexture);
            materials.push_back(std::move(material));

            std::printf("Batch [%s]: %zu vertices, %zu indices\n",
                partMaterial.name.c_str(), batch.vertices.size(), batch.indices.size());
        }

        OrbitCamera orbitCamera;
        orbitCamera.target = glm::dvec3(10.0, 0.0, -5.0);

        Core::Camera mainCamera;

        while (!window.ShouldClose())
        {
            window.PollEvents();

            if (window.IsKeyPressed(Core::KeyCode::Escape))
            {
                window.Close();
                continue;
            }

            const Core::InputSnapshot& input = window.GetInputSnapshot();

            if (window.IsMouseButtonDown(Core::MouseButton::Left))
            {
                orbitCamera.yaw += input.mouseDeltaX * 0.005;
                orbitCamera.pitch = std::clamp(orbitCamera.pitch - input.mouseDeltaY * 0.005, -1.5, 1.5);
            }

            orbitCamera.distance = std::clamp(orbitCamera.distance - input.scrollWheelDelta * 2.0, 3.0, 120.0);

            mainCamera.position = orbitCamera.Position();
            mainCamera.view = glm::lookAt(mainCamera.position, glm::vec3(orbitCamera.target), glm::vec3(0.0f, 1.0f, 0.0f));
            mainCamera.projection = glm::perspective(glm::radians(50.0f), renderer.GetAspectRatio(), 0.1f, 500.0f);
            mainCamera.projection[1][1] *= -1.0f;

            if (renderer.BeginFrame())
            {
                renderer.BeginRenderPass(glm::vec4(0.03f, 0.035f, 0.05f, 1.0f));
                renderer.SetCamera(mainCamera);

                for (size_t i = 0; i < meshes.size(); i++)
                {
                    renderer.DrawMesh(meshes[i], materials[i], glm::mat4(1.0f));
                }

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
