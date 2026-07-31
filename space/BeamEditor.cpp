#include "BeamEditor.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <utility>

namespace Editor
{
    namespace
    {
        const float kMinimumSize = 0.05f;
        const float kGrid = 0.25f;
        const float kAngleStep = 0.26179939f;
        const float kGroundSize = 40.0f;
        const float kRotateSpeed = 0.006f;
        const float kEpsilon = 0.000001f;
        const float kMinimumArea = 0.0004f;
        const float kSectionPresets[4] = { 0.125f, 0.25f, 0.5f, 1.0f };
        const float kThicknessPresets[4] = { 0.03f, 0.06f, 0.12f, 0.25f };

        const glm::vec3 kPaletteColors[static_cast<int>(Swatch::Count)] =
        {
            glm::vec3(0.66f, 0.69f, 0.74f),
            glm::vec3(1.00f, 0.55f, 0.12f),
            glm::vec3(0.20f, 0.90f, 1.00f),
            glm::vec3(0.14f, 0.15f, 0.18f),
            glm::vec3(0.35f, 0.55f, 0.70f),
            glm::vec3(0.30f, 0.34f, 0.46f),
            glm::vec3(0.42f, 0.52f, 0.50f),
            glm::vec3(0.95f, 0.25f, 0.25f),
            glm::vec3(0.35f, 0.90f, 0.35f),
            glm::vec3(0.35f, 0.50f, 1.00f),
            glm::vec3(0.16f, 0.17f, 0.20f),
            glm::vec3(0.20f, 0.95f, 0.45f),
            glm::vec3(0.98f, 0.85f, 0.25f)
        };

        float SnapValue(float value, float step)
        {
            return std::round(value / step) * step;
        }

        glm::vec3 MirrorPoint(const glm::vec3& point)
        {
            return glm::vec3(-point.x, point.y, point.z);
        }

        bool IntersectHorizontalPlane(const Ray& ray, float planeHeight, glm::vec3& outPoint)
        {
            if (std::fabs(ray.direction.y) < kEpsilon)
            {
                return false;
            }

            const float distance = (planeHeight - ray.origin.y) / ray.direction.y;

            if (distance < 0.0f)
            {
                return false;
            }

            outPoint = ray.origin + ray.direction * distance;
            return true;
        }

        bool ClosestPointOnAxis(const Ray& ray, const glm::vec3& linePoint, const glm::vec3& lineDirection, float& outParameter)
        {
            const glm::vec3 offset = linePoint - ray.origin;
            const float projection = glm::dot(lineDirection, ray.direction);
            const float denominator = 1.0f - projection * projection;

            if (std::fabs(denominator) < 0.0001f)
            {
                return false;
            }

            const float alongLine = glm::dot(lineDirection, offset);
            const float alongRay = glm::dot(ray.direction, offset);

            outParameter = (projection * alongRay - alongLine) / denominator;
            return true;
        }

        bool RayHitsTriangle(const Ray& ray, const glm::vec3& cornerA, const glm::vec3& cornerB, const glm::vec3& cornerC, float& outDistance)
        {
            const glm::vec3 edge1 = cornerB - cornerA;
            const glm::vec3 edge2 = cornerC - cornerA;
            const glm::vec3 sideVector = glm::cross(ray.direction, edge2);
            const float determinant = glm::dot(edge1, sideVector);

            if (std::fabs(determinant) < kEpsilon)
            {
                return false;
            }

            const float inverseDeterminant = 1.0f / determinant;
            const glm::vec3 toOrigin = ray.origin - cornerA;
            const float barycentricU = glm::dot(toOrigin, sideVector) * inverseDeterminant;

            if (barycentricU < 0.0f || barycentricU > 1.0f)
            {
                return false;
            }

            const glm::vec3 crossVector = glm::cross(toOrigin, edge1);
            const float barycentricV = glm::dot(ray.direction, crossVector) * inverseDeterminant;

            if (barycentricV < 0.0f || barycentricU + barycentricV > 1.0f)
            {
                return false;
            }

            const float distance = glm::dot(edge2, crossVector) * inverseDeterminant;

            if (distance < 0.0f)
            {
                return false;
            }

            outDistance = distance;
            return true;
        }

        Core::Mesh MakeBoxMesh(Core::VulkanContext& context)
        {
            const glm::vec3 faceCorners[6][4] =
            {
                { glm::vec3(0.5f, -0.5f, 0.5f), glm::vec3(0.5f, -0.5f, -0.5f), glm::vec3(0.5f, 0.5f, -0.5f), glm::vec3(0.5f, 0.5f, 0.5f) },
                { glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(-0.5f, -0.5f, 0.5f), glm::vec3(-0.5f, 0.5f, 0.5f), glm::vec3(-0.5f, 0.5f, -0.5f) },
                { glm::vec3(-0.5f, 0.5f, 0.5f), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(0.5f, 0.5f, -0.5f), glm::vec3(-0.5f, 0.5f, -0.5f) },
                { glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.5f, -0.5f, -0.5f), glm::vec3(0.5f, -0.5f, 0.5f), glm::vec3(-0.5f, -0.5f, 0.5f) },
                { glm::vec3(-0.5f, -0.5f, 0.5f), glm::vec3(0.5f, -0.5f, 0.5f), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(-0.5f, 0.5f, 0.5f) },
                { glm::vec3(0.5f, -0.5f, -0.5f), glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(-0.5f, 0.5f, -0.5f), glm::vec3(0.5f, 0.5f, -0.5f) }
            };

            const glm::vec3 faceNormals[6] =
            {
                glm::vec3(1.0f, 0.0f, 0.0f),
                glm::vec3(-1.0f, 0.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 0.0f),
                glm::vec3(0.0f, -1.0f, 0.0f),
                glm::vec3(0.0f, 0.0f, 1.0f),
                glm::vec3(0.0f, 0.0f, -1.0f)
            };

            const glm::vec2 cornerUvs[4] =
            {
                glm::vec2(0.0f, 0.0f),
                glm::vec2(1.0f, 0.0f),
                glm::vec2(1.0f, 1.0f),
                glm::vec2(0.0f, 1.0f)
            };

            std::vector<Core::Vertex> vertices;
            std::vector<uint32_t> indices;

            for (int face = 0; face < 6; face++)
            {
                const uint32_t base = static_cast<uint32_t>(vertices.size());

                for (int corner = 0; corner < 4; corner++)
                {
                    Core::Vertex vertex;
                    vertex.position = faceCorners[face][corner];
                    vertex.normal = faceNormals[face];
                    vertex.uv = cornerUvs[corner];
                    vertices.push_back(vertex);
                }

                indices.push_back(base + 0);
                indices.push_back(base + 1);
                indices.push_back(base + 2);
                indices.push_back(base + 2);
                indices.push_back(base + 3);
                indices.push_back(base + 0);
            }

            return Core::Mesh::CreateCustom(context, vertices, indices);
        }

        Core::Mesh MakePrismMesh(Core::VulkanContext& context)
        {
            const glm::vec3 flat[3] =
            {
                glm::vec3(0.0f, 0.0f, 0.0f),
                glm::vec3(1.0f, 0.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 0.0f)
            };

            const glm::vec2 flatUvs[3] =
            {
                glm::vec2(0.0f, 0.0f),
                glm::vec2(1.0f, 0.0f),
                glm::vec2(0.0f, 1.0f)
            };

            std::vector<Core::Vertex> vertices;
            std::vector<uint32_t> indices;

            for (int corner = 0; corner < 3; corner++)
            {
                Core::Vertex vertex;
                vertex.position = flat[corner] + glm::vec3(0.0f, 0.0f, 0.5f);
                vertex.normal = glm::vec3(0.0f, 0.0f, 1.0f);
                vertex.uv = flatUvs[corner];
                vertices.push_back(vertex);
            }

            indices.push_back(0);
            indices.push_back(1);
            indices.push_back(2);

            for (int corner = 0; corner < 3; corner++)
            {
                Core::Vertex vertex;
                vertex.position = flat[corner] - glm::vec3(0.0f, 0.0f, 0.5f);
                vertex.normal = glm::vec3(0.0f, 0.0f, -1.0f);
                vertex.uv = flatUvs[corner];
                vertices.push_back(vertex);
            }

            indices.push_back(3);
            indices.push_back(5);
            indices.push_back(4);

            for (int edge = 0; edge < 3; edge++)
            {
                const glm::vec3 start = flat[edge];
                const glm::vec3 end = flat[(edge + 1) % 3];
                const glm::vec3 direction = glm::normalize(end - start);
                const glm::vec3 normal = glm::cross(direction, glm::vec3(0.0f, 0.0f, 1.0f));

                const glm::vec3 quad[4] =
                {
                    start - glm::vec3(0.0f, 0.0f, 0.5f),
                    end - glm::vec3(0.0f, 0.0f, 0.5f),
                    end + glm::vec3(0.0f, 0.0f, 0.5f),
                    start + glm::vec3(0.0f, 0.0f, 0.5f)
                };

                const glm::vec2 quadUvs[4] =
                {
                    glm::vec2(0.0f, 0.0f),
                    glm::vec2(1.0f, 0.0f),
                    glm::vec2(1.0f, 1.0f),
                    glm::vec2(0.0f, 1.0f)
                };

                const uint32_t base = static_cast<uint32_t>(vertices.size());

                for (int corner = 0; corner < 4; corner++)
                {
                    Core::Vertex vertex;
                    vertex.position = quad[corner];
                    vertex.normal = normal;
                    vertex.uv = quadUvs[corner];
                    vertices.push_back(vertex);
                }

                indices.push_back(base + 0);
                indices.push_back(base + 1);
                indices.push_back(base + 2);
                indices.push_back(base + 2);
                indices.push_back(base + 3);
                indices.push_back(base + 0);
            }

            return Core::Mesh::CreateCustom(context, vertices, indices);
        }

        Core::Texture MakeWhiteTexture(Core::VulkanContext& context)
        {
            const std::vector<uint8_t> pixels(4 * 4 * 4, 255);

            Core::TextureConfig config;
            config.width = 4;
            config.height = 4;
            config.format = VK_FORMAT_R8G8B8A8_SRGB;
            config.generateMipmaps = false;

            return Core::Texture(context, config, pixels.data(), static_cast<VkDeviceSize>(pixels.size()));
        }
    }

    BeamEditor::BeamEditor(Core::VulkanContext& context, Core::Window& window, Core::Renderer& renderer, const Core::GraphicsPipeline& pipeline)
        : window(window)
        , renderer(renderer)
        , pipeline(pipeline)
        , boxMesh(MakeBoxMesh(context))
        , prismMesh(MakePrismMesh(context))
        , whiteTexture(MakeWhiteTexture(context))
    {
        palette.reserve(static_cast<size_t>(Swatch::Count));

        for (int index = 0; index < static_cast<int>(Swatch::Count); index++)
        {
            palette.push_back(renderer.CreateMaterial(pipeline));

            Core::Material& material = palette.back();
            material.SetVec3("u_AlbedoColor", kPaletteColors[index]);
            material.SetFloat("u_Roughness", 0.85f);
            material.SetFloat("u_Metallic", 0.0f);
            material.SetVec2("u_Tiling", glm::vec2(1.0f, 1.0f));
            material.SetTexture("u_AlbedoMap", whiteTexture);
        }

        Beam keel;
        keel.size = glm::vec3(sectionSize, sectionSize, 8.0f);
        keel.center = glm::vec3(0.0f, 0.0f, 0.0f);
        beams.push_back(keel);
        selectedBeam = 0;

        previousMouse = window.GetMousePos();
        lastTime = std::chrono::steady_clock::now();
    }

    void BeamEditor::Update()
    {
        const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        float deltaTime = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;

        if (deltaTime > 0.1f)
        {
            deltaTime = 0.1f;
        }

        const glm::vec2 mouse = window.GetMousePos();
        mouseDelta = mouse - previousMouse;
        previousMouse = mouse;

        UpdateCameraInput();
        UpdateCameraMatrices();
        UpdateHover();
        UpdateDrag();
        UpdateModeKeys();
        UpdateActionKeys(deltaTime);
        UpdateStatus();
    }

    void BeamEditor::UpdateCameraInput()
    {
        const float scroll = static_cast<float>(window.GetScrollDelta());

        if (std::fabs(scroll) > kEpsilon)
        {
            cameraDistance *= std::pow(0.88f, scroll);

            if (cameraDistance < 0.6f)
            {
                cameraDistance = 0.6f;
            }

            if (cameraDistance > 300.0f)
            {
                cameraDistance = 300.0f;
            }
        }

        if (window.IsMouseButtonDown(Core::MouseButton::Right))
        {
            cameraYaw -= mouseDelta.x * 0.005f;
            cameraPitch += mouseDelta.y * 0.005f;

            if (cameraPitch > 1.53f)
            {
                cameraPitch = 1.53f;
            }

            if (cameraPitch < -1.53f)
            {
                cameraPitch = -1.53f;
            }
        }

        if (window.IsMouseButtonDown(Core::MouseButton::Middle))
        {
            const glm::vec3 forward = glm::normalize(cameraTarget - camera.position);
            const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
            const glm::vec3 up = glm::normalize(glm::cross(right, forward));
            const float scale = cameraDistance * 0.0015f;

            cameraTarget += (-right * mouseDelta.x + up * mouseDelta.y) * scale;
        }
    }

    void BeamEditor::UpdateCameraMatrices()
    {
        glm::vec3 offset;
        offset.x = cameraDistance * std::cos(cameraPitch) * std::sin(cameraYaw);
        offset.y = cameraDistance * std::sin(cameraPitch);
        offset.z = cameraDistance * std::cos(cameraPitch) * std::cos(cameraYaw);

        camera.position = cameraTarget + offset;
        camera.view = glm::lookAt(camera.position, cameraTarget, glm::vec3(0.0f, 1.0f, 0.0f));
        camera.projection = glm::perspective(glm::radians(50.0f), renderer.GetAspectRatio(), 0.05f, 800.0f);
        camera.projection[1][1] *= -1.0f;
    }

    void BeamEditor::UpdateHover()
    {
        if (dragMode != DragMode::None)
        {
            return;
        }

        const Ray ray = MakeMouseRay();
        FaceHit hit;

        if (PickFace(ray, hit))
        {
            hoveredFace = hit;
        }
        else
        {
            hoveredFace = FaceHit();
        }

        placement = ComputePlacement(ray);
    }

    void BeamEditor::UpdateDrag()
    {
        if (tool == Tool::Plate)
        {
            if (window.IsMouseButtonPressed(Core::MouseButton::Left))
            {
                AddPlatePoint();
            }

            return;
        }

        const bool leftDown = window.IsMouseButtonDown(Core::MouseButton::Left);

        if (window.IsMouseButtonPressed(Core::MouseButton::Left))
        {
            BeginDrag();
            return;
        }

        if (!leftDown)
        {
            if (dragMode != DragMode::None)
            {
                EndDrag();
            }

            return;
        }

        if (dragMode == DragMode::None)
        {
            return;
        }

        const Ray ray = MakeMouseRay();

        if (dragMode == DragMode::Resize)
        {
            ApplyResize(ray);
        }
        else if (dragMode == DragMode::Move)
        {
            ApplyMove(ray);
        }
        else if (dragMode == DragMode::Rotate)
        {
            ApplyRotateDrag();
        }
    }

    void BeamEditor::BeginDrag()
    {
        dragMode = DragMode::None;

        const Ray ray = MakeMouseRay();

        FaceHit hit;
        const bool hasBeam = PickFace(ray, hit);

        int plateIndex = -1;
        float plateDistance = 0.0f;
        const bool hasPlate = PickPlate(ray, plateIndex, plateDistance);

        if (hasPlate && (!hasBeam || plateDistance < hit.distance))
        {
            selectedPlate = plateIndex;
            selectedBeam = -1;
            return;
        }

        if (hasBeam && hit.beamIndex != selectedBeam)
        {
            selectedBeam = hit.beamIndex;
            selectedPlate = -1;
            return;
        }

        if (selectedBeam < 0)
        {
            return;
        }

        const Beam& beam = beams[selectedBeam];

        if (tool == Tool::Rotate)
        {
            dragStartOrientation = beam.orientation;
            dragStartMouseX = window.GetMousePos().x;
            dragMode = DragMode::Rotate;
            return;
        }

        if (tool == Tool::Move)
        {
            dragStartCenter = beam.center;

            if (verticalMove)
            {
                dragAxis = glm::vec3(0.0f, 1.0f, 0.0f);
                dragAnchor = beam.center;

                if (ClosestPointOnAxis(ray, dragAnchor, dragAxis, dragStartParameter))
                {
                    dragMode = DragMode::Move;
                }
            }
            else
            {
                glm::vec3 point;

                if (IntersectHorizontalPlane(ray, beam.center.y, point))
                {
                    dragOffset = beam.center - point;
                    dragMode = DragMode::Move;
                }
            }

            return;
        }

        if (!hasBeam)
        {
            selectedBeam = -1;
            return;
        }

        dragFace = hit;
        dragAxis = FaceNormal(beam, hit.axis, hit.sign);
        dragAnchor = beam.center - dragAxis * (beam.size[hit.axis] * 0.5f);
        dragStartLength = beam.size[hit.axis];

        if (ClosestPointOnAxis(ray, dragAnchor, dragAxis, dragStartParameter))
        {
            dragMode = DragMode::Resize;
        }
    }

    void BeamEditor::EndDrag()
    {
        if (dragMode == DragMode::Resize && dragFace.beamIndex >= 0 && dragFace.beamIndex < static_cast<int>(beams.size()))
        {
            const Beam& beam = beams[dragFace.beamIndex];
            const float length = beam.size[dragFace.axis];

            if (length >= beam.size.x && length >= beam.size.y && length >= beam.size.z)
            {
                newBeamLength = length;
            }
        }

        dragMode = DragMode::None;
    }

    void BeamEditor::ApplyResize(const Ray& ray)
    {
        if (dragFace.beamIndex < 0 || dragFace.beamIndex >= static_cast<int>(beams.size()))
        {
            return;
        }

        float parameter = 0.0f;

        if (!ClosestPointOnAxis(ray, dragAnchor, dragAxis, parameter))
        {
            return;
        }

        float length = dragStartLength + (parameter - dragStartParameter);

        if (snapMode != SnapMode::Free)
        {
            length = SnapValue(length, kGrid);
        }

        if (length < kMinimumSize)
        {
            length = kMinimumSize;
        }

        Beam& beam = beams[dragFace.beamIndex];
        beam.size[dragFace.axis] = length;
        beam.center = dragAnchor + dragAxis * (length * 0.5f);
    }

    void BeamEditor::ApplyMove(const Ray& ray)
    {
        if (selectedBeam < 0)
        {
            return;
        }

        Beam& beam = beams[selectedBeam];

        if (verticalMove)
        {
            float parameter = 0.0f;

            if (!ClosestPointOnAxis(ray, dragAnchor, dragAxis, parameter))
            {
                return;
            }

            float height = dragStartCenter.y + (parameter - dragStartParameter);

            if (snapMode != SnapMode::Free)
            {
                height = SnapValue(height, kGrid);
            }

            beam.center.y = height;
            return;
        }

        glm::vec3 point;

        if (!IntersectHorizontalPlane(ray, dragStartCenter.y, point))
        {
            return;
        }

        glm::vec3 center = point + dragOffset;
        center.y = dragStartCenter.y;

        if (snapMode != SnapMode::Free)
        {
            center.x = SnapValue(center.x, kGrid);
            center.z = SnapValue(center.z, kGrid);
        }

        beam.center = center;
    }

    void BeamEditor::ApplyRotateDrag()
    {
        if (selectedBeam < 0)
        {
            return;
        }

        float angle = (window.GetMousePos().x - dragStartMouseX) * kRotateSpeed;

        if (snapMode != SnapMode::Free)
        {
            angle = SnapValue(angle, kAngleStep);
        }

        glm::vec3 axis(0.0f);
        axis[activeAxis] = 1.0f;

        Beam& beam = beams[selectedBeam];

        if (localAxis)
        {
            beam.orientation = glm::normalize(dragStartOrientation * glm::angleAxis(angle, axis));
        }
        else
        {
            beam.orientation = glm::normalize(glm::angleAxis(angle, axis) * dragStartOrientation);
        }
    }

    void BeamEditor::RotateSelected(float angle)
    {
        if (selectedBeam < 0)
        {
            return;
        }

        glm::vec3 axis(0.0f);
        axis[activeAxis] = 1.0f;

        Beam& beam = beams[selectedBeam];

        if (localAxis)
        {
            beam.orientation = glm::normalize(beam.orientation * glm::angleAxis(angle, axis));
        }
        else
        {
            beam.orientation = glm::normalize(glm::angleAxis(angle, axis) * beam.orientation);
        }
    }

    void BeamEditor::NudgeSelected(const glm::vec3& direction, float amount)
    {
        if (selectedBeam < 0)
        {
            return;
        }

        beams[selectedBeam].center += direction * amount;
    }

    void BeamEditor::UpdateModeKeys()
    {
        if (window.IsKeyPressed(Core::KeyCode::S))
        {
            tool = Tool::Select;
        }

        if (window.IsKeyPressed(Core::KeyCode::G))
        {
            tool = Tool::Move;
        }

        if (window.IsKeyPressed(Core::KeyCode::R))
        {
            tool = Tool::Rotate;
        }

        if (window.IsKeyPressed(Core::KeyCode::P))
        {
            tool = Tool::Plate;
        }

        if (window.IsKeyPressed(Core::KeyCode::X))
        {
            activeAxis = 0;
        }

        if (window.IsKeyPressed(Core::KeyCode::Y))
        {
            activeAxis = 1;
        }

        if (window.IsKeyPressed(Core::KeyCode::Z))
        {
            activeAxis = 2;
        }

        if (window.IsKeyPressed(Core::KeyCode::L))
        {
            localAxis = !localAxis;
        }

        if (window.IsKeyPressed(Core::KeyCode::V))
        {
            verticalMove = !verticalMove;
        }

        if (window.IsKeyPressed(Core::KeyCode::M))
        {
            mirrorEnabled = !mirrorEnabled;
        }

        if (window.IsKeyPressed(Core::KeyCode::C))
        {
            fanMode = !fanMode;
        }

        if (window.IsKeyPressed(Core::KeyCode::N))
        {
            if (snapMode == SnapMode::Free)
            {
                snapMode = SnapMode::Grid;
            }
            else if (snapMode == SnapMode::Grid)
            {
                snapMode = SnapMode::Face;
            }
            else if (snapMode == SnapMode::Face)
            {
                snapMode = SnapMode::Ends;
            }
            else
            {
                snapMode = SnapMode::Free;
            }
        }

        if (window.IsKeyPressed(Core::KeyCode::Num1))
        {
            SetSizePreset(0);
        }

        if (window.IsKeyPressed(Core::KeyCode::Num2))
        {
            SetSizePreset(1);
        }

        if (window.IsKeyPressed(Core::KeyCode::Num3))
        {
            SetSizePreset(2);
        }

        if (window.IsKeyPressed(Core::KeyCode::Num4))
        {
            SetSizePreset(3);
        }
    }

    void BeamEditor::SetSizePreset(int index)
    {
        if (tool == Tool::Plate)
        {
            plateThickness = kThicknessPresets[index];
            return;
        }

        sectionSize = kSectionPresets[index];
    }

    void BeamEditor::UpdateActionKeys(float deltaTime)
    {
        if (window.IsKeyPressed(Core::KeyCode::Tab))
        {
            CycleSelection();
        }

        if (window.IsKeyPressed(Core::KeyCode::Space))
        {
            FocusSelection();
        }

        if (window.IsKeyPressed(Core::KeyCode::Enter))
        {
            BakeMirror();
        }

        if (window.IsKeyPressed(Core::KeyCode::Backspace))
        {
            RemoveLastPlatePoint();
        }

        if (dragMode == DragMode::None)
        {
            if (window.IsKeyPressed(Core::KeyCode::B))
            {
                AddBeamAtCursor();
            }

            if (window.IsKeyPressed(Core::KeyCode::Insert))
            {
                DuplicateSelected();
            }

            if (window.IsKeyPressed(Core::KeyCode::Delete))
            {
                DeleteSelection();
            }
        }

        float step = kGrid;

        if (snapMode == SnapMode::Free)
        {
            step = 0.05f;
        }

        const bool left = ConsumeKey(Core::KeyCode::Left, repeatLeft, deltaTime);
        const bool right = ConsumeKey(Core::KeyCode::Right, repeatRight, deltaTime);
        const bool up = ConsumeKey(Core::KeyCode::Up, repeatUp, deltaTime);
        const bool down = ConsumeKey(Core::KeyCode::Down, repeatDown, deltaTime);

        if (tool == Tool::Rotate)
        {
            if (left)
            {
                RotateSelected(-kAngleStep);
            }

            if (right)
            {
                RotateSelected(kAngleStep);
            }

            if (up)
            {
                RotateSelected(kAngleStep * 0.2f);
            }

            if (down)
            {
                RotateSelected(-kAngleStep * 0.2f);
            }

            return;
        }

        const glm::vec3 axisDirection = ActiveAxisDirection();

        if (left)
        {
            NudgeSelected(axisDirection, -step);
        }

        if (right)
        {
            NudgeSelected(axisDirection, step);
        }

        if (up)
        {
            NudgeSelected(glm::vec3(0.0f, 1.0f, 0.0f), step);
        }

        if (down)
        {
            NudgeSelected(glm::vec3(0.0f, 1.0f, 0.0f), -step);
        }
    }

    bool BeamEditor::ConsumeKey(Core::KeyCode key, KeyRepeat& repeat, float deltaTime)
    {
        if (window.IsKeyPressed(key))
        {
            repeat.active = true;
            repeat.timer = 0.0f;
            return true;
        }

        if (!window.IsKeyDown(key))
        {
            repeat.active = false;
            repeat.timer = 0.0f;
            return false;
        }

        if (!repeat.active)
        {
            return false;
        }

        repeat.timer += deltaTime;

        if (repeat.timer >= 0.4f)
        {
            repeat.timer = 0.34f;
            return true;
        }

        return false;
    }

    void BeamEditor::AddPlatePoint()
    {
        if (!placement.valid)
        {
            return;
        }

        pendingCorners.push_back(MakeAnchor(placement));

        if (pendingCorners.size() >= 3)
        {
            CreatePlate();
        }
    }

    void BeamEditor::RemoveLastPlatePoint()
    {
        if (pendingCorners.empty())
        {
            return;
        }

        pendingCorners.pop_back();
    }

    void BeamEditor::CreatePlate()
    {
        glm::vec3 corners[3];

        for (int index = 0; index < 3; index++)
        {
            corners[index] = ResolveAnchor(pendingCorners[index]);
        }

        const glm::vec3 area = glm::cross(corners[1] - corners[0], corners[2] - corners[0]);

        if (glm::length(area) < kMinimumArea)
        {
            pendingCorners.clear();
            return;
        }

        Plate plate;
        plate.corners[0] = pendingCorners[0];
        plate.corners[1] = pendingCorners[1];
        plate.corners[2] = pendingCorners[2];
        plate.thickness = plateThickness;

        plates.push_back(plate);
        selectedPlate = static_cast<int>(plates.size()) - 1;
        selectedBeam = -1;

        if (fanMode)
        {
            const Anchor first = pendingCorners[0];
            const Anchor last = pendingCorners[2];

            pendingCorners.clear();
            pendingCorners.push_back(first);
            pendingCorners.push_back(last);
            return;
        }

        pendingCorners.clear();
    }

    Anchor BeamEditor::MakeAnchor(const Placement& source) const
    {
        Anchor anchor;
        anchor.worldPoint = source.point;
        anchor.beamIndex = -1;

        if (!source.onFace || source.beamIndex < 0 || source.beamIndex >= static_cast<int>(beams.size()))
        {
            return anchor;
        }

        const glm::mat4 inverseModel = glm::inverse(MakeTransform(beams[source.beamIndex]));

        anchor.beamIndex = source.beamIndex;
        anchor.localPoint = glm::vec3(inverseModel * glm::vec4(source.point, 1.0f));
        return anchor;
    }

    glm::vec3 BeamEditor::ResolveAnchor(const Anchor& anchor) const
    {
        if (anchor.beamIndex < 0 || anchor.beamIndex >= static_cast<int>(beams.size()))
        {
            return anchor.worldPoint;
        }

        const glm::mat4 model = MakeTransform(beams[anchor.beamIndex]);
        return glm::vec3(model * glm::vec4(anchor.localPoint, 1.0f));
    }

    void BeamEditor::ResolvePlate(const Plate& plate, glm::vec3* outCorners) const
    {
        for (int index = 0; index < 3; index++)
        {
            outCorners[index] = ResolveAnchor(plate.corners[index]);
        }
    }

    glm::mat4 BeamEditor::MakePlateTransform(const glm::vec3* corners, float thickness) const
    {
        const glm::vec3 edge1 = corners[1] - corners[0];
        const glm::vec3 edge2 = corners[2] - corners[0];

        glm::vec3 normal = glm::cross(edge1, edge2);
        const float area = glm::length(normal);

        if (area < kMinimumArea)
        {
            return glm::scale(glm::mat4(1.0f), glm::vec3(0.0f));
        }

        normal /= area;

        glm::mat4 transform(1.0f);
        transform[0] = glm::vec4(edge1, 0.0f);
        transform[1] = glm::vec4(edge2, 0.0f);
        transform[2] = glm::vec4(normal * thickness, 0.0f);
        transform[3] = glm::vec4(corners[0], 1.0f);
        return transform;
    }

    void BeamEditor::AddBeamAtCursor()
    {
        if (!placement.valid)
        {
            return;
        }

        beams.push_back(MakePlacedBeam(placement));
        selectedBeam = static_cast<int>(beams.size()) - 1;
        selectedPlate = -1;
        hoveredFace = FaceHit();
    }

    void BeamEditor::DuplicateSelected()
    {
        if (selectedBeam < 0)
        {
            return;
        }

        Beam beam = beams[selectedBeam];
        const glm::vec3 direction = ActiveAxisDirection();

        float offset = kGrid;

        if (localAxis)
        {
            offset = beam.size[activeAxis];
        }

        beam.center += direction * offset;

        beams.push_back(beam);
        selectedBeam = static_cast<int>(beams.size()) - 1;
    }

    void BeamEditor::DeleteSelection()
    {
        if (selectedPlate >= 0)
        {
            DeletePlate(selectedPlate);
            return;
        }

        if (selectedBeam >= 0)
        {
            DeleteBeam(selectedBeam);
        }
    }

    void BeamEditor::DeletePlate(int index)
    {
        if (index < 0 || index >= static_cast<int>(plates.size()))
        {
            return;
        }

        plates.erase(plates.begin() + index);
        selectedPlate = -1;
    }

    void BeamEditor::DeleteBeam(int index)
    {
        if (index < 0 || index >= static_cast<int>(beams.size()))
        {
            return;
        }

        for (size_t plateIndex = 0; plateIndex < plates.size(); plateIndex++)
        {
            for (int corner = 0; corner < 3; corner++)
            {
                Anchor& anchor = plates[plateIndex].corners[corner];

                if (anchor.beamIndex == index)
                {
                    anchor.worldPoint = ResolveAnchor(anchor);
                    anchor.beamIndex = -1;
                }
            }
        }

        for (size_t pendingIndex = 0; pendingIndex < pendingCorners.size(); pendingIndex++)
        {
            Anchor& anchor = pendingCorners[pendingIndex];

            if (anchor.beamIndex == index)
            {
                anchor.worldPoint = ResolveAnchor(anchor);
                anchor.beamIndex = -1;
            }
        }

        beams.erase(beams.begin() + index);

        for (size_t plateIndex = 0; plateIndex < plates.size(); plateIndex++)
        {
            for (int corner = 0; corner < 3; corner++)
            {
                Anchor& anchor = plates[plateIndex].corners[corner];

                if (anchor.beamIndex > index)
                {
                    anchor.beamIndex--;
                }
            }
        }

        for (size_t pendingIndex = 0; pendingIndex < pendingCorners.size(); pendingIndex++)
        {
            Anchor& anchor = pendingCorners[pendingIndex];

            if (anchor.beamIndex > index)
            {
                anchor.beamIndex--;
            }
        }

        selectedBeam = -1;
        hoveredFace = FaceHit();
        dragMode = DragMode::None;
    }

    void BeamEditor::CycleSelection()
    {
        if (beams.empty())
        {
            selectedBeam = -1;
            return;
        }

        selectedPlate = -1;
        selectedBeam = (selectedBeam + 1) % static_cast<int>(beams.size());
    }

    void BeamEditor::FocusSelection()
    {
        if (selectedBeam < 0)
        {
            cameraTarget = glm::vec3(0.0f);
            return;
        }

        const Beam& beam = beams[selectedBeam];
        cameraTarget = beam.center;

        float extent = beam.size.x;

        if (beam.size.y > extent)
        {
            extent = beam.size.y;
        }

        if (beam.size.z > extent)
        {
            extent = beam.size.z;
        }

        cameraDistance = extent * 2.5f + 1.0f;
    }

    void BeamEditor::BakeMirror()
    {
        if (!mirrorEnabled)
        {
            return;
        }

        const size_t beamCount = beams.size();
        std::vector<int> mapping(beamCount, -1);

        for (size_t index = 0; index < beamCount; index++)
        {
            if (std::fabs(beams[index].center.x) < 0.001f)
            {
                continue;
            }

            mapping[index] = static_cast<int>(beams.size());
            beams.push_back(MakeMirrored(beams[index]));
        }

        const size_t plateCount = plates.size();

        for (size_t index = 0; index < plateCount; index++)
        {
            Plate mirrored;
            mirrored.thickness = plates[index].thickness;

            const int order[3] = { 0, 2, 1 };

            for (int corner = 0; corner < 3; corner++)
            {
                const Anchor& source = plates[index].corners[order[corner]];

                Anchor anchor;
                anchor.localPoint = source.localPoint;
                anchor.worldPoint = MirrorPoint(ResolveAnchor(source));
                anchor.beamIndex = -1;

                if (source.beamIndex >= 0 && source.beamIndex < static_cast<int>(mapping.size()))
                {
                    anchor.beamIndex = mapping[source.beamIndex];
                }

                mirrored.corners[corner] = anchor;
            }

            plates.push_back(mirrored);
        }

        mirrorEnabled = false;
    }

    Ray BeamEditor::MakeMouseRay() const
    {
        const glm::vec2 mouse = window.GetMousePos();
        const float width = static_cast<float>(window.GetWidth());
        const float height = static_cast<float>(window.GetHeight());

        Ray ray;

        if (width < 1.0f || height < 1.0f)
        {
            return ray;
        }

        glm::vec2 normalized;
        normalized.x = (2.0f * mouse.x / width) - 1.0f;
        normalized.y = (2.0f * mouse.y / height) - 1.0f;

        const glm::mat4 inverseViewProjection = glm::inverse(camera.projection * camera.view);

        glm::vec4 nearPoint = inverseViewProjection * glm::vec4(normalized.x, normalized.y, 0.0f, 1.0f);
        glm::vec4 farPoint = inverseViewProjection * glm::vec4(normalized.x, normalized.y, 1.0f, 1.0f);

        nearPoint /= nearPoint.w;
        farPoint /= farPoint.w;

        ray.origin = glm::vec3(nearPoint);
        ray.direction = glm::normalize(glm::vec3(farPoint - nearPoint));
        return ray;
    }

    bool BeamEditor::PickFace(const Ray& ray, FaceHit& outHit) const
    {
        bool found = false;
        FaceHit best;
        best.distance = std::numeric_limits<float>::max();

        for (size_t index = 0; index < beams.size(); index++)
        {
            float distance = 0.0f;
            int axis = 0;
            float sign = 1.0f;

            if (RayHitsBeam(ray, beams[index], distance, axis, sign))
            {
                if (distance < best.distance)
                {
                    best.beamIndex = static_cast<int>(index);
                    best.axis = axis;
                    best.sign = sign;
                    best.distance = distance;
                    best.point = ray.origin + ray.direction * distance;
                    found = true;
                }
            }
        }

        if (found)
        {
            outHit = best;
        }

        return found;
    }

    bool BeamEditor::PickPlate(const Ray& ray, int& outIndex, float& outDistance) const
    {
        bool found = false;
        float bestDistance = std::numeric_limits<float>::max();

        for (size_t index = 0; index < plates.size(); index++)
        {
            glm::vec3 corners[3];
            ResolvePlate(plates[index], corners);

            float distance = 0.0f;

            if (!RayHitsTriangle(ray, corners[0], corners[1], corners[2], distance))
            {
                continue;
            }

            if (distance < bestDistance)
            {
                bestDistance = distance;
                outIndex = static_cast<int>(index);
                found = true;
            }
        }

        if (found)
        {
            outDistance = bestDistance;
        }

        return found;
    }

    bool BeamEditor::RayHitsBeam(const Ray& ray, const Beam& beam, float& outDistance, int& outAxis, float& outSign) const
    {
        const glm::mat4 inverseModel = glm::inverse(MakeTransform(beam));
        const glm::vec3 localOrigin = glm::vec3(inverseModel * glm::vec4(ray.origin, 1.0f));
        const glm::vec3 localDirection = glm::vec3(inverseModel * glm::vec4(ray.direction, 0.0f));

        float entry = -std::numeric_limits<float>::max();
        float exit = std::numeric_limits<float>::max();
        int entryAxis = 0;
        float entrySign = 1.0f;

        for (int axis = 0; axis < 3; axis++)
        {
            if (std::fabs(localDirection[axis]) < kEpsilon)
            {
                if (localOrigin[axis] < -0.5f || localOrigin[axis] > 0.5f)
                {
                    return false;
                }
            }
            else
            {
                float nearHit = (-0.5f - localOrigin[axis]) / localDirection[axis];
                float farHit = (0.5f - localOrigin[axis]) / localDirection[axis];
                float sign = -1.0f;

                if (nearHit > farHit)
                {
                    std::swap(nearHit, farHit);
                    sign = 1.0f;
                }

                if (nearHit > entry)
                {
                    entry = nearHit;
                    entryAxis = axis;
                    entrySign = sign;
                }

                if (farHit < exit)
                {
                    exit = farHit;
                }

                if (entry > exit)
                {
                    return false;
                }
            }
        }

        if (entry < 0.0f)
        {
            return false;
        }

        outDistance = entry;
        outAxis = entryAxis;
        outSign = entrySign;
        return true;
    }

    Placement BeamEditor::ComputePlacement(const Ray& ray) const
    {
        Placement result;

        FaceHit hit;

        if (PickFace(ray, hit))
        {
            const Beam& beam = beams[hit.beamIndex];
            const int axisU = (hit.axis + 1) % 3;
            const int axisV = (hit.axis + 2) % 3;

            glm::vec3 localU(0.0f);
            glm::vec3 localV(0.0f);
            localU[axisU] = 1.0f;
            localV[axisV] = 1.0f;

            const glm::vec3 worldU = glm::normalize(beam.orientation * localU);
            const glm::vec3 worldV = glm::normalize(beam.orientation * localV);
            const glm::vec3 faceCenter = FaceCenter(beam, hit.axis, hit.sign);

            const float halfU = beam.size[axisU] * 0.5f;
            const float halfV = beam.size[axisV] * 0.5f;

            float offsetU = glm::dot(hit.point - faceCenter, worldU);
            float offsetV = glm::dot(hit.point - faceCenter, worldV);

            if (snapMode == SnapMode::Face)
            {
                offsetU = 0.0f;
                offsetV = 0.0f;
            }
            else if (snapMode == SnapMode::Grid)
            {
                offsetU = SnapValue(offsetU, kGrid);
                offsetV = SnapValue(offsetV, kGrid);
            }
            else if (snapMode == SnapMode::Ends)
            {
                const float candidatesU[3] = { -halfU, 0.0f, halfU };
                const float candidatesV[3] = { -halfV, 0.0f, halfV };

                float bestU = 0.0f;
                float bestV = 0.0f;
                float bestDistance = std::numeric_limits<float>::max();

                for (int indexU = 0; indexU < 3; indexU++)
                {
                    for (int indexV = 0; indexV < 3; indexV++)
                    {
                        const float deltaU = candidatesU[indexU] - offsetU;
                        const float deltaV = candidatesV[indexV] - offsetV;
                        const float distance = deltaU * deltaU + deltaV * deltaV;

                        if (distance < bestDistance)
                        {
                            bestDistance = distance;
                            bestU = candidatesU[indexU];
                            bestV = candidatesV[indexV];
                        }
                    }
                }

                offsetU = bestU;
                offsetV = bestV;
            }

            if (offsetU > halfU)
            {
                offsetU = halfU;
            }

            if (offsetU < -halfU)
            {
                offsetU = -halfU;
            }

            if (offsetV > halfV)
            {
                offsetV = halfV;
            }

            if (offsetV < -halfV)
            {
                offsetV = -halfV;
            }

            result.valid = true;
            result.onFace = true;
            result.beamIndex = hit.beamIndex;
            result.point = faceCenter + worldU * offsetU + worldV * offsetV;
            result.orientation = beam.orientation;
            result.axis = hit.axis;
            result.sign = hit.sign;
            return result;
        }

        glm::vec3 groundPoint;

        if (!IntersectHorizontalPlane(ray, 0.0f, groundPoint))
        {
            return result;
        }

        if (snapMode != SnapMode::Free)
        {
            groundPoint.x = SnapValue(groundPoint.x, kGrid);
            groundPoint.z = SnapValue(groundPoint.z, kGrid);
        }

        result.valid = true;
        result.onFace = false;
        result.beamIndex = -1;
        result.point = groundPoint;
        result.orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        result.axis = activeAxis;
        result.sign = 1.0f;

        if (activeAxis == 1)
        {
            return result;
        }

        result.point.y = sectionSize * 0.5f;
        return result;
    }

    Beam BeamEditor::MakePlacedBeam(const Placement& source) const
    {
        glm::vec3 localDirection(0.0f);
        localDirection[source.axis] = source.sign;

        const glm::vec3 worldDirection = glm::normalize(source.orientation * localDirection);

        Beam beam;
        beam.orientation = source.orientation;
        beam.size = glm::vec3(sectionSize);
        beam.size[source.axis] = newBeamLength;
        beam.center = source.point + worldDirection * (newBeamLength * 0.5f);
        return beam;
    }

    Beam BeamEditor::MakeMirrored(const Beam& beam) const
    {
        Beam mirrored;
        mirrored.center = MirrorPoint(beam.center);
        mirrored.orientation = glm::quat(beam.orientation.w, beam.orientation.x, -beam.orientation.y, -beam.orientation.z);
        mirrored.size = beam.size;
        return mirrored;
    }

    glm::mat4 BeamEditor::MakeTransform(const Beam& beam) const
    {
        const glm::mat4 translation = glm::translate(glm::mat4(1.0f), beam.center);
        const glm::mat4 rotation = glm::mat4_cast(beam.orientation);
        const glm::mat4 scale = glm::scale(glm::mat4(1.0f), beam.size);

        return translation * rotation * scale;
    }

    glm::vec3 BeamEditor::FaceNormal(const Beam& beam, int axis, float sign) const
    {
        glm::vec3 localAxisVector(0.0f);
        localAxisVector[axis] = sign;

        return glm::normalize(beam.orientation * localAxisVector);
    }

    glm::vec3 BeamEditor::FaceCenter(const Beam& beam, int axis, float sign) const
    {
        return beam.center + FaceNormal(beam, axis, sign) * (beam.size[axis] * 0.5f);
    }

    glm::vec3 BeamEditor::ActiveAxisDirection() const
    {
        glm::vec3 axis(0.0f);
        axis[activeAxis] = 1.0f;

        if (!localAxis || selectedBeam < 0)
        {
            return axis;
        }

        return glm::normalize(beams[selectedBeam].orientation * axis);
    }

    Core::Material& BeamEditor::GetMaterial(Swatch swatch)
    {
        return palette[static_cast<size_t>(swatch)];
    }

    void BeamEditor::DrawBeam(const Beam& beam, Swatch swatch)
    {
        renderer.DrawMesh(boxMesh, GetMaterial(swatch), MakeTransform(beam));
    }

    void BeamEditor::DrawPlate(const glm::vec3* corners, float thickness, Swatch swatch)
    {
        renderer.DrawMesh(prismMesh, GetMaterial(swatch), MakePlateTransform(corners, thickness));
    }

    void BeamEditor::DrawMarker(const glm::vec3& position, float size, Swatch swatch)
    {
        Beam marker;
        marker.center = position;
        marker.size = glm::vec3(size);
        DrawBeam(marker, swatch);
    }

    void BeamEditor::Render()
    {
        renderer.SetCamera(camera);

        Beam ground;
        ground.center = glm::vec3(0.0f, -0.03f, 0.0f);
        ground.size = glm::vec3(kGroundSize, 0.05f, kGroundSize);
        DrawBeam(ground, Swatch::Ground);

        if (mirrorEnabled)
        {
            Beam plane;
            plane.center = glm::vec3(0.0f, 2.0f, 0.0f);
            plane.size = glm::vec3(0.01f, 8.0f, kGroundSize * 0.6f);
            DrawBeam(plane, Swatch::Mirror);
        }

        for (size_t index = 0; index < beams.size(); index++)
        {
            if (static_cast<int>(index) == selectedBeam)
            {
                DrawBeam(beams[index], Swatch::Selected);
            }
            else
            {
                DrawBeam(beams[index], Swatch::Beam);
            }

            if (mirrorEnabled)
            {
                DrawBeam(MakeMirrored(beams[index]), Swatch::Mirror);
            }
        }

        for (size_t index = 0; index < plates.size(); index++)
        {
            glm::vec3 corners[3];
            ResolvePlate(plates[index], corners);

            if (static_cast<int>(index) == selectedPlate)
            {
                DrawPlate(corners, plates[index].thickness, Swatch::Selected);
            }
            else
            {
                DrawPlate(corners, plates[index].thickness, Swatch::Hull);
            }

            if (mirrorEnabled)
            {
                const glm::vec3 mirrored[3] = { MirrorPoint(corners[0]), MirrorPoint(corners[2]), MirrorPoint(corners[1]) };
                DrawPlate(mirrored, plates[index].thickness, Swatch::Mirror);
            }
        }

        if (selectedBeam >= 0)
        {
            const Beam& beam = beams[selectedBeam];
            const glm::vec3 direction = ActiveAxisDirection();
            const glm::vec3 stubCenter = beam.center + direction * (beam.size[activeAxis] * 0.5f + 0.35f);

            Swatch axisSwatch = Swatch::AxisX;

            if (activeAxis == 1)
            {
                axisSwatch = Swatch::AxisY;
            }
            else if (activeAxis == 2)
            {
                axisSwatch = Swatch::AxisZ;
            }

            DrawMarker(stubCenter, 0.18f, axisSwatch);
        }

        if (tool == Tool::Plate)
        {
            for (size_t index = 0; index < pendingCorners.size(); index++)
            {
                DrawMarker(ResolveAnchor(pendingCorners[index]), 0.16f, Swatch::Snap);
            }

            if (placement.valid)
            {
                DrawMarker(placement.point, 0.12f, Swatch::Snap);

                if (pendingCorners.size() == 2)
                {
                    const glm::vec3 preview[3] =
                    {
                        ResolveAnchor(pendingCorners[0]),
                        ResolveAnchor(pendingCorners[1]),
                        placement.point
                    };

                    DrawPlate(preview, plateThickness, Swatch::Ghost);

                    if (mirrorEnabled)
                    {
                        const glm::vec3 mirrored[3] = { MirrorPoint(preview[0]), MirrorPoint(preview[2]), MirrorPoint(preview[1]) };
                        DrawPlate(mirrored, plateThickness, Swatch::Mirror);
                    }
                }
            }
        }
        else if (dragMode == DragMode::None && placement.valid && tool == Tool::Select)
        {
            const Beam preview = MakePlacedBeam(placement);
            DrawBeam(preview, Swatch::Ghost);

            if (mirrorEnabled)
            {
                DrawBeam(MakeMirrored(preview), Swatch::Mirror);
            }

            DrawMarker(placement.point, 0.12f, Swatch::Snap);
        }

        DrawHud();
    }

    void BeamEditor::DrawHud()
    {
        HudFrame frame;
        frame.forward = glm::normalize(cameraTarget - camera.position);
        frame.right = glm::normalize(glm::cross(frame.forward, glm::vec3(0.0f, 1.0f, 0.0f)));
        frame.up = glm::normalize(glm::cross(frame.right, frame.forward));
        frame.origin = camera.position + frame.forward * 0.3f;

        const float halfHeight = 0.3f * std::tan(glm::radians(25.0f));
        const float halfWidth = halfHeight * renderer.GetAspectRatio();
        const float size = halfHeight * 0.075f;
        const float gap = size * 1.5f;
        const float groupGap = size * 2.6f;
        const float baseY = -halfHeight + size * 2.0f;

        float cursor = -halfWidth + size * 2.0f;

        DrawSwatch(frame, cursor, baseY, size, Swatch::Selected, tool == Tool::Select);
        cursor += gap;
        DrawSwatch(frame, cursor, baseY, size, Swatch::Active, tool == Tool::Move);
        cursor += gap;
        DrawSwatch(frame, cursor, baseY, size, Swatch::Warm, tool == Tool::Rotate);
        cursor += gap;
        DrawSwatch(frame, cursor, baseY, size, Swatch::Hull, tool == Tool::Plate);
        cursor += groupGap;

        DrawSwatch(frame, cursor, baseY, size, Swatch::AxisX, activeAxis == 0);
        cursor += gap;
        DrawSwatch(frame, cursor, baseY, size, Swatch::AxisY, activeAxis == 1);
        cursor += gap;
        DrawSwatch(frame, cursor, baseY, size, Swatch::AxisZ, activeAxis == 2);
        cursor += gap;
        DrawSwatch(frame, cursor, baseY, size, Swatch::Snap, localAxis);
        cursor += groupGap;

        DrawSwatch(frame, cursor, baseY, size, Swatch::Warm, snapMode == SnapMode::Free);
        cursor += gap;
        DrawSwatch(frame, cursor, baseY, size, Swatch::AxisZ, snapMode == SnapMode::Grid);
        cursor += gap;
        DrawSwatch(frame, cursor, baseY, size, Swatch::Snap, snapMode == SnapMode::Face);
        cursor += gap;
        DrawSwatch(frame, cursor, baseY, size, Swatch::Active, snapMode == SnapMode::Ends);
        cursor += groupGap;

        DrawSwatch(frame, cursor, baseY, size, Swatch::Ghost, mirrorEnabled);
        cursor += gap;
        DrawSwatch(frame, cursor, baseY, size, Swatch::AxisY, verticalMove);
        cursor += gap;
        DrawSwatch(frame, cursor, baseY, size, Swatch::Hull, fanMode);
        cursor += groupGap;

        for (int index = 0; index < 3; index++)
        {
            DrawSwatch(frame, cursor, baseY, size, Swatch::Snap, index < static_cast<int>(pendingCorners.size()));
            cursor += gap;
        }
    }

    void BeamEditor::DrawSwatch(const HudFrame& frame, float x, float y, float size, Swatch swatch, bool active)
    {
        float height = size;
        Swatch color = swatch;

        if (!active)
        {
            height = size * 0.3f;
            color = Swatch::Inactive;
        }

        const glm::vec3 position = frame.origin + frame.right * x + frame.up * (y + height * 0.5f - size * 0.5f);

        glm::mat4 basis(1.0f);
        basis[0] = glm::vec4(frame.right, 0.0f);
        basis[1] = glm::vec4(frame.up, 0.0f);
        basis[2] = glm::vec4(-frame.forward, 0.0f);

        const glm::mat4 transform = glm::translate(glm::mat4(1.0f), position) * basis * glm::scale(glm::mat4(1.0f), glm::vec3(size, height, size * 0.05f));

        renderer.DrawMesh(boxMesh, GetMaterial(color), transform);
    }

    const Core::Camera& BeamEditor::GetCamera() const
    {
        return camera;
    }

    const std::vector<Beam>& BeamEditor::GetBeams() const
    {
        return beams;
    }

    std::vector<Beam>& BeamEditor::GetBeams()
    {
        return beams;
    }

    const std::vector<Plate>& BeamEditor::GetPlates() const
    {
        return plates;
    }

    void BeamEditor::UpdateStatus()
    {
        std::string toolName = "SELECT";

        if (tool == Tool::Move)
        {
            toolName = "MOVE";
        }
        else if (tool == Tool::Rotate)
        {
            toolName = "ROTATE";
        }
        else if (tool == Tool::Plate)
        {
            toolName = "PLATE";
        }

        std::string axisName = "X";

        if (activeAxis == 1)
        {
            axisName = "Y";
        }
        else if (activeAxis == 2)
        {
            axisName = "Z";
        }

        std::string spaceName = "world";

        if (localAxis)
        {
            spaceName = "local";
        }

        std::string snapName = "free";

        if (snapMode == SnapMode::Grid)
        {
            snapName = "grid";
        }
        else if (snapMode == SnapMode::Face)
        {
            snapName = "face";
        }
        else if (snapMode == SnapMode::Ends)
        {
            snapName = "ends";
        }

        std::string mirrorName = "off";

        if (mirrorEnabled)
        {
            mirrorName = "on";
        }

        std::string fanName = "off";

        if (fanMode)
        {
            fanName = "on";
        }

        char buffer[400];

        std::snprintf(buffer, sizeof(buffer), "%s | axis %s (%s) | snap %s | mirror %s | fan %s | section %.3f | length %.2f | plate %.3f",
            toolName.c_str(), axisName.c_str(), spaceName.c_str(), snapName.c_str(), mirrorName.c_str(), fanName.c_str(), sectionSize, newBeamLength, plateThickness);

        const std::string currentMode = buffer;

        if (modeText != currentMode)
        {
            modeText = currentMode;
            std::cout << currentMode << std::endl;
        }

        char titleBuffer[520];

        if (selectedPlate >= 0)
        {
            std::snprintf(titleBuffer, sizeof(titleBuffer), "Hull | beams %d | plates %d | points %d/3 | %s | plate selected",
                static_cast<int>(beams.size()), static_cast<int>(plates.size()), static_cast<int>(pendingCorners.size()), currentMode.c_str());
        }
        else if (selectedBeam >= 0)
        {
            const Beam& beam = beams[selectedBeam];
            std::snprintf(titleBuffer, sizeof(titleBuffer), "Hull | beams %d | plates %d | points %d/3 | %s | beam %.2f x %.2f x %.2f",
                static_cast<int>(beams.size()), static_cast<int>(plates.size()), static_cast<int>(pendingCorners.size()), currentMode.c_str(), beam.size.x, beam.size.y, beam.size.z);
        }
        else
        {
            std::snprintf(titleBuffer, sizeof(titleBuffer), "Hull | beams %d | plates %d | points %d/3 | %s | no selection",
                static_cast<int>(beams.size()), static_cast<int>(plates.size()), static_cast<int>(pendingCorners.size()), currentMode.c_str());
        }

        if (titleText != titleBuffer)
        {
            titleText = titleBuffer;
            window.SetTitle(titleText);
        }
    }
}