#pragma once

#include "Window.h"
#include "VulkanContext.h"
#include "Renderer.h"
#include "Pipeline.h"
#include "Mesh.h"
#include "Material.h"
#include "Texture.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <chrono>
#include <string>
#include <vector>

namespace Editor
{
    struct Beam
    {
        glm::vec3 center = glm::vec3(0.0f);
        glm::quat orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        glm::vec3 size = glm::vec3(1.0f);
    };

    struct Anchor
    {
        int beamIndex = -1;
        glm::vec3 localPoint = glm::vec3(0.0f);
        glm::vec3 worldPoint = glm::vec3(0.0f);
    };

    struct Plate
    {
        Anchor corners[3];
        float thickness = 0.06f;
    };

    struct Ray
    {
        glm::vec3 origin = glm::vec3(0.0f);
        glm::vec3 direction = glm::vec3(0.0f, 0.0f, -1.0f);
    };

    struct FaceHit
    {
        int beamIndex = -1;
        int axis = 1;
        float sign = 1.0f;
        float distance = 0.0f;
        glm::vec3 point = glm::vec3(0.0f);
    };

    struct Placement
    {
        bool valid = false;
        bool onFace = false;
        int beamIndex = -1;
        glm::vec3 point = glm::vec3(0.0f);
        glm::quat orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        int axis = 0;
        float sign = 1.0f;
    };

    struct KeyRepeat
    {
        float timer = 0.0f;
        bool active = false;
    };

    struct HudFrame
    {
        glm::vec3 origin = glm::vec3(0.0f);
        glm::vec3 right = glm::vec3(1.0f, 0.0f, 0.0f);
        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
        glm::vec3 forward = glm::vec3(0.0f, 0.0f, -1.0f);
    };

    enum class Tool
    {
        Select,
        Move,
        Rotate,
        Plate
    };

    enum class SnapMode
    {
        Free,
        Grid,
        Face,
        Ends
    };

    enum class DragMode
    {
        None,
        Resize,
        Move,
        Rotate
    };

    enum class Swatch
    {
        Beam,
        Selected,
        Snap,
        Ground,
        Ghost,
        Mirror,
        Hull,
        AxisX,
        AxisY,
        AxisZ,
        Inactive,
        Active,
        Warm,
        Count
    };

    class BeamEditor
    {
    public:
        BeamEditor(Core::VulkanContext& context, Core::Window& window, Core::Renderer& renderer, const Core::GraphicsPipeline& pipeline);

        BeamEditor(const BeamEditor&) = delete;
        BeamEditor& operator=(const BeamEditor&) = delete;

        void Update();
        void Render();

        const Core::Camera& GetCamera() const;
        const std::vector<Beam>& GetBeams() const;
        std::vector<Beam>& GetBeams();
        const std::vector<Plate>& GetPlates() const;

    private:
        void UpdateCameraInput();
        void UpdateCameraMatrices();
        void UpdateHover();
        void UpdateDrag();
        void UpdateModeKeys();
        void UpdateActionKeys(float deltaTime);
        void UpdateStatus();

        void BeginDrag();
        void EndDrag();
        void ApplyResize(const Ray& ray);
        void ApplyMove(const Ray& ray);
        void ApplyRotateDrag();

        void RotateSelected(float angle);
        void NudgeSelected(const glm::vec3& direction, float amount);
        void AddBeamAtCursor();
        void DuplicateSelected();
        void DeleteSelection();
        void DeleteBeam(int index);
        void DeletePlate(int index);
        void CycleSelection();
        void FocusSelection();
        void BakeMirror();
        void SetSizePreset(int index);

        void AddPlatePoint();
        void RemoveLastPlatePoint();
        void CreatePlate();

        Ray MakeMouseRay() const;
        bool PickFace(const Ray& ray, FaceHit& outHit) const;
        bool PickPlate(const Ray& ray, int& outIndex, float& outDistance) const;
        bool RayHitsBeam(const Ray& ray, const Beam& beam, float& outDistance, int& outAxis, float& outSign) const;
        Placement ComputePlacement(const Ray& ray) const;
        Beam MakePlacedBeam(const Placement& source) const;
        Beam MakeMirrored(const Beam& beam) const;

        Anchor MakeAnchor(const Placement& source) const;
        glm::vec3 ResolveAnchor(const Anchor& anchor) const;
        void ResolvePlate(const Plate& plate, glm::vec3* outCorners) const;
        glm::mat4 MakePlateTransform(const glm::vec3* corners, float thickness) const;

        glm::mat4 MakeTransform(const Beam& beam) const;
        glm::vec3 FaceNormal(const Beam& beam, int axis, float sign) const;
        glm::vec3 FaceCenter(const Beam& beam, int axis, float sign) const;
        glm::vec3 ActiveAxisDirection() const;

        bool ConsumeKey(Core::KeyCode key, KeyRepeat& repeat, float deltaTime);
        Core::Material& GetMaterial(Swatch swatch);

        void DrawBeam(const Beam& beam, Swatch swatch);
        void DrawPlate(const glm::vec3* corners, float thickness, Swatch swatch);
        void DrawMarker(const glm::vec3& position, float size, Swatch swatch);
        void DrawHud();
        void DrawSwatch(const HudFrame& frame, float x, float y, float size, Swatch swatch, bool active);

        Core::Window& window;
        Core::Renderer& renderer;
        const Core::GraphicsPipeline& pipeline;

        Core::Mesh boxMesh;
        Core::Mesh prismMesh;
        Core::Texture whiteTexture;
        std::vector<Core::Material> palette;

        std::vector<Beam> beams;
        std::vector<Plate> plates;
        std::vector<Anchor> pendingCorners;

        int selectedBeam = -1;
        int selectedPlate = -1;
        FaceHit hoveredFace;
        Placement placement;

        Core::Camera camera;
        glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
        float cameraDistance = 12.0f;
        float cameraYaw = 0.9f;
        float cameraPitch = 0.45f;

        glm::vec2 previousMouse = glm::vec2(0.0f);
        glm::vec2 mouseDelta = glm::vec2(0.0f);

        Tool tool = Tool::Select;
        SnapMode snapMode = SnapMode::Ends;
        DragMode dragMode = DragMode::None;

        int activeAxis = 2;
        bool localAxis = true;
        bool mirrorEnabled = true;
        bool verticalMove = false;
        bool fanMode = true;

        float sectionSize = 0.25f;
        float newBeamLength = 2.0f;
        float plateThickness = 0.06f;

        FaceHit dragFace;
        glm::vec3 dragAxis = glm::vec3(0.0f, 1.0f, 0.0f);
        glm::vec3 dragAnchor = glm::vec3(0.0f);
        glm::vec3 dragOffset = glm::vec3(0.0f);
        glm::vec3 dragStartCenter = glm::vec3(0.0f);
        glm::quat dragStartOrientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        float dragStartParameter = 0.0f;
        float dragStartLength = 1.0f;
        float dragStartMouseX = 0.0f;

        KeyRepeat repeatLeft;
        KeyRepeat repeatRight;
        KeyRepeat repeatUp;
        KeyRepeat repeatDown;

        std::chrono::steady_clock::time_point lastTime;
        std::string titleText;
        std::string modeText;
    };
}