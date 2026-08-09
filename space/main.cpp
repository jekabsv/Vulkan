#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "VulkanAbstraction.h"
#include "DigitEditor.h"
#include "MnistData.h"
#include "NeuralNetwork.h"
#include "NetworkVisualizer.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

// --- Network shape: everything below derives from these four numbers. ---
// 784 = 28x28 MNIST pixels, 10 = digit classes 0-9. The hidden size/depth are picked generous
// enough to actually solve MNIST well (784-128-128-10 lands around 96-98% test accuracy after a
// few epochs); the visualizer never shows all of it at once, only the first N active neurons
// per layer (see kVisibleNeuronsPerLayer below).
constexpr int kInputSize = 784;
constexpr int kHiddenSize = 128;
constexpr int kHiddenDepth = 2;
constexpr int kOutputSize = 10;

// --- Training hyperparameters ---
constexpr int kBatchSize = 32;
constexpr int kEpochs = 6;
constexpr float kLearningRate = 0.1f;

// --- Visualization ---
constexpr int kVisibleNeuronsPerLayer = 25;
constexpr int kRenderEdges = 1;
constexpr float kEdgeRenderPercentage = 0.2f;

// --- Hand-drawn digit editor ---
constexpr float kEditorCanvasWorldSize = 6.0f; // world-space size of the drawing canvas quad while editing
constexpr int kSavedDigitSlots = 9; // keys 1-9

namespace
{
    // Expected under space/data/ (working directory = project directory when run from Visual
    // Studio): train-images-idx3-ubyte, train-labels-idx1-ubyte, t10k-images-idx3-ubyte,
    // t10k-labels-idx1-ubyte (the standard MNIST files, get them from any MNIST mirror). If
    // they're missing, a synthetic placeholder dataset is used instead so the pipeline still runs.
    const std::string kDataDirectory = "data";
    const std::string kWeightsPath = "data/mnist_weights.bin";

    NN::Dataset LoadOrSynthesizeDataset(const std::string& directory, const std::string& split, int sampleCount, unsigned seed)
    {
        NN::Dataset dataset = NN::MnistLoader::LoadSplit(directory, split);

        if (dataset.Empty() || dataset.imageWidth * dataset.imageHeight != kInputSize)
        {
            std::cerr << "[MNIST] Could not find/read '" << split << "' files under '" << directory
                << "' - using a synthetic placeholder dataset instead. Drop the real MNIST idx-ubyte "
                << "files there for real results." << std::endl;
            return NN::MnistLoader::MakeSyntheticDataset(sampleCount, 28, 28, seed);
        }

        return dataset;
    }

    void TrainNetwork(NN::NeuralNetwork& network, const NN::Dataset& trainSet, const NN::Dataset& testSet)
    {
        std::vector<int> indices(trainSet.Size());
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 shuffleRng(42);

        std::cout << "[Train] " << trainSet.Size() << " training samples, " << testSet.Size() << " test samples." << std::endl;

        for (int epoch = 0; epoch < kEpochs; epoch++)
        {
            std::shuffle(indices.begin(), indices.end(), shuffleRng);

            double totalLoss = 0.0;

            for (size_t start = 0; start < indices.size(); start += kBatchSize)
            {
                const size_t end = std::min(start + kBatchSize, indices.size());

                network.ZeroGradients();

                for (size_t k = start; k < end; k++)
                {
                    const int sampleIndex = indices[k];
                    totalLoss += network.AccumulateGradients(trainSet.images[sampleIndex], NN::OneHot(trainSet.labels[sampleIndex], kOutputSize));
                }

                network.ApplyGradients(kLearningRate, static_cast<int>(end - start));
            }

            int correct = 0;
            for (size_t i = 0; i < testSet.Size(); i++)
            {
                if (network.Predict(testSet.images[i]) == testSet.labels[i])
                {
                    correct++;
                }
            }

            const double accuracy = testSet.Empty() ? 0.0 : 100.0 * static_cast<double>(correct) / static_cast<double>(testSet.Size());
            const double averageLoss = indices.empty() ? 0.0 : totalLoss / static_cast<double>(indices.size());

            std::cout << "[Train] epoch " << (epoch + 1) << "/" << kEpochs
                << " - avg loss: " << averageLoss
                << " - test accuracy: " << accuracy << "%" << std::endl;
        }
    }

    std::string FormatFloat(float value, int decimals)
    {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.*f", decimals, value);
        return std::string(buffer);
    }

    enum class AppMode
    {
        Visualizing,
        Editing
    };
}

int main()
{
    try
    {
        // ---- Phase 1: data + network. Pure console work, done before opening a window so
        // training doesn't leave an unresponsive GUI sitting on screen for the minute or two it takes. ----
        const NN::Dataset trainSet = LoadOrSynthesizeDataset(kDataDirectory, "train", 4000, 1234u);
        const NN::Dataset testSet = LoadOrSynthesizeDataset(kDataDirectory, "t10k", 800, 5678u);

        NN::NeuralNetwork network(kInputSize, kHiddenSize, kHiddenDepth, kOutputSize);

        if (network.LoadWeights(kWeightsPath))
        {
            std::cout << "[Train] Loaded cached weights from " << kWeightsPath << std::endl;
        }
        else
        {
            TrainNetwork(network, trainSet, testSet);
            if (network.SaveWeights(kWeightsPath))
            {
                std::cout << "[Train] Saved weights to " << kWeightsPath << std::endl;
            }
        }

        // ---- Phase 2: visualization ----
        Core::Window window(1280, 720, "Neural Network Visualizer");
        Core::VulkanContext context(window);
        Core::Renderer renderer(context, window);
        Core::AssetManager assets(context, renderer);

        assets.LoadShader("triangle_vert", "vert.spv");
        assets.LoadShader("triangle_frag", "frag.spv");
        assets.LoadShader("text_instanced_vert", "text_instanced_vert.spv");
        assets.LoadShader("text_instanced_frag", "text_instanced_frag.spv");

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

        assets.SetMesh("cube", Core::Mesh::CreateCube(context));
        const Core::Mesh& cubeMesh = assets.GetMesh("cube");

        assets.SetMesh("quad", Core::Mesh::CreateQuad(context));
        const Core::Mesh& quadMesh = assets.GetMesh("quad");

        Core::TextureConfig whiteTextureConfig;
        whiteTextureConfig.width = 1;
        whiteTextureConfig.height = 1;
        whiteTextureConfig.format = VK_FORMAT_R8G8B8A8_SRGB;
        whiteTextureConfig.generateMipmaps = false;
        const std::vector<uint8_t> whitePixels = { 255, 255, 255, 255 };
        assets.SetTexture("white", Core::Texture(context, whiteTextureConfig, whitePixels.data(), static_cast<VkDeviceSize>(whitePixels.size())));

        Core::Material& nodeMaterial = assets.SetMaterial("node", renderer.CreateMaterial(assets.GetPipeline("pbr")));
        nodeMaterial.SetVec3("u_AlbedoColor", glm::vec3(1.0f));
        nodeMaterial.SetFloat("u_Roughness", 1.0f);
        nodeMaterial.SetFloat("u_Metallic", 0.0f);
        nodeMaterial.SetVec2("u_Tiling", glm::vec2(1.0f, 1.0f));
        nodeMaterial.SetTexture("u_AlbedoMap", assets.GetTexture("white"));

        Core::Material& edgeMaterial = assets.SetMaterial("edge", renderer.CreateMaterial(assets.GetPipeline("pbr")));
        edgeMaterial.SetVec3("u_AlbedoColor", glm::vec3(1.0f));
        edgeMaterial.SetFloat("u_Roughness", 1.0f);
        edgeMaterial.SetFloat("u_Metallic", 0.0f);
        edgeMaterial.SetVec2("u_Tiling", glm::vec2(1.0f, 1.0f));
        edgeMaterial.SetTexture("u_AlbedoMap", assets.GetTexture("white"));

        // --- Input image: the network only ever sees an image as a flat 784-float vector, so
        // this reconstructs it into an actual 28x28 texture and displays it on a quad in front of
        // the input layer. Always shown in MNIST's own convention (black background, white
        // strokes) - whether the source is a real test sample or a hand-drawn digit that's just
        // been inverted from the editor's paper convention (see ApplySample below).
        Core::TextureConfig inputImageTextureConfig;
        inputImageTextureConfig.width = static_cast<uint32_t>(testSet.imageWidth);
        inputImageTextureConfig.height = static_cast<uint32_t>(testSet.imageHeight);
        inputImageTextureConfig.format = VK_FORMAT_R8G8B8A8_SRGB;
        inputImageTextureConfig.generateMipmaps = false;
        inputImageTextureConfig.sampler.magFilter = VK_FILTER_NEAREST; // keep individual MNIST pixels crisp/blocky instead of blurred
        inputImageTextureConfig.sampler.minFilter = VK_FILTER_NEAREST;
        inputImageTextureConfig.sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        inputImageTextureConfig.sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

        const std::vector<uint8_t> initialImagePixels = NN::ToGrayscaleRGBA(testSet.images[0]);
        Core::Texture& inputImageTexture = assets.SetTexture("input_image",
            Core::Texture(context, inputImageTextureConfig, initialImagePixels.data(), static_cast<VkDeviceSize>(initialImagePixels.size())));

        Core::Material& inputImageMaterial = assets.SetMaterial("input_image", renderer.CreateMaterial(assets.GetPipeline("pbr")));
        inputImageMaterial.SetVec3("u_AlbedoColor", glm::vec3(1.0f));
        inputImageMaterial.SetFloat("u_Roughness", 1.0f);
        inputImageMaterial.SetFloat("u_Metallic", 0.0f);
        inputImageMaterial.SetVec2("u_Tiling", glm::vec2(1.0f, 1.0f));
        inputImageMaterial.SetTexture("u_AlbedoMap", inputImageTexture);

        constexpr float kInputImageOffsetX = 1.6f; // distance to the left of the input layer (x=0)
        constexpr float kInputImageSize = 1.6f;    // world-space width/height of the displayed square

        // --- Hand-drawn digit editor: a 28x28 canvas quad shown in its own dedicated camera view
        // (swapped in for the whole screen while editing, same textured-quad trick as the input
        // image above), plus 9 save slots. Uses the same texture-update pattern as inputImageTexture.
        NN::DigitEditor editor;

        Core::TextureConfig editorCanvasTextureConfig;
        editorCanvasTextureConfig.width = static_cast<uint32_t>(NN::DigitEditor::kGridSize);
        editorCanvasTextureConfig.height = static_cast<uint32_t>(NN::DigitEditor::kGridSize);
        editorCanvasTextureConfig.format = VK_FORMAT_R8G8B8A8_SRGB;
        editorCanvasTextureConfig.generateMipmaps = false;
        editorCanvasTextureConfig.sampler.magFilter = VK_FILTER_NEAREST;
        editorCanvasTextureConfig.sampler.minFilter = VK_FILTER_NEAREST;
        editorCanvasTextureConfig.sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        editorCanvasTextureConfig.sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

        const std::vector<uint8_t> initialCanvasPixels = NN::ToGrayscaleRGBA(editor.GetCanvas());
        Core::Texture& editorCanvasTexture = assets.SetTexture("editor_canvas",
            Core::Texture(context, editorCanvasTextureConfig, initialCanvasPixels.data(), static_cast<VkDeviceSize>(initialCanvasPixels.size())));

        Core::Material& editorCanvasMaterial = assets.SetMaterial("editor_canvas", renderer.CreateMaterial(assets.GetPipeline("pbr")));
        editorCanvasMaterial.SetVec3("u_AlbedoColor", glm::vec3(1.0f));
        editorCanvasMaterial.SetFloat("u_Roughness", 1.0f);
        editorCanvasMaterial.SetFloat("u_Metallic", 0.0f);
        editorCanvasMaterial.SetVec2("u_Tiling", glm::vec2(1.0f, 1.0f));
        editorCanvasMaterial.SetTexture("u_AlbedoMap", editorCanvasTexture);

        // All 9 slots start "empty white" - just DigitEditor's own blank canvas state, copied in.
        std::array<std::vector<float>, kSavedDigitSlots> savedDigits;
        for (std::vector<float>& slot : savedDigits)
        {
            slot = editor.GetCanvas();
        }

        AppMode appMode = AppMode::Visualizing;
        bool insertModeActive = false;
        bool canvasDirty = false;

        // A perspective camera aimed straight down -Z at a plane still gives a purely linear
        // (affine) mapping between screen pixels and world XY on that one plane - no need for a
        // dedicated orthographic camera type just for this.
        const float kEditorFovRadians = glm::radians(50.0f);
        const float editorHalfHeight = kEditorCanvasWorldSize * 0.5f * 1.15f; // 15% margin around the canvas
        const float editorCameraDistance = editorHalfHeight / std::tan(kEditorFovRadians * 0.5f);

        Core::Camera editorCamera;
        editorCamera.SetPosition(glm::vec3(0.0f, 0.0f, editorCameraDistance));
        editorCamera.LookAt(glm::vec3(0.0f, 0.0f, 0.0f));

        auto ScreenToCanvasGridPos = [&](const glm::vec2& mousePos, float& outX, float& outY)
            {
                const float aspect = renderer.GetAspectRatio();
                const float halfHeightAtPlane = editorCameraDistance * std::tan(kEditorFovRadians * 0.5f);
                const float halfWidthAtPlane = halfHeightAtPlane * aspect;

                const float ndcX = (mousePos.x / static_cast<float>(window.GetWidth())) * 2.0f - 1.0f;
                const float ndcY = 1.0f - (mousePos.y / static_cast<float>(window.GetHeight())) * 2.0f;

                const float worldX = ndcX * halfWidthAtPlane;
                const float worldY = ndcY * halfHeightAtPlane;

                outX = ((worldX / kEditorCanvasWorldSize) + 0.5f) * static_cast<float>(NN::DigitEditor::kGridSize);
                outY = (0.5f - (worldY / kEditorCanvasWorldSize)) * static_cast<float>(NN::DigitEditor::kGridSize);
            };

        const float fontPixelHeight = 48.0f;
        Core::Font& textFont = assets.LoadFont("main", assets.GetPipeline("text"), "font.ttf", fontPixelHeight);
        const float textScale = 0.5f / fontPixelHeight;

        NN::VisualizerConfig visualizerConfig;
        visualizerConfig.visibleNeuronsPerLayer = kVisibleNeuronsPerLayer;
        visualizerConfig.renderEdges = (kRenderEdges != 0);
        visualizerConfig.edgeRenderPercentage = kEdgeRenderPercentage;

        NN::NetworkLayout layout;
        NN::PropagationAnimator animator;

        std::mt19937 sampleRng(std::random_device{}());
        std::uniform_int_distribution<size_t> sampleDist(0, std::max<size_t>(1, testSet.Size()) - 1);

        auto ApplySample = [&](const std::vector<float>& image, std::optional<uint8_t> trueLabel)
            {
                network.Forward(image);
                layout.Build(network, visualizerConfig);
                animator.Begin(network, layout, trueLabel, visualizerConfig);

                const std::vector<uint8_t> imagePixels = NN::ToGrayscaleRGBA(image);
                inputImageTexture.SetData(imagePixels.data(), static_cast<VkDeviceSize>(imagePixels.size()));
            };

        auto PickNewSample = [&]()
            {
                const size_t sampleIndex = sampleDist(sampleRng);
                ApplySample(testSet.images[sampleIndex], testSet.labels[sampleIndex]);
            };

        auto FeedSavedDigit = [&](int slotIndex)
            {
                ApplySample(NN::InvertPaperToMnist(savedDigits[slotIndex]), std::nullopt);
            };

        PickNewSample();

        const float totalWidth = layout.TotalWidth();
        const float halfHeight = static_cast<float>(kVisibleNeuronsPerLayer - 1) * 0.5f * visualizerConfig.neuronSpacing + 1.0f;
        const float cameraDistance = std::max(totalWidth, halfHeight * 2.0f) * 0.9f + 3.0f;

        Core::Camera mainCamera;
        mainCamera.SetPosition(glm::vec3(totalWidth * 0.5f, 0.3f, cameraDistance));
        mainCamera.LookAt(glm::vec3(totalWidth * 0.5f, 0.3f, 0.0f));

        // Fixed units/second movement doesn't work across scene sizes: fast enough to close a
        // large kVisibleNeuronsPerLayer's camera distance makes close-up inspection overshoot
        // wildly, slow enough for close-up precision takes forever to cross a big scene. Scaling
        // speed by current distance from the network's center fixes both at once (same trick as
        // most 3D editors' fly-cams): far away covers ground fast, and it naturally slows to a
        // crawl as you approach, right when precision matters.
        const glm::vec3 networkCenter(totalWidth * 0.5f, 0.3f, 0.0f);
        constexpr float kCameraMoveSpeedFactor = 1.5f; // fraction of current distance-to-center moved per second
        constexpr float kCameraMinMoveSpeed = 0.5f;    // units/second floor so you can still creep in at point-blank range

        auto lastFrameTime = std::chrono::high_resolution_clock::now();
        glm::vec2 lastMousePos = window.GetMousePos();

        while (!window.ShouldClose())
        {
            window.PollEvents();

            if (window.IsKeyPressed(Core::KeyCode::Escape))
            {
                window.Close();
                continue;
            }

            // Which 1-9 key (if any) was pressed this frame - shared by both modes below, each
            // interpreting it differently.
            int pressedDigit = 0;
            for (int n = 1; n <= 9; n++)
            {
                const Core::KeyCode code = static_cast<Core::KeyCode>(static_cast<uint16_t>(Core::KeyCode::Num0) + n);
                if (window.IsKeyPressed(code))
                {
                    pressedDigit = n;
                    break;
                }
            }

            const auto now = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(now - lastFrameTime).count();
            lastFrameTime = now;
            deltaTime = std::min(deltaTime, 0.1f);

            if (appMode == AppMode::Editing)
            {
                if (window.IsKeyPressed(Core::KeyCode::LeftBracket)) editor.ShrinkBrush();
                if (window.IsKeyPressed(Core::KeyCode::RightBracket)) editor.GrowBrush();

                if (window.IsMouseButtonDown(Core::MouseButton::Left))
                {
                    float gridX = 0.0f;
                    float gridY = 0.0f;
                    ScreenToCanvasGridPos(window.GetMousePos(), gridX, gridY);
                    editor.PaintAt(gridX, gridY);
                    canvasDirty = true;
                }

                // Second press while editing: save to that slot and exit back to visualizing.
                if (pressedDigit != 0)
                {
                    savedDigits[pressedDigit - 1] = editor.GetCanvas();
                    appMode = AppMode::Visualizing;
                }
            }
            else // AppMode::Visualizing
            {
                if (window.IsKeyPressed(Core::KeyCode::I))
                {
                    insertModeActive = !insertModeActive;
                }

                if (window.IsKeyPressed(Core::KeyCode::Space))
                {
                    PickNewSample();
                }

                if (pressedDigit != 0)
                {
                    if (insertModeActive)
                    {
                        FeedSavedDigit(pressedDigit - 1);
                        insertModeActive = false;
                    }
                    else
                    {
                        editor.Clear();
                        canvasDirty = true;
                        appMode = AppMode::Editing;
                    }
                }

                const float distanceToCenter = glm::length(mainCamera.GetPosition() - networkCenter);
                const float moveSpeed = std::max(kCameraMinMoveSpeed, distanceToCenter * kCameraMoveSpeedFactor);
                const float moveStep = moveSpeed * deltaTime;

                if (window.IsKeyDown(Core::KeyCode::W)) mainCamera.MoveForward(moveStep);
                if (window.IsKeyDown(Core::KeyCode::S)) mainCamera.MoveForward(-moveStep);
                if (window.IsKeyDown(Core::KeyCode::A)) mainCamera.MoveRight(-moveStep);
                if (window.IsKeyDown(Core::KeyCode::D)) mainCamera.MoveRight(moveStep);
                if (window.IsKeyDown(Core::KeyCode::E)) mainCamera.MoveWorld(glm::vec3(0.0f, moveStep, 0.0f));
                if (window.IsKeyDown(Core::KeyCode::Q)) mainCamera.MoveWorld(glm::vec3(0.0f, -moveStep, 0.0f));

                if (window.IsMouseButtonDown(Core::MouseButton::Right))
                {
                    const glm::vec2 mousePos = window.GetMousePos();
                    const glm::vec2 mouseDelta = mousePos - lastMousePos;

                    constexpr float mouseSensitivity = 0.0025f;
                    mainCamera.Yaw(-mouseDelta.x * mouseSensitivity);
                    mainCamera.Pitch(-mouseDelta.y * mouseSensitivity);
                }

                animator.Update(deltaTime);
            }

            const glm::vec2 mousePos = window.GetMousePos();
            lastMousePos = mousePos;

            if (renderer.BeginFrame())
            {
                renderer.BeginRenderPass(glm::vec4(0.05f, 0.05f, 0.07f, 1.0f));

                if (appMode == AppMode::Editing)
                {
                    if (canvasDirty)
                    {
                        const std::vector<uint8_t> canvasPixels = NN::ToGrayscaleRGBA(editor.GetCanvas());
                        editorCanvasTexture.SetData(canvasPixels.data(), static_cast<VkDeviceSize>(canvasPixels.size()));
                        canvasDirty = false;
                    }

                    editorCamera.SetPerspective(kEditorFovRadians, renderer.GetAspectRatio(), 0.1f, 100.0f);
                    renderer.SetCamera(editorCamera);

                    const glm::mat4 canvasTransform = glm::scale(glm::mat4(1.0f), glm::vec3(kEditorCanvasWorldSize, kEditorCanvasWorldSize, 1.0f));
                    renderer.DrawMesh(quadMesh, editorCanvasMaterial, canvasTransform);

                    renderer.BeginBatch();
                    renderer.SubmitText(textFont, "Drawing - pen size " + FormatFloat(editor.GetBrushRadius(), 1) + "  ([ / ] to resize)",
                        glm::vec3(-kEditorCanvasWorldSize * 0.5f, kEditorCanvasWorldSize * 0.5f + 0.2f, 0.0f), textScale * 0.7f, glm::vec4(1.0f));
                    renderer.SubmitText(textFont, "press 1-9 to save this drawing and exit",
                        glm::vec3(-kEditorCanvasWorldSize * 0.5f, -kEditorCanvasWorldSize * 0.5f - 0.2f, 0.0f), textScale * 0.65f,
                        glm::vec4(0.85f, 0.85f, 0.9f, 1.0f));
                    renderer.FlushBatch();
                }
                else
                {
                    mainCamera.SetPerspective(glm::radians(50.0f), renderer.GetAspectRatio(), 0.1f, 100.0f);
                    renderer.SetCamera(mainCamera);

                    renderer.BeginBatch();

                    const glm::mat4 imageTransform = glm::scale(
                        glm::translate(glm::mat4(1.0f), glm::vec3(-kInputImageOffsetX, 0.0f, 0.0f)),
                        glm::vec3(kInputImageSize, kInputImageSize, 1.0f));
                    renderer.DrawMesh(quadMesh, inputImageMaterial, imageTransform);
                    renderer.SubmitText(textFont, "input image", glm::vec3(-kInputImageOffsetX - kInputImageSize * 0.5f, -kInputImageSize * 0.5f - 0.25f, 0.0f),
                        textScale * 0.85f, glm::vec4(0.85f, 0.85f, 0.9f, 1.0f));

                    if (insertModeActive)
                    {
                        renderer.SubmitText(textFont, "INSERT MODE - press 1-9 to load a saved drawing (I to cancel)",
                            glm::vec3(-kInputImageOffsetX - kInputImageSize * 0.5f, kInputImageSize * 0.5f + 0.4f, 0.0f),
                            textScale * 0.85f, glm::vec4(1.0f, 0.9f, 0.3f, 1.0f));
                    }

                    animator.Draw(renderer, nodeMaterial, edgeMaterial, cubeMesh, textFont, textScale);
                    renderer.FlushBatch();
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
