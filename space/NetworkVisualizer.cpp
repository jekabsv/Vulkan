#include "NetworkVisualizer.h"

#include "NeuralNetwork.h"

#include "Font.h"
#include "Material.h"
#include "Mesh.h"
#include "Renderer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace NN
{

    namespace
    {
        std::string FormatFloat(float value, int decimals)
        {
            char buffer[32];
            std::snprintf(buffer, sizeof(buffer), "%.*f", decimals, value);
            return std::string(buffer);
        }

        // Saturating diverging color: negative weights read blue->red is a common choice, but
        // red/green is used elsewhere in this file for correctness, so weights use blue (negative)
        // to orange (positive) instead to stay visually distinct from the correct/incorrect cues.
        glm::vec3 WeightColor(float weight)
        {
            const float t = glm::clamp(std::abs(weight) / (std::abs(weight) + 1.0f), 0.0f, 1.0f);
            const glm::vec3 negative(0.85f, 0.3f, 0.3f);
            const glm::vec3 positive(0.3f, 0.55f, 0.95f);
            const glm::vec3 base = weight >= 0.0f ? positive : negative;
            return glm::mix(glm::vec3(0.16f, 0.16f, 0.2f), base, t);
        }

        glm::quat RotationBetween(const glm::vec3& from, const glm::vec3& to)
        {
            const glm::vec3 f = glm::normalize(from);
            const glm::vec3 t = glm::normalize(to);
            const float d = glm::clamp(glm::dot(f, t), -1.0f, 1.0f);

            if (d > 0.9999f)
            {
                return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
            }

            if (d < -0.9999f)
            {
                glm::vec3 axis = glm::cross(glm::vec3(1.0f, 0.0f, 0.0f), f);
                if (glm::dot(axis, axis) < 1e-6f)
                {
                    axis = glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), f);
                }
                axis = glm::normalize(axis);
                return glm::angleAxis(3.14159265f, axis);
            }

            const glm::vec3 axis = glm::normalize(glm::cross(f, t));
            const float angle = std::acos(d);
            return glm::angleAxis(angle, axis);
        }

        // Deterministic keep/skip decision for one edge, used to thin out edgeRenderPercentage of
        // edges without any extra per-edge state or per-frame RNG: it's a pure hash of the edge's
        // identity, so the same edges are kept every frame (no flicker) and scattered roughly
        // uniformly across the layer rather than always dropping a contiguous block of them.
        bool ShouldRenderEdge(int transition, int fromVisible, int toVisible, float percentage)
        {
            if (percentage >= 1.0f) return true;
            if (percentage <= 0.0f) return false;

            uint32_t h = static_cast<uint32_t>(transition) * 73856093u
                ^ static_cast<uint32_t>(fromVisible) * 19349663u
                ^ static_cast<uint32_t>(toVisible) * 83492791u;
            h ^= h >> 13;
            h *= 0x5bd1e995u;
            h ^= h >> 15;

            constexpr uint32_t kBucketCount = 100000u;
            const float normalized = static_cast<float>(h % kBucketCount) / static_cast<float>(kBucketCount);
            return normalized < percentage;
        }

        glm::mat4 MakeBeamTransform(const glm::vec3& from, const glm::vec3& to, float thickness)
        {
            const glm::vec3 direction = to - from;
            const float length = glm::length(direction);

            if (length < 1e-5f)
            {
                return glm::translate(glm::mat4(1.0f), from);
            }

            const glm::vec3 midpoint = (from + to) * 0.5f;
            const glm::quat rotation = RotationBetween(glm::vec3(0.0f, 0.0f, 1.0f), direction / length);

            glm::mat4 transform = glm::translate(glm::mat4(1.0f), midpoint);
            transform *= glm::mat4_cast(rotation);
            transform = glm::scale(transform, glm::vec3(thickness, thickness, length));
            return transform;
        }
    }

    void NetworkLayout::Build(const NeuralNetwork& network, const VisualizerConfig& config)
    {
        m_Layers.clear();
        m_Edges.clear();
        m_LayerSpacing = config.layerSpacing;

        const int transitionCount = network.GetLayerCount();
        const int activationLayerCount = transitionCount + 1;
        const std::vector<std::vector<float>>& activations = network.GetActivations();

        m_Layers.resize(activationLayerCount);

        for (int layer = 0; layer < activationLayerCount; layer++)
        {
            const bool isOutputLayer = (layer == activationLayerCount - 1);
            const bool isInputLayer = (layer == 0);
            const std::vector<float>& layerActivations = activations[layer];

            std::vector<int> chosenIndices;

            if (isOutputLayer)
            {
                chosenIndices.reserve(layerActivations.size());
                for (int i = 0; i < static_cast<int>(layerActivations.size()); i++)
                {
                    chosenIndices.push_back(i);
                }
            }
            else
            {
                for (int i = 0; i < static_cast<int>(layerActivations.size())
                    && static_cast<int>(chosenIndices.size()) < config.visibleNeuronsPerLayer; i++)
                {
                    const float value = layerActivations[i];
                    const bool active = isInputLayer ? (value > config.activeThreshold) : (value > 0.0f);
                    if (active)
                    {
                        chosenIndices.push_back(i);
                    }
                }

                // Fallback so a dead/blank layer still shows something instead of being empty.
                if (chosenIndices.empty())
                {
                    for (int i = 0; i < static_cast<int>(layerActivations.size())
                        && static_cast<int>(chosenIndices.size()) < config.visibleNeuronsPerLayer; i++)
                    {
                        chosenIndices.push_back(i);
                    }
                }
            }

            const float x = static_cast<float>(layer) * config.layerSpacing;
            const int count = static_cast<int>(chosenIndices.size());
            const float startY = -static_cast<float>(count - 1) * 0.5f * config.neuronSpacing;

            std::vector<VisibleNeuron>& visibleLayer = m_Layers[layer];
            visibleLayer.resize(count);

            for (int i = 0; i < count; i++)
            {
                visibleLayer[i].neuronIndex = chosenIndices[i];
                visibleLayer[i].position = glm::vec3(x, startY + static_cast<float>(i) * config.neuronSpacing, 0.0f);
            }
        }

        m_Edges.resize(transitionCount);

        for (int transition = 0; transition < transitionCount; transition++)
        {
            const std::vector<VisibleNeuron>& fromLayer = m_Layers[transition];
            const std::vector<VisibleNeuron>& toLayer = m_Layers[transition + 1];

            std::vector<VisibleEdge>& edges = m_Edges[transition];
            edges.reserve(fromLayer.size() * toLayer.size());

            for (int f = 0; f < static_cast<int>(fromLayer.size()); f++)
            {
                for (int t = 0; t < static_cast<int>(toLayer.size()); t++)
                {
                    edges.push_back(VisibleEdge{ f, t });
                }
            }
        }
    }

    float NetworkLayout::TotalWidth() const
    {
        return static_cast<float>(std::max(0, LayerCount() - 1)) * m_LayerSpacing;
    }

    void PropagationAnimator::Begin(const NeuralNetwork& network, const NetworkLayout& layout, std::optional<uint8_t> trueLabel, const VisualizerConfig& config)
    {
        m_Network = &network;
        m_Layout = &layout;
        m_Config = config;
        m_TrueLabel = trueLabel;
        m_TransitionCount = network.GetLayerCount();

        m_CurrentTransition = 0;
        m_Phase = PropagationPhase::Travel;
        m_PhaseTimer = 0.0f;

        m_LayerState.assign(layout.LayerCount(), LayerState::NotReached);
        if (!m_LayerState.empty())
        {
            m_LayerState[0] = LayerState::Active;
        }
    }

    void PropagationAnimator::Update(float deltaTime)
    {
        if (m_Phase == PropagationPhase::Idle || m_Network == nullptr)
        {
            return;
        }

        m_PhaseTimer += deltaTime;

        switch (m_Phase)
        {
        case PropagationPhase::Travel:
            if (m_PhaseTimer >= m_Config.travelDuration)
            {
                m_PhaseTimer = 0.0f;
                m_Phase = PropagationPhase::Arrive;
                m_LayerState[m_CurrentTransition + 1] = LayerState::Arriving;
            }
            break;

        case PropagationPhase::Arrive:
            if (m_PhaseTimer >= m_Config.arriveDuration)
            {
                m_PhaseTimer = 0.0f;
                m_Phase = PropagationPhase::Activate;
            }
            break;

        case PropagationPhase::Activate:
            if (m_PhaseTimer >= m_Config.activateDuration)
            {
                m_PhaseTimer = 0.0f;
                m_LayerState[m_CurrentTransition + 1] = LayerState::Active;
                m_LayerState[m_CurrentTransition] = LayerState::Dimmed;

                m_CurrentTransition++;

                if (m_CurrentTransition >= m_TransitionCount)
                {
                    m_Phase = PropagationPhase::HoldOutput;
                }
                else
                {
                    m_Phase = PropagationPhase::Travel;
                }
            }
            break;

        case PropagationPhase::HoldOutput:
            if (m_PhaseTimer >= m_Config.outputHoldDuration)
            {
                m_Phase = PropagationPhase::Idle;
            }
            break;

        default:
            break;
        }
    }

    void PropagationAnimator::Draw(Core::Renderer& renderer, Core::Material& nodeMaterial, Core::Material& edgeMaterial,
        const Core::Mesh& cubeMesh, Core::Font& font, float textScale) const
    {
        if (m_Network == nullptr || m_Layout == nullptr)
        {
            return;
        }

        const std::vector<std::vector<float>>& activations = m_Network->GetActivations();
        const int outputLayerIndex = m_TransitionCount;
        const bool finished = (m_Phase == PropagationPhase::HoldOutput || m_Phase == PropagationPhase::Idle);
        const int maxDrawnTransition = finished ? (m_TransitionCount - 1) : m_CurrentTransition;

        int predictedDigit = 0;
        if (!activations.back().empty())
        {
            predictedDigit = static_cast<int>(std::max_element(activations.back().begin(), activations.back().end()) - activations.back().begin());
        }

        // --- Edges: everything up to the current transition stays drawn, just dimmed once the
        // signal has moved past it. Only the transition actively in the Travel phase is bright.
        // This whole block (plus the travelling tokens and arrive-phase labels below) is the
        // O(visible^2) part of the draw - skippable via VisualizerConfig::renderEdges for scenes
        // where kVisibleNeuronsPerLayer is large enough that it costs real frame time. ---
        if (m_Config.renderEdges)
        {
            for (int transition = 0; transition <= maxDrawnTransition; transition++)
            {
                const bool isTravellingNow = (transition == m_CurrentTransition && m_Phase == PropagationPhase::Travel);
                const float brightness = isTravellingNow ? 1.0f : m_Config.dimBrightness;

                const std::vector<VisibleNeuron>& fromLayer = m_Layout->Neurons(transition);
                const std::vector<VisibleNeuron>& toLayer = m_Layout->Neurons(transition + 1);
                const Layer& weightLayer = m_Network->GetLayer(transition);

                for (const VisibleEdge& edge : m_Layout->Edges(transition))
                {
                    if (!ShouldRenderEdge(transition, edge.fromVisible, edge.toVisible, m_Config.edgeRenderPercentage))
                    {
                        continue;
                    }

                    const VisibleNeuron& fromNeuron = fromLayer[edge.fromVisible];
                    const VisibleNeuron& toNeuron = toLayer[edge.toVisible];
                    const float weight = weightLayer.weights[toNeuron.neuronIndex][fromNeuron.neuronIndex];

                    edgeMaterial.SetVec3("u_AlbedoColor", WeightColor(weight) * brightness);
                    const glm::mat4 transform = MakeBeamTransform(fromNeuron.position, toNeuron.position, m_Config.edgeThickness);
                    renderer.DrawMesh(cubeMesh, edgeMaterial, transform);
                }
            }

            // --- Travel phase: a token per edge carries the source value along the edge, picking up
            // the destination bias as it goes ("adding bias while numbers are on the edges"). ---
            if (m_Phase == PropagationPhase::Travel)
            {
                const float t = glm::clamp(m_PhaseTimer / m_Config.travelDuration, 0.0f, 1.0f);
                const std::vector<VisibleNeuron>& fromLayer = m_Layout->Neurons(m_CurrentTransition);
                const std::vector<VisibleNeuron>& toLayer = m_Layout->Neurons(m_CurrentTransition + 1);
                const Layer& weightLayer = m_Network->GetLayer(m_CurrentTransition);
                const std::vector<float>& fromActivations = activations[m_CurrentTransition];

                for (const VisibleEdge& edge : m_Layout->Edges(m_CurrentTransition))
                {
                    if (!ShouldRenderEdge(m_CurrentTransition, edge.fromVisible, edge.toVisible, m_Config.edgeRenderPercentage))
                    {
                        continue;
                    }

                    const VisibleNeuron& fromNeuron = fromLayer[edge.fromVisible];
                    const VisibleNeuron& toNeuron = toLayer[edge.toVisible];

                    const float sourceValue = fromActivations[fromNeuron.neuronIndex];
                    const float bias = weightLayer.biases[toNeuron.neuronIndex];
                    const float travelValue = sourceValue + bias * t;

                    const glm::vec3 tokenPos = glm::mix(fromNeuron.position, toNeuron.position, t);
                    const glm::mat4 tokenTransform = glm::scale(glm::translate(glm::mat4(1.0f), tokenPos), glm::vec3(m_Config.nodeSize * 0.3f));

                    nodeMaterial.SetVec3("u_AlbedoColor", glm::vec3(1.0f, 0.9f, 0.5f));
                    renderer.DrawMesh(cubeMesh, nodeMaterial, tokenTransform);
                    renderer.SubmitText(font, FormatFloat(travelValue, 2), tokenPos + glm::vec3(0.0f, m_Config.nodeSize * 0.6f, 0.0f),
                        textScale * 0.85f, glm::vec4(1.0f, 0.95f, 0.7f, 1.0f));
                }
            }

            // --- Arrive phase: each incoming value gets multiplied by its edge weight right at the
            // destination node ("multiplying weights in the nodes itself"). ---
            if (m_Phase == PropagationPhase::Arrive)
            {
                const std::vector<VisibleNeuron>& fromLayer = m_Layout->Neurons(m_CurrentTransition);
                const std::vector<VisibleNeuron>& toLayer = m_Layout->Neurons(m_CurrentTransition + 1);
                const Layer& weightLayer = m_Network->GetLayer(m_CurrentTransition);
                const std::vector<float>& fromActivations = activations[m_CurrentTransition];

                for (const VisibleEdge& edge : m_Layout->Edges(m_CurrentTransition))
                {
                    if (!ShouldRenderEdge(m_CurrentTransition, edge.fromVisible, edge.toVisible, m_Config.edgeRenderPercentage))
                    {
                        continue;
                    }

                    const VisibleNeuron& fromNeuron = fromLayer[edge.fromVisible];
                    const VisibleNeuron& toNeuron = toLayer[edge.toVisible];

                    const float sourceValue = fromActivations[fromNeuron.neuronIndex];
                    const float bias = weightLayer.biases[toNeuron.neuronIndex];
                    const float weight = weightLayer.weights[toNeuron.neuronIndex][fromNeuron.neuronIndex];
                    const float contribution = (sourceValue + bias) * weight;

                    const float fanIndex = static_cast<float>(edge.fromVisible) - static_cast<float>(fromLayer.size() - 1) * 0.5f;
                    const glm::vec3 labelPos = toNeuron.position
                        + glm::vec3(fanIndex * 0.14f, m_Config.nodeSize * 0.7f + std::abs(fanIndex) * 0.05f, 0.0f);

                    renderer.SubmitText(font, "x" + FormatFloat(weight, 2) + "=" + FormatFloat(contribution, 2), labelPos, textScale * 0.7f,
                        weight >= 0.0f ? glm::vec4(0.6f, 0.85f, 1.0f, 1.0f) : glm::vec4(1.0f, 0.6f, 0.6f, 1.0f));
                }
            }
        }

        // --- Nodes ---
        for (int layer = 0; layer < m_Layout->LayerCount(); layer++)
        {
            if (m_LayerState[layer] == LayerState::NotReached)
            {
                continue;
            }

            float brightness = m_Config.dimBrightness;
            float pulse = 1.0f;

            if (m_LayerState[layer] == LayerState::Active)
            {
                brightness = 1.0f;
            }
            else if (m_LayerState[layer] == LayerState::Arriving)
            {
                const float t = glm::clamp(m_PhaseTimer / m_Config.arriveDuration, 0.0f, 1.0f);
                brightness = glm::mix(0.2f, 1.0f, t);
                pulse = 1.0f + 0.2f * std::sin(t * 3.14159265f);
            }

            const bool isOutputLayer = (layer == outputLayerIndex);
            const std::vector<float>& layerActivations = activations[layer];

            for (const VisibleNeuron& neuron : m_Layout->Neurons(layer))
            {
                const float value = layerActivations[neuron.neuronIndex];

                const float normalized = glm::clamp(value / (value + 1.0f), 0.0f, 1.0f);
                const glm::vec3 low(0.06f, 0.07f, 0.11f);
                const glm::vec3 high = isOutputLayer ? glm::vec3(0.2f, 0.75f, 0.4f) : glm::vec3(0.95f, 0.55f, 0.15f);
                glm::vec3 color = glm::mix(low, high, normalized) * brightness;

                if (isOutputLayer && neuron.neuronIndex == predictedDigit && m_LayerState[layer] == LayerState::Active)
                {
                    color = glm::vec3(0.25f, 0.95f, 0.35f) * brightness;
                }

                nodeMaterial.SetVec3("u_AlbedoColor", color);
                const glm::mat4 transform = glm::scale(glm::translate(glm::mat4(1.0f), neuron.position), glm::vec3(m_Config.nodeSize * pulse));
                renderer.DrawMesh(cubeMesh, nodeMaterial, transform);

                const std::string label = isOutputLayer
                    ? (std::to_string(neuron.neuronIndex) + ": " + FormatFloat(value * 100.0f, 1) + "%")
                    : FormatFloat(value, 2);

                renderer.SubmitText(font, label, neuron.position + glm::vec3(m_Config.nodeSize * 0.7f, -0.06f, 0.0f), textScale * 0.85f,
                    glm::vec4(1.0f, 1.0f, 1.0f, glm::clamp(brightness + 0.3f, 0.0f, 1.0f)));
            }
        }

        // --- Final interpretation ---
        if (finished)
        {
            const float confidence = activations.back().empty() ? 0.0f : activations.back()[predictedDigit];

            std::string resultText;
            glm::vec4 resultColor;

            if (m_TrueLabel.has_value())
            {
                const bool correct = (predictedDigit == static_cast<int>(*m_TrueLabel));
                resultText = "Predicted: " + std::to_string(predictedDigit)
                    + " (" + FormatFloat(confidence * 100.0f, 1) + "%)   Actual: " + std::to_string(static_cast<int>(*m_TrueLabel))
                    + (correct ? "   [correct]" : "   [wrong]");
                resultColor = correct ? glm::vec4(0.4f, 1.0f, 0.5f, 1.0f) : glm::vec4(1.0f, 0.45f, 0.45f, 1.0f);
            }
            else
            {
                resultText = "Predicted: " + std::to_string(predictedDigit)
                    + " (" + FormatFloat(confidence * 100.0f, 1) + "%)   (hand-drawn, no set truth)";
                resultColor = glm::vec4(0.6f, 0.85f, 1.0f, 1.0f);
            }

            const glm::vec3 textPos(m_Layout->TotalWidth() * 0.5f - 1.6f, -3.f, 0.0f);
            renderer.SubmitText(font, resultText, textPos, textScale * 1.3f, resultColor);

            renderer.SubmitText(font, "Space: next sample", textPos + glm::vec3(0.0f, -0.5f, 0.0f), textScale * 0.9f,
                glm::vec4(0.8f, 0.8f, 0.85f, 1.0f));
        }
    }

} // namespace NN
