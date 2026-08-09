#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <glm/glm.hpp>

namespace Core
{
    class Renderer;
    class Material;
    class Mesh;
    class Font;
}

namespace NN
{

    class NeuralNetwork;

    struct VisibleNeuron
    {
        int neuronIndex{ 0 }; // index into the full activation vector for this layer
        glm::vec3 position{ 0.0f };
    };

    struct VisibleEdge
    {
        int fromVisible{ 0 }; // index into the previous layer's VisibleNeuron list
        int toVisible{ 0 };   // index into this layer's VisibleNeuron list
    };

    struct VisualizerConfig
    {
        int visibleNeuronsPerLayer{ 10 };
        bool renderEdges{ true }; // edges (and the tokens/labels that travel along them) cost O(visible^2) draws per transition
        float edgeRenderPercentage{ 1.0f }; // 0..1: fraction of edges kept when renderEdges is true, thinned out deterministically
        float activeThreshold{ 0.02f };

        float layerSpacing{ 2.6f };
        float neuronSpacing{ 0.55f };
        float nodeSize{ 0.28f };
        float edgeThickness{ 0.02f };

        float travelDuration{ 1.0f };
        float arriveDuration{ 0.35f };
        float activateDuration{ 0.35f };
        float outputHoldDuration{ 4.0f };

        float dimBrightness{ 0.3f };
    };

    // Picks, for each activation layer, the first N "active" neurons (input: pixel above
    // threshold, hidden: post-ReLU > 0, output: always all of them - it's already small) and
    // lays them out on a flat grid: layers along X, neurons within a layer along Y.
    class NetworkLayout
    {
    public:
        void Build(const NeuralNetwork& network, const VisualizerConfig& config);

        int LayerCount() const { return static_cast<int>(m_Layers.size()); }
        const std::vector<VisibleNeuron>& Neurons(int layerIndex) const { return m_Layers[layerIndex]; }
        const std::vector<VisibleEdge>& Edges(int transitionIndex) const { return m_Edges[transitionIndex]; }

        float TotalWidth() const;

    private:
        std::vector<std::vector<VisibleNeuron>> m_Layers; // one entry per activation layer (0..weightLayerCount)
        std::vector<std::vector<VisibleEdge>> m_Edges;    // one entry per weight layer (transition)
        float m_LayerSpacing{ 1.0f };
    };

    enum class PropagationPhase
    {
        Travel,     // values move along the edges of the current transition; bias accrues en route
        Arrive,     // values reach the destination nodes and get multiplied by their edge weight
        Activate,   // destination node sums its contributions and applies ReLU/Softmax
        HoldOutput, // animation finished, output layer stays lit while the result is shown
        Idle        // waiting for a new sample
    };

    // Plays back an already-computed forward pass (NeuralNetwork::Forward must have been called
    // before Begin()) as a slow layer-by-layer reveal. It recomputes nothing - the real forward
    // pass is instantaneous; this only narrates it on a timer for legibility.
    class PropagationAnimator
    {
    public:
        // trueLabel is empty for images with no ground truth (hand-drawn digits fed from the
        // editor) - the final interpretation just skips the "Actual: .../[correct]/[wrong]" line then.
        void Begin(const NeuralNetwork& network, const NetworkLayout& layout, std::optional<uint8_t> trueLabel, const VisualizerConfig& config);
        void Update(float deltaTime);
        bool IsIdle() const { return m_Phase == PropagationPhase::Idle; }

        void Draw(Core::Renderer& renderer, Core::Material& nodeMaterial, Core::Material& edgeMaterial,
            const Core::Mesh& cubeMesh, Core::Font& font, float textScale) const;

    private:
        enum class LayerState
        {
            NotReached, // signal hasn't gotten here yet - not drawn at all
            Arriving,   // currently ramping in during the Arrive phase
            Active,     // fully lit - either currently holding a value or the final output
            Dimmed      // its value has moved on; still drawn, just not bright ("not so bright")
        };

        const NeuralNetwork* m_Network{ nullptr };
        const NetworkLayout* m_Layout{ nullptr };
        VisualizerConfig m_Config;
        std::optional<uint8_t> m_TrueLabel;

        int m_TransitionCount{ 0 };
        int m_CurrentTransition{ 0 };
        PropagationPhase m_Phase{ PropagationPhase::Idle };
        float m_PhaseTimer{ 0.0f };

        std::vector<LayerState> m_LayerState; // one per activation layer
    };

} // namespace NN
