#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace NN
{

    // A single fully-connected layer, mapping an input activation vector to an output
    // activation vector. Deliberately NOT a flat matrix: every output neuron owns its own
    // weight vector (weights[outNeuron][inNeuron]), and forward/backward walk these vectors
    // one neuron at a time with plain loops. That is what lets the visualizer point at "neuron
    // j's weight vector" as a distinct object and animate it, and it is what the multiply and
    // bias-add steps below operate on separately instead of doing one matrix-vector multiply.
    struct Layer
    {
        std::vector<std::vector<float>> weights; // weights[outNeuron][inNeuron]
        std::vector<float> biases;                // biases[outNeuron]

        std::vector<std::vector<float>> weightGradients; // same shape as weights, accumulated over a batch
        std::vector<float> biasGradients;                 // same shape as biases

        int InWidth() const { return weights.empty() ? 0 : static_cast<int>(weights[0].size()); }
        int OutWidth() const { return static_cast<int>(weights.size()); }
    };

    // Plain multi-layer perceptron: Input -> HiddenDepth hidden layers (uniform width, ReLU) ->
    // Output (Softmax). Trained with mini-batch SGD via manual backpropagation - no linear
    // algebra library, every sum is an explicit loop over a neuron's own weight vector.
    class NeuralNetwork
    {
    public:
        NeuralNetwork(int inputSize, int hiddenSize, int hiddenDepth, int outputSize);

        void RandomizeWeights(unsigned seed);

        // Computes activations for every layer (including the input at index 0) from `input`.
        // Results are cached (GetActivations/GetPreActivations) for both the visualizer and Backward().
        const std::vector<float>& Forward(const std::vector<float>& input);

        // Runs Forward(input), then accumulates weight/bias gradients for a Softmax+CrossEntropy
        // loss against a one-hot `target` into each Layer's *Gradients members. Does not apply
        // them - call ApplyGradients() after a full mini-batch. Returns the sample's loss.
        float AccumulateGradients(const std::vector<float>& input, const std::vector<float>& target);

        void ApplyGradients(float learningRate, int batchSize);
        void ZeroGradients();

        int Predict(const std::vector<float>& input); // runs Forward(), returns argmax digit

        // activations[0] = input, activations[i+1] = output of weight-layer i (post-activation).
        const std::vector<std::vector<float>>& GetActivations() const { return m_Activations; }
        // preActivations[i] = raw weighted sum + bias for weight-layer i, before ReLU/Softmax.
        const std::vector<std::vector<float>>& GetPreActivations() const { return m_PreActivations; }

        int GetLayerCount() const { return static_cast<int>(m_Layers.size()); } // hiddenDepth + 1
        const Layer& GetLayer(int weightLayerIndex) const { return m_Layers[weightLayerIndex]; }
        int GetWidth(int activationLayerIndex) const { return m_Widths[activationLayerIndex]; } // 0..GetLayerCount()

        bool SaveWeights(const std::string& path) const;
        bool LoadWeights(const std::string& path);

    private:
        std::vector<int> m_Widths; // size GetLayerCount()+1
        std::vector<Layer> m_Layers;

        std::vector<std::vector<float>> m_Activations;
        std::vector<std::vector<float>> m_PreActivations;
        std::vector<std::vector<float>> m_Deltas; // scratch space reused by AccumulateGradients, one per weight-layer

        std::mt19937 m_Rng;
    };

    // One-hot encodes `label` into a vector of length `numClasses`.
    std::vector<float> OneHot(uint8_t label, int numClasses);

} // namespace NN
