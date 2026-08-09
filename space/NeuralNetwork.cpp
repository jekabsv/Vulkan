#include "NeuralNetwork.h"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace NN
{

    namespace
    {
        float ReLU(float x) { return x > 0.0f ? x : 0.0f; }
        float ReLUDerivative(float preActivation) { return preActivation > 0.0f ? 1.0f : 0.0f; }

        void Softmax(const std::vector<float>& z, std::vector<float>& out)
        {
            const float maxZ = *std::max_element(z.begin(), z.end());

            float sum = 0.0f;
            for (size_t i = 0; i < z.size(); i++)
            {
                out[i] = std::exp(z[i] - maxZ);
                sum += out[i];
            }
            for (float& v : out)
            {
                v /= sum;
            }
        }

        constexpr uint32_t kWeightsFileMagic = 0x4E4E574B; // "NNWK"
    }

    NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int hiddenDepth, int outputSize)
        : m_Rng(std::random_device{}())
    {
        m_Widths.push_back(inputSize);
        for (int i = 0; i < hiddenDepth; i++)
        {
            m_Widths.push_back(hiddenSize);
        }
        m_Widths.push_back(outputSize);

        const int layerCount = static_cast<int>(m_Widths.size()) - 1;

        m_Layers.resize(layerCount);
        m_PreActivations.resize(layerCount);
        m_Deltas.resize(layerCount);
        m_Activations.resize(layerCount + 1);

        for (int l = 0; l < layerCount; l++)
        {
            const int inWidth = m_Widths[l];
            const int outWidth = m_Widths[l + 1];

            Layer& layer = m_Layers[l];
            layer.weights.assign(outWidth, std::vector<float>(inWidth, 0.0f));
            layer.biases.assign(outWidth, 0.0f);
            layer.weightGradients.assign(outWidth, std::vector<float>(inWidth, 0.0f));
            layer.biasGradients.assign(outWidth, 0.0f);

            m_PreActivations[l].assign(outWidth, 0.0f);
            m_Deltas[l].assign(outWidth, 0.0f);
        }

        for (int i = 0; i <= layerCount; i++)
        {
            m_Activations[i].assign(m_Widths[i], 0.0f);
        }

        RandomizeWeights(std::random_device{}());
    }

    void NeuralNetwork::RandomizeWeights(unsigned seed)
    {
        m_Rng.seed(seed);

        for (Layer& layer : m_Layers)
        {
            // He initialization: scaled for ReLU hidden layers, still reasonable for the
            // Softmax output layer. Keeps activations from exploding/vanishing at width 784.
            const float stddev = std::sqrt(2.0f / static_cast<float>(layer.InWidth()));
            std::normal_distribution<float> dist(0.0f, stddev);

            for (std::vector<float>& weightVector : layer.weights)
            {
                for (float& w : weightVector)
                {
                    w = dist(m_Rng);
                }
            }
            std::fill(layer.biases.begin(), layer.biases.end(), 0.0f);
        }
    }

    const std::vector<float>& NeuralNetwork::Forward(const std::vector<float>& input)
    {
        m_Activations[0] = input;

        const int layerCount = GetLayerCount();

        for (int l = 0; l < layerCount; l++)
        {
            const Layer& layer = m_Layers[l];
            const std::vector<float>& prevActivations = m_Activations[l];
            std::vector<float>& z = m_PreActivations[l];

            // --- Multiply pass: each output neuron walks its OWN weight vector against the
            // previous layer's activations. Deliberately not a matrix-vector multiply call. ---
            for (int j = 0; j < layer.OutWidth(); j++)
            {
                float sum = 0.0f;
                const std::vector<float>& weightVector = layer.weights[j];

                for (int i = 0; i < layer.InWidth(); i++)
                {
                    sum += weightVector[i] * prevActivations[i];
                }

                z[j] = sum;
            }

            // --- Bias pass: adding the layer's bias vector is its own separate step. ---
            for (int j = 0; j < layer.OutWidth(); j++)
            {
                z[j] += layer.biases[j];
            }

            // --- Activation pass ---
            std::vector<float>& activations = m_Activations[l + 1];

            if (l == layerCount - 1)
            {
                Softmax(z, activations);
            }
            else
            {
                for (int j = 0; j < layer.OutWidth(); j++)
                {
                    activations[j] = ReLU(z[j]);
                }
            }
        }

        return m_Activations.back();
    }

    float NeuralNetwork::AccumulateGradients(const std::vector<float>& input, const std::vector<float>& target)
    {
        Forward(input);

        const int layerCount = GetLayerCount();
        const std::vector<float>& output = m_Activations.back();

        float loss = 0.0f;
        for (size_t k = 0; k < target.size(); k++)
        {
            loss -= target[k] * std::log(std::max(output[k], 1e-7f));
        }

        // Output layer delta: combined Softmax + CrossEntropy gradient (dLoss/dz = predicted - target).
        for (size_t j = 0; j < output.size(); j++)
        {
            m_Deltas[layerCount - 1][j] = output[j] - target[j];
        }

        // Backpropagate through hidden layers: delta[l][i] = (sum_j weights[l+1][j][i] * delta[l+1][j]) * ReLU'(z[l][i]).
        // Weights are stored per-output-neuron, so this is an explicit transpose-style double loop, not a matrix multiply.
        for (int l = layerCount - 2; l >= 0; l--)
        {
            const Layer& nextLayer = m_Layers[l + 1];
            std::vector<float>& delta = m_Deltas[l];
            const std::vector<float>& z = m_PreActivations[l];

            std::fill(delta.begin(), delta.end(), 0.0f);

            for (int j = 0; j < nextLayer.OutWidth(); j++)
            {
                const float nextDelta = m_Deltas[l + 1][j];
                const std::vector<float>& weightVector = nextLayer.weights[j];

                for (int i = 0; i < nextLayer.InWidth(); i++)
                {
                    delta[i] += weightVector[i] * nextDelta;
                }
            }

            for (int i = 0; i < static_cast<int>(delta.size()); i++)
            {
                delta[i] *= ReLUDerivative(z[i]);
            }
        }

        // Gradients: weightGradients[l][j][i] += delta[l][j] * activations[l][i], biasGradients[l][j] += delta[l][j].
        for (int l = 0; l < layerCount; l++)
        {
            Layer& layer = m_Layers[l];
            const std::vector<float>& prevActivations = m_Activations[l];
            const std::vector<float>& delta = m_Deltas[l];

            for (int j = 0; j < layer.OutWidth(); j++)
            {
                layer.biasGradients[j] += delta[j];

                std::vector<float>& weightGradientVector = layer.weightGradients[j];
                for (int i = 0; i < layer.InWidth(); i++)
                {
                    weightGradientVector[i] += delta[j] * prevActivations[i];
                }
            }
        }

        return loss;
    }

    void NeuralNetwork::ApplyGradients(float learningRate, int batchSize)
    {
        if (batchSize <= 0)
        {
            return;
        }

        const float scale = learningRate / static_cast<float>(batchSize);

        for (Layer& layer : m_Layers)
        {
            for (int j = 0; j < layer.OutWidth(); j++)
            {
                layer.biases[j] -= scale * layer.biasGradients[j];

                std::vector<float>& weightVector = layer.weights[j];
                const std::vector<float>& weightGradientVector = layer.weightGradients[j];

                for (int i = 0; i < layer.InWidth(); i++)
                {
                    weightVector[i] -= scale * weightGradientVector[i];
                }
            }
        }

        ZeroGradients();
    }

    void NeuralNetwork::ZeroGradients()
    {
        for (Layer& layer : m_Layers)
        {
            for (std::vector<float>& weightGradientVector : layer.weightGradients)
            {
                std::fill(weightGradientVector.begin(), weightGradientVector.end(), 0.0f);
            }
            std::fill(layer.biasGradients.begin(), layer.biasGradients.end(), 0.0f);
        }
    }

    int NeuralNetwork::Predict(const std::vector<float>& input)
    {
        const std::vector<float>& output = Forward(input);
        return static_cast<int>(std::max_element(output.begin(), output.end()) - output.begin());
    }

    bool NeuralNetwork::SaveWeights(const std::string& path) const
    {
        std::ofstream file(path, std::ios::binary);
        if (!file)
        {
            return false;
        }

        const uint32_t magic = kWeightsFileMagic;
        const uint32_t widthCount = static_cast<uint32_t>(m_Widths.size());

        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        file.write(reinterpret_cast<const char*>(&widthCount), sizeof(widthCount));
        file.write(reinterpret_cast<const char*>(m_Widths.data()), sizeof(int) * m_Widths.size());

        for (const Layer& layer : m_Layers)
        {
            for (const std::vector<float>& weightVector : layer.weights)
            {
                file.write(reinterpret_cast<const char*>(weightVector.data()), sizeof(float) * weightVector.size());
            }
            file.write(reinterpret_cast<const char*>(layer.biases.data()), sizeof(float) * layer.biases.size());
        }

        return static_cast<bool>(file);
    }

    bool NeuralNetwork::LoadWeights(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
        {
            return false;
        }

        uint32_t magic = 0;
        uint32_t widthCount = 0;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        file.read(reinterpret_cast<char*>(&widthCount), sizeof(widthCount));

        if (!file || magic != kWeightsFileMagic || widthCount != m_Widths.size())
        {
            return false;
        }

        std::vector<int> widths(widthCount);
        file.read(reinterpret_cast<char*>(widths.data()), sizeof(int) * widthCount);

        if (!file || widths != m_Widths)
        {
            return false;
        }

        for (Layer& layer : m_Layers)
        {
            for (std::vector<float>& weightVector : layer.weights)
            {
                file.read(reinterpret_cast<char*>(weightVector.data()), sizeof(float) * weightVector.size());
            }
            file.read(reinterpret_cast<char*>(layer.biases.data()), sizeof(float) * layer.biases.size());
        }

        return static_cast<bool>(file);
    }

    std::vector<float> OneHot(uint8_t label, int numClasses)
    {
        std::vector<float> result(numClasses, 0.0f);
        if (label < numClasses)
        {
            result[label] = 1.0f;
        }
        return result;
    }

} // namespace NN
