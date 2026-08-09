#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace NN
{

    // Images normalized to [0,1], one flattened row-major vector per sample.
    struct Dataset
    {
        std::vector<std::vector<float>> images;
        std::vector<uint8_t> labels;
        int imageWidth{ 0 };
        int imageHeight{ 0 };

        size_t Size() const { return images.size(); }
        bool Empty() const { return images.empty(); }
    };

    class MnistLoader
    {
    public:
        // Reads a standard MNIST idx3-ubyte image file. Returns false (leaving outputs
        // untouched) if the file can't be opened or fails the format's magic-number check.
        static bool LoadImages(const std::string& path, std::vector<std::vector<float>>& outImages, int& outWidth, int& outHeight);

        // Reads a standard MNIST idx1-ubyte label file.
        static bool LoadLabels(const std::string& path, std::vector<uint8_t>& outLabels);

        // Tries a small set of common MNIST filename spellings under `directory` for the given
        // split ("train" or "t10k"). Returns an empty Dataset if nothing could be loaded.
        static Dataset LoadSplit(const std::string& directory, const std::string& split);

        // Deterministic placeholder dataset used when real MNIST files aren't present, so the
        // whole pipeline (train -> predict -> visualize) still runs end to end. Each class 0-9
        // gets a distinct procedural stripe/checker pattern plus noise - NOT realistic digits,
        // just enough structure for the network to learn a real (if unimpressive) decision boundary.
        static Dataset MakeSyntheticDataset(int sampleCount, int width, int height, unsigned seed);
    };

    // Expands a normalized [0,1] grayscale image (as stored in Dataset::images) into a tightly
    // packed RGBA8 pixel buffer suitable for Core::Texture - so the raw pixel vector that feeds
    // the network's input layer can also be reconstructed and displayed as an actual image.
    std::vector<uint8_t> ToGrayscaleRGBA(const std::vector<float>& image);

} // namespace NN
