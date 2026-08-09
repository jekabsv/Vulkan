#include "MnistData.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>

namespace NN
{

    namespace
    {
        uint32_t ReadBigEndianU32(std::ifstream& file)
        {
            uint8_t bytes[4]{};
            file.read(reinterpret_cast<char*>(bytes), 4);
            return (static_cast<uint32_t>(bytes[0]) << 24)
                | (static_cast<uint32_t>(bytes[1]) << 16)
                | (static_cast<uint32_t>(bytes[2]) << 8)
                | static_cast<uint32_t>(bytes[3]);
        }

        constexpr uint32_t kImageMagic = 0x00000803;
        constexpr uint32_t kLabelMagic = 0x00000801;
    }

    bool MnistLoader::LoadImages(const std::string& path, std::vector<std::vector<float>>& outImages, int& outWidth, int& outHeight)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
        {
            return false;
        }

        const uint32_t magic = ReadBigEndianU32(file);
        if (magic != kImageMagic)
        {
            return false;
        }

        const uint32_t count = ReadBigEndianU32(file);
        const uint32_t rows = ReadBigEndianU32(file);
        const uint32_t cols = ReadBigEndianU32(file);

        if (!file || rows == 0 || cols == 0)
        {
            return false;
        }

        const size_t pixelsPerImage = static_cast<size_t>(rows) * cols;
        std::vector<uint8_t> raw(pixelsPerImage);

        outImages.clear();
        outImages.reserve(count);

        for (uint32_t i = 0; i < count; i++)
        {
            file.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(pixelsPerImage));
            if (!file)
            {
                return false;
            }

            std::vector<float> image(pixelsPerImage);
            for (size_t p = 0; p < pixelsPerImage; p++)
            {
                image[p] = static_cast<float>(raw[p]) / 255.0f;
            }
            outImages.push_back(std::move(image));
        }

        outWidth = static_cast<int>(cols);
        outHeight = static_cast<int>(rows);
        return true;
    }

    bool MnistLoader::LoadLabels(const std::string& path, std::vector<uint8_t>& outLabels)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
        {
            return false;
        }

        const uint32_t magic = ReadBigEndianU32(file);
        if (magic != kLabelMagic)
        {
            return false;
        }

        const uint32_t count = ReadBigEndianU32(file);
        if (!file)
        {
            return false;
        }

        outLabels.resize(count);
        file.read(reinterpret_cast<char*>(outLabels.data()), count);

        return static_cast<bool>(file);
    }

    Dataset MnistLoader::LoadSplit(const std::string& directory, const std::string& split)
    {
        Dataset dataset;

        const bool isTrain = (split == "train");

        const std::vector<std::string> imageCandidates = isTrain
            ? std::vector<std::string>{ "train-images-idx3-ubyte", "train-images.idx3-ubyte" }
        : std::vector<std::string>{ "t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte" };

        const std::vector<std::string> labelCandidates = isTrain
            ? std::vector<std::string>{ "train-labels-idx1-ubyte", "train-labels.idx1-ubyte" }
        : std::vector<std::string>{ "t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte" };

        bool imagesLoaded = false;
        for (const std::string& name : imageCandidates)
        {
            if (LoadImages(directory + "/" + name, dataset.images, dataset.imageWidth, dataset.imageHeight))
            {
                imagesLoaded = true;
                break;
            }
        }

        bool labelsLoaded = false;
        for (const std::string& name : labelCandidates)
        {
            if (LoadLabels(directory + "/" + name, dataset.labels))
            {
                labelsLoaded = true;
                break;
            }
        }

        if (!imagesLoaded || !labelsLoaded || dataset.images.size() != dataset.labels.size())
        {
            return Dataset{};
        }

        return dataset;
    }

    Dataset MnistLoader::MakeSyntheticDataset(int sampleCount, int width, int height, unsigned seed)
    {
        Dataset dataset;
        dataset.imageWidth = width;
        dataset.imageHeight = height;
        dataset.images.reserve(sampleCount);
        dataset.labels.reserve(sampleCount);

        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> classDist(0, 9);
        std::uniform_real_distribution<float> noiseDist(-0.15f, 0.15f);

        for (int s = 0; s < sampleCount; s++)
        {
            const int digitClass = classDist(rng);
            std::vector<float> image(static_cast<size_t>(width) * height, 0.0f);

            // Each class gets a distinct orientation/frequency of sine stripes, so classes are
            // separable but not trivially identical - a stand-in for real digit strokes.
            const float angle = static_cast<float>(digitClass) * (3.14159265f / 10.0f);
            const float frequency = 1.5f + static_cast<float>(digitClass) * 0.35f;
            const float dx = std::cos(angle);
            const float dy = std::sin(angle);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    const float nx = static_cast<float>(x) / static_cast<float>(width) - 0.5f;
                    const float ny = static_cast<float>(y) / static_cast<float>(height) - 0.5f;
                    const float wave = std::sin((nx * dx + ny * dy) * frequency * 6.2831853f);

                    float value = 0.5f + 0.5f * wave + noiseDist(rng);
                    value = std::max(0.0f, std::min(1.0f, value));

                    image[static_cast<size_t>(y) * width + x] = value;
                }
            }

            dataset.images.push_back(std::move(image));
            dataset.labels.push_back(static_cast<uint8_t>(digitClass));
        }

        return dataset;
    }

    std::vector<uint8_t> ToGrayscaleRGBA(const std::vector<float>& image)
    {
        std::vector<uint8_t> pixels(image.size() * 4);

        for (size_t i = 0; i < image.size(); i++)
        {
            const uint8_t value = static_cast<uint8_t>(std::clamp(image[i], 0.0f, 1.0f) * 255.0f);
            pixels[i * 4 + 0] = value;
            pixels[i * 4 + 1] = value;
            pixels[i * 4 + 2] = value;
            pixels[i * 4 + 3] = 255;
        }

        return pixels;
    }

} // namespace NN
