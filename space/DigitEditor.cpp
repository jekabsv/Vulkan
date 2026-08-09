#include "DigitEditor.h"

#include <algorithm>
#include <cmath>

namespace NN
{

    namespace
    {
        constexpr float kMinBrushRadius = 0.5f;
        constexpr float kMaxBrushRadius = 6.0f;
        constexpr float kBrushRadiusStep = 0.5f;
    }

    DigitEditor::DigitEditor()
        : m_Canvas(static_cast<size_t>(kGridSize) * kGridSize, 1.0f)
    {
    }

    void DigitEditor::Clear()
    {
        std::fill(m_Canvas.begin(), m_Canvas.end(), 1.0f);
    }

    void DigitEditor::PaintAt(float centerX, float centerY)
    {
        const int minX = std::max(0, static_cast<int>(std::floor(centerX - m_BrushRadius)));
        const int maxX = std::min(kGridSize - 1, static_cast<int>(std::ceil(centerX + m_BrushRadius)));
        const int minY = std::max(0, static_cast<int>(std::floor(centerY - m_BrushRadius)));
        const int maxY = std::min(kGridSize - 1, static_cast<int>(std::ceil(centerY + m_BrushRadius)));

        for (int y = minY; y <= maxY; y++)
        {
            for (int x = minX; x <= maxX; x++)
            {
                const float dx = (static_cast<float>(x) + 0.5f) - centerX;
                const float dy = (static_cast<float>(y) + 0.5f) - centerY;
                const float distance = std::sqrt(dx * dx + dy * dy);

                if (distance > m_BrushRadius)
                {
                    continue;
                }

                // Soft/faded edge: full ink at the brush center, fading smoothly to nothing at
                // its radius. Squaring the linear falloff makes the taper read softer than a
                // straight cone.
                const float falloff = 1.0f - (distance / m_BrushRadius);
                const float strength = falloff * falloff;

                const size_t index = static_cast<size_t>(y) * kGridSize + x;
                m_Canvas[index] = std::max(0.0f, m_Canvas[index] - strength);
            }
        }
    }

    void DigitEditor::GrowBrush()
    {
        m_BrushRadius = std::min(kMaxBrushRadius, m_BrushRadius + kBrushRadiusStep);
    }

    void DigitEditor::ShrinkBrush()
    {
        m_BrushRadius = std::max(kMinBrushRadius, m_BrushRadius - kBrushRadiusStep);
    }

    std::vector<float> DigitEditor::ToMnistInput() const
    {
        return InvertPaperToMnist(m_Canvas);
    }

    std::vector<float> InvertPaperToMnist(const std::vector<float>& paperImage)
    {
        std::vector<float> inverted(paperImage.size());
        for (size_t i = 0; i < paperImage.size(); i++)
        {
            inverted[i] = 1.0f - paperImage[i];
        }
        return inverted;
    }

} // namespace NN
