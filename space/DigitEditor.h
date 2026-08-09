#pragma once

#include <vector>

namespace NN
{

    // A tiny 28x28 paint canvas. Values are in [0,1] where 1.0 = blank white paper and painting
    // subtracts ink (moving toward 0.0 = fully inked), so overlapping strokes darken further, up
    // to solid black - the natural "pen on white paper" reading. That's the opposite of MNIST's
    // own convention (black background, white strokes), so ToMnistInput()/InvertPaperToMnist()
    // below flip it right before anything gets fed to the network - the editor and the saved
    // slots both stay in the natural paper convention for display, only the network ever sees
    // the inverted version.
    class DigitEditor
    {
    public:
        static constexpr int kGridSize = 28;

        DigitEditor();

        void Clear();

        // centerX/centerY are continuous grid coordinates (0..kGridSize), not pixel indices, so
        // the brush can be positioned and fall off sub-pixel accurately.
        void PaintAt(float centerX, float centerY);

        void GrowBrush();
        void ShrinkBrush();
        float GetBrushRadius() const { return m_BrushRadius; }

        const std::vector<float>& GetCanvas() const { return m_Canvas; }

        std::vector<float> ToMnistInput() const;

    private:
        std::vector<float> m_Canvas;
        float m_BrushRadius{ 1.5f };
    };

    // Inverts a white-paper/dark-ink image (as drawn by DigitEditor, or as stored in a saved
    // slot) into MNIST's black-background/white-stroke convention.
    std::vector<float> InvertPaperToMnist(const std::vector<float>& paperImage);

} // namespace NN
