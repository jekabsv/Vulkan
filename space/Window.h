#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

//#include "Utils.h"
#include <glm/glm.hpp>

struct GLFWwindow;

namespace Core
{

    enum class KeyCode : uint16_t
    {
        Unknown = 0,
        Space, A, B, C, D, E, F, G, H, I, J, K, L, M,
        N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
        Escape, Enter, Tab, Backspace, Insert, Delete,
        Right, Left, Down, Up,
        LeftBracket, RightBracket,
        Count
    };

    enum class MouseButton : uint8_t
    {
        Left = 0,
        Right,
        Middle,
        Button3, Button4, Button5, Button6, Button7,
        Count
    };

    enum class KeyState : uint8_t
    {
        Released = 0,
        Pressed,
        Held
    };

    constexpr std::size_t KeyCodeCount = static_cast<std::size_t>(KeyCode::Count);
    constexpr std::size_t MouseButtonCount = static_cast<std::size_t>(MouseButton::Count);

    struct InputSnapshot
    {
        std::array<KeyState, KeyCodeCount> keyStates{};
        std::array<KeyState, MouseButtonCount> mouseButtonStates{};

        double mouseX{ 0.0 };
        double mouseY{ 0.0 };
        double mouseDeltaX{ 0.0 };
        double mouseDeltaY{ 0.0 };
        double scrollWheelDelta{ 0.0 };
    };

    class Window
    {
    public:
        Window(uint32_t width, uint32_t height, const std::string& title);
        ~Window();

        Window(const Window&) = delete;
        Window& operator=(const Window&) = delete;
        Window(Window&& other) noexcept;
        Window& operator=(Window&& other) noexcept;

        void PollEvents();
        void WaitEvents();
        bool ShouldClose() const;
        void Close();

        uint32_t GetWidth() const { return m_Width; }
        uint32_t GetHeight() const { return m_Height; }
        bool IsMinimized() const { return m_Width == 0 || m_Height == 0; }
        bool WasResized() const { return m_Resized; }
        void ResetResizeFlag() { m_Resized = false; }

        const std::string& GetTitle() const { return m_Title; }
        void SetTitle(const std::string& title);

        glm::vec2 GetMousePos() const;
        double GetScrollDelta() const { return m_CurrentSnapshot.scrollWheelDelta; }

        KeyState GetKeyboardKey(KeyCode key) const;
        KeyState GetMouseKey(MouseButton button) const;

        bool IsKeyDown(KeyCode key) const;
        bool IsKeyPressed(KeyCode key) const;
        bool IsMouseButtonDown(MouseButton button) const;
        bool IsMouseButtonPressed(MouseButton button) const;

        const InputSnapshot& GetInputSnapshot() const { return m_CurrentSnapshot; }

        GLFWwindow* GetNativeHandle() const { return m_WindowHandle; }

    private:
        void SetupCallbacks();
        void ResolveInputStates();
        void DestroyWindow();

        static Window* FromHandle(GLFWwindow* window);

        static void FramebufferResizeCallback(GLFWwindow* window, int width, int height);
        static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
        static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
        static void CursorPosCallback(GLFWwindow* window, double xpos, double ypos);
        static void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    private:
        uint32_t m_Width{ 0 };
        uint32_t m_Height{ 0 };
        std::string m_Title;
        bool m_Resized{ false };

        GLFWwindow* m_WindowHandle{ nullptr };

        InputSnapshot m_CurrentSnapshot;
        std::array<bool, KeyCodeCount> m_KeyDown{};
        std::array<bool, MouseButtonCount> m_MouseButtonDown{};

        double m_LastMouseX{ 0.0 };
        double m_LastMouseY{ 0.0 };
    };

} // namespace Core