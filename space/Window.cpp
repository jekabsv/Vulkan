#include "Window.h"

#include <GLFW/glfw3.h>

#include <stdexcept>
#include <utility>

namespace Core
{

    namespace
    {
        uint32_t s_WindowCount = 0;

        static_assert(static_cast<uint16_t>(KeyCode::Z) - static_cast<uint16_t>(KeyCode::A) == 25,
            "KeyCode letters must stay contiguous from A to Z");
        static_assert(static_cast<uint16_t>(KeyCode::Num9) - static_cast<uint16_t>(KeyCode::Num0) == 9,
            "KeyCode digits must stay contiguous from Num0 to Num9");
        static_assert(GLFW_MOUSE_BUTTON_LAST + 1 <= static_cast<int>(MouseButtonCount),
            "MouseButton must cover every GLFW mouse button");

        KeyCode TranslateKey(int key)
        {
            if (key >= GLFW_KEY_A && key <= GLFW_KEY_Z)
            {
                const int offset = key - GLFW_KEY_A;
                return static_cast<KeyCode>(static_cast<uint16_t>(KeyCode::A) + offset);
            }

            if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9)
            {
                const int offset = key - GLFW_KEY_0;
                return static_cast<KeyCode>(static_cast<uint16_t>(KeyCode::Num0) + offset);
            }

            switch (key)
            {
            case GLFW_KEY_SPACE:     return KeyCode::Space;
            case GLFW_KEY_ESCAPE:    return KeyCode::Escape;
            case GLFW_KEY_ENTER:     return KeyCode::Enter;
            case GLFW_KEY_TAB:       return KeyCode::Tab;
            case GLFW_KEY_BACKSPACE: return KeyCode::Backspace;
            case GLFW_KEY_INSERT:    return KeyCode::Insert;
            case GLFW_KEY_DELETE:    return KeyCode::Delete;
            case GLFW_KEY_RIGHT:     return KeyCode::Right;
            case GLFW_KEY_LEFT:      return KeyCode::Left;
            case GLFW_KEY_DOWN:      return KeyCode::Down;
            case GLFW_KEY_UP:        return KeyCode::Up;
            case GLFW_KEY_LEFT_BRACKET:  return KeyCode::LeftBracket;
            case GLFW_KEY_RIGHT_BRACKET: return KeyCode::RightBracket;
            default:                 return KeyCode::Unknown;
            }
        }

        bool TranslateMouseButton(int button, MouseButton& outButton)
        {
            if (button < 0 || button >= static_cast<int>(MouseButtonCount))
            {
                return false;
            }

            outButton = static_cast<MouseButton>(button);
            return true;
        }
    }

    Window::Window(uint32_t width, uint32_t height, const std::string& title) :
        m_Width(width), m_Height(height), m_Title(title)
    {
        if (s_WindowCount == 0)
        {
            if (glfwInit() != GLFW_TRUE)
            {
                throw std::runtime_error("Failed to initialize GLFW");
            }
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        m_WindowHandle = glfwCreateWindow(
            static_cast<int>(width),
            static_cast<int>(height),
            m_Title.c_str(),
            nullptr,
            nullptr);

        if (m_WindowHandle == nullptr)
        {
            if (s_WindowCount == 0)
            {
                glfwTerminate();
            }

            throw std::runtime_error("Failed to create GLFW window");
        }

        s_WindowCount++;

        int framebufferWidth = 0;
        int framebufferHeight = 0;
        glfwGetFramebufferSize(m_WindowHandle, &framebufferWidth, &framebufferHeight);
        m_Width = static_cast<uint32_t>(framebufferWidth);
        m_Height = static_cast<uint32_t>(framebufferHeight);

        double cursorX = 0.0;
        double cursorY = 0.0;
        glfwGetCursorPos(m_WindowHandle, &cursorX, &cursorY);
        m_CurrentSnapshot.mouseX = cursorX;
        m_CurrentSnapshot.mouseY = cursorY;
        m_LastMouseX = cursorX;
        m_LastMouseY = cursorY;

        glfwSetWindowUserPointer(m_WindowHandle, this);
        SetupCallbacks();
    }

    Window::~Window()
    {
        DestroyWindow();
    }

    Window::Window(Window&& other) noexcept
        : m_Width(other.m_Width), m_Height(other.m_Height), m_Title(std::move(other.m_Title)), m_Resized(other.m_Resized)
        , m_WindowHandle(other.m_WindowHandle), m_CurrentSnapshot(other.m_CurrentSnapshot), m_KeyDown(other.m_KeyDown)
        , m_MouseButtonDown(other.m_MouseButtonDown), m_LastMouseX(other.m_LastMouseX), m_LastMouseY(other.m_LastMouseY)
    {
        other.m_WindowHandle = nullptr;

        if (m_WindowHandle != nullptr)
        {
            glfwSetWindowUserPointer(m_WindowHandle, this);
        }
    }

    Window& Window::operator=(Window&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        DestroyWindow();

        m_Width = other.m_Width;
        m_Height = other.m_Height;
        m_Title = std::move(other.m_Title);
        m_Resized = other.m_Resized;
        m_WindowHandle = other.m_WindowHandle;
        m_CurrentSnapshot = other.m_CurrentSnapshot;
        m_KeyDown = other.m_KeyDown;
        m_MouseButtonDown = other.m_MouseButtonDown;
        m_LastMouseX = other.m_LastMouseX;
        m_LastMouseY = other.m_LastMouseY;

        other.m_WindowHandle = nullptr;

        if (m_WindowHandle != nullptr)
        {
            glfwSetWindowUserPointer(m_WindowHandle, this);
        }

        return *this;
    }

    void Window::DestroyWindow()
    {
        if (m_WindowHandle == nullptr)
        {
            return;
        }

        glfwDestroyWindow(m_WindowHandle);
        m_WindowHandle = nullptr;

        s_WindowCount--;

        if (s_WindowCount == 0)
        {
            glfwTerminate();
        }
    }

    void Window::SetupCallbacks()
    {
        glfwSetFramebufferSizeCallback(m_WindowHandle, FramebufferResizeCallback);
        glfwSetKeyCallback(m_WindowHandle, KeyCallback);
        glfwSetMouseButtonCallback(m_WindowHandle, MouseButtonCallback);
        glfwSetCursorPosCallback(m_WindowHandle, CursorPosCallback);
        glfwSetScrollCallback(m_WindowHandle, ScrollCallback);
    }

    void Window::PollEvents()
    {
        m_CurrentSnapshot.scrollWheelDelta = 0.0;
        m_CurrentSnapshot.mouseDeltaX = 0.0;
        m_CurrentSnapshot.mouseDeltaY = 0.0;

        glfwPollEvents();

        m_CurrentSnapshot.mouseDeltaX = m_CurrentSnapshot.mouseX - m_LastMouseX;
        m_CurrentSnapshot.mouseDeltaY = m_CurrentSnapshot.mouseY - m_LastMouseY;
        m_LastMouseX = m_CurrentSnapshot.mouseX;
        m_LastMouseY = m_CurrentSnapshot.mouseY;

        ResolveInputStates();
    }

    void Window::WaitEvents()
    {
        glfwWaitEvents();
    }

    void Window::ResolveInputStates()
    {
        for (std::size_t i = 0; i < KeyCodeCount; i++)
        {
            if (m_KeyDown[i])
            {
                if (m_CurrentSnapshot.keyStates[i] == KeyState::Released)
                {
                    m_CurrentSnapshot.keyStates[i] = KeyState::Pressed;
                }
                else
                {
                    m_CurrentSnapshot.keyStates[i] = KeyState::Held;
                }
            }
            else
            {
                m_CurrentSnapshot.keyStates[i] = KeyState::Released;
            }
        }

        for (std::size_t i = 0; i < MouseButtonCount; i++)
        {
            if (m_MouseButtonDown[i])
            {
                if (m_CurrentSnapshot.mouseButtonStates[i] == KeyState::Released)
                {
                    m_CurrentSnapshot.mouseButtonStates[i] = KeyState::Pressed;
                }
                else
                {
                    m_CurrentSnapshot.mouseButtonStates[i] = KeyState::Held;
                }
            }
            else
            {
                m_CurrentSnapshot.mouseButtonStates[i] = KeyState::Released;
            }
        }
    }

    bool Window::ShouldClose() const
    {
        if (m_WindowHandle == nullptr)
        {
            return true;
        }

        return glfwWindowShouldClose(m_WindowHandle) == GLFW_TRUE;
    }

    void Window::Close()
    {
        if (m_WindowHandle == nullptr)
        {
            return;
        }

        glfwSetWindowShouldClose(m_WindowHandle, GLFW_TRUE);
    }

    void Window::SetTitle(const std::string& title)
    {
        m_Title = title;

        if (m_WindowHandle != nullptr)
        {
            glfwSetWindowTitle(m_WindowHandle, m_Title.c_str());
        }
    }

    glm::vec2 Window::GetMousePos() const
    {
        return glm::vec2{ static_cast<float>(m_CurrentSnapshot.mouseX), static_cast<float>(m_CurrentSnapshot.mouseY) };
    }

    KeyState Window::GetKeyboardKey(KeyCode key) const
    {
        const std::size_t index = static_cast<std::size_t>(key);

        if (index >= KeyCodeCount)
        {
            return KeyState::Released;
        }

        return m_CurrentSnapshot.keyStates[index];
    }

    KeyState Window::GetMouseKey(MouseButton button) const
    {
        const std::size_t index = static_cast<std::size_t>(button);

        if (index >= MouseButtonCount)
        {
            return KeyState::Released;
        }

        return m_CurrentSnapshot.mouseButtonStates[index];
    }

    bool Window::IsKeyDown(KeyCode key) const
    {
        const KeyState state = GetKeyboardKey(key);
        return state == KeyState::Pressed || state == KeyState::Held;
    }

    bool Window::IsKeyPressed(KeyCode key) const
    {
        return GetKeyboardKey(key) == KeyState::Pressed;
    }

    bool Window::IsMouseButtonDown(MouseButton button) const
    {
        const KeyState state = GetMouseKey(button);
        return state == KeyState::Pressed || state == KeyState::Held;
    }

    bool Window::IsMouseButtonPressed(MouseButton button) const
    {
        return GetMouseKey(button) == KeyState::Pressed;
    }

    Window* Window::FromHandle(GLFWwindow* window)
    {
        return static_cast<Window*>(glfwGetWindowUserPointer(window));
    }

    void Window::FramebufferResizeCallback(GLFWwindow* window, int width, int height)
    {
        Window* self = FromHandle(window);

        if (self == nullptr)
        {
            return;
        }

        self->m_Width = static_cast<uint32_t>(width);
        self->m_Height = static_cast<uint32_t>(height);
        self->m_Resized = true;
    }

    void Window::KeyCallback(GLFWwindow* window, int key, int, int action, int)
    {
        Window* self = FromHandle(window);

        if (self == nullptr)
        {
            return;
        }

        if (action == GLFW_REPEAT)
        {
            return;
        }

        const KeyCode code = TranslateKey(key);

        if (code == KeyCode::Unknown)
        {
            return;
        }

        const std::size_t index = static_cast<std::size_t>(code);

        if (action == GLFW_PRESS)
        {
            self->m_KeyDown[index] = true;
        }
        else
        {
            self->m_KeyDown[index] = false;
        }
    }

    void Window::MouseButtonCallback(GLFWwindow* window, int button, int action, int)
    {
        Window* self = FromHandle(window);

        if (self == nullptr)
        {
            return;
        }

        MouseButton mapped = MouseButton::Left;

        if (!TranslateMouseButton(button, mapped))
        {
            return;
        }

        const std::size_t index = static_cast<std::size_t>(mapped);

        if (action == GLFW_PRESS)
        {
            self->m_MouseButtonDown[index] = true;
        }
        else
        {
            self->m_MouseButtonDown[index] = false;
        }
    }

    void Window::CursorPosCallback(GLFWwindow* window, double xpos, double ypos)
    {
        Window* self = FromHandle(window);

        if (self == nullptr)
        {
            return;
        }

        self->m_CurrentSnapshot.mouseX = xpos;
        self->m_CurrentSnapshot.mouseY = ypos;
    }

    void Window::ScrollCallback(GLFWwindow* window, double, double yoffset)
    {
        Window* self = FromHandle(window);

        if (self == nullptr)
        {
            return;
        }

        self->m_CurrentSnapshot.scrollWheelDelta += yoffset;
    }

} // namespace Core