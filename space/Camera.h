#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace Core
{

    class Camera
    {
    public:
        Camera() = default;
        Camera(const glm::vec3& position, const glm::quat& orientation);

        void SetPosition(const glm::vec3& position) { m_Position = position; }
        void SetOrientation(const glm::quat& orientation);
        void LookAt(const glm::vec3& target, const glm::vec3& up = { 0.0f, 1.0f, 0.0f });

        void Move(const glm::vec3& localOffset);
        void MoveForward(float distance);
        void MoveRight(float distance);
        void MoveUp(float distance);

        void MoveWorld(const glm::vec3& worldOffset) { m_Position += worldOffset; }

        void Rotate(const glm::quat& delta) { m_Orientation = glm::normalize(delta * m_Orientation); }
        void RotateLocal(const glm::quat& delta) { m_Orientation = glm::normalize(m_Orientation * delta); }
        void RotateAxis(const glm::vec3& axis, float radians) { Rotate(glm::angleAxis(radians, axis)); }

      
        void Yaw(float radians, const glm::vec3& worldUp = { 0.0f, 1.0f, 0.0f });
        void Pitch(float radians, float maxPitchRadians = glm::radians(89.0f));
        void Roll(float radians);

        void SetPerspective(float verticalFovRadians, float aspectRatio, float nearPlane, float farPlane);

        const glm::vec3& GetPosition() const { return m_Position; }
        const glm::quat& GetOrientation() const { return m_Orientation; }

        glm::vec3 GetForward() const { return m_Orientation * glm::vec3(0.0f, 0.0f, -1.0f); }
        glm::vec3 GetRight() const { return m_Orientation * glm::vec3(1.0f, 0.0f, 0.0f); }
        glm::vec3 GetUp() const { return m_Orientation * glm::vec3(0.0f, 1.0f, 0.0f); }

        glm::mat4 GetViewMatrix() const;
        const glm::mat4& GetProjectionMatrix() const { return m_Projection; }

    private:
        void SyncPitchFromOrientation();

    private:
        glm::vec3 m_Position{ 0.0f, 0.0f, 0.0f };
        glm::quat m_Orientation{ 1.0f, 0.0f, 0.0f, 0.0f };

        glm::mat4 m_Projection{ 1.0f };

        float m_Pitch{ 0.0f };
    };

} // namespace Core