#include "Camera.h"

#include <glm/gtc/matrix_transform.hpp>

namespace Core
{

    Camera::Camera(const glm::vec3& position, const glm::quat& orientation)
        : m_Position(position)
        , m_Orientation(glm::normalize(orientation))
    {
        SyncPitchFromOrientation();
    }

    void Camera::SetOrientation(const glm::quat& orientation)
    {
        m_Orientation = glm::normalize(orientation);
        SyncPitchFromOrientation();
    }

    void Camera::LookAt(const glm::vec3& target, const glm::vec3& up)
    {
        const glm::vec3 forward = glm::normalize(target - m_Position);
        const glm::vec3 right = glm::normalize(glm::cross(forward, up));
        const glm::vec3 trueUp = glm::cross(right, forward);

        const glm::mat3 rotation(right, trueUp, -forward);
        m_Orientation = glm::normalize(glm::quat_cast(rotation));
        SyncPitchFromOrientation();
    }

    void Camera::Move(const glm::vec3& localOffset)
    {
        m_Position += GetRight() * localOffset.x + GetUp() * localOffset.y + GetForward() * localOffset.z;
    }

    void Camera::MoveForward(float distance)
    {
        m_Position += GetForward() * distance;
    }

    void Camera::MoveRight(float distance)
    {
        m_Position += GetRight() * distance;
    }

    void Camera::MoveUp(float distance)
    {
        m_Position += GetUp() * distance;
    }

    void Camera::Yaw(float radians, const glm::vec3& worldUp)
    {
        m_Orientation = glm::normalize(glm::angleAxis(radians, worldUp) * m_Orientation);
    }

    void Camera::Pitch(float radians, float maxPitchRadians)
    {
        const float clampedPitch = glm::clamp(m_Pitch + radians, -maxPitchRadians, maxPitchRadians);
        const float delta = clampedPitch - m_Pitch;
        m_Pitch = clampedPitch;

        m_Orientation = glm::normalize(m_Orientation * glm::angleAxis(delta, glm::vec3(1.0f, 0.0f, 0.0f)));
    }

    void Camera::Roll(float radians)
    {
        m_Orientation = glm::normalize(m_Orientation * glm::angleAxis(radians, glm::vec3(0.0f, 0.0f, -1.0f)));
    }

    void Camera::SetPerspective(float verticalFovRadians, float aspectRatio, float nearPlane, float farPlane)
    {
        m_Projection = glm::perspective(verticalFovRadians, aspectRatio, nearPlane, farPlane);
        m_Projection[1][1] *= -1.0f;
    }

    glm::mat4 Camera::GetViewMatrix() const
    {
        const glm::mat4 rotation = glm::mat4_cast(glm::conjugate(m_Orientation));
        const glm::mat4 translation = glm::translate(glm::mat4(1.0f), -m_Position);
        return rotation * translation;
    }

    void Camera::SyncPitchFromOrientation()
    {
        m_Pitch = glm::asin(glm::clamp(GetForward().y, -1.0f, 1.0f));
    }

} // namespace Core