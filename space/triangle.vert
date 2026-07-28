#version 450

layout(set = 0, binding = 0) uniform CameraBuffer
{
    mat4 u_View;
    mat4 u_Projection;
    mat4 u_ViewProjection;
    vec4 u_CameraPosition;
};

layout(push_constant) uniform PushConstants
{
    mat4 u_Model;
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec3 fragWorldPosition;

void main()
{
    vec4 worldPosition = u_Model * vec4(inPosition, 1.0);

    vec4 clipPos = u_ViewProjection * worldPosition;
    
    clipPos.y = -clipPos.y;

    gl_Position = clipPos;

    fragNormal = mat3(u_Model) * inNormal;
    fragUV = inUV;
    fragWorldPosition = worldPosition.xyz;
}
