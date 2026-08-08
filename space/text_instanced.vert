#version 450

layout(set = 0, binding = 0) uniform CameraBuffer
{
    mat4 u_View;
    mat4 u_Projection;
    mat4 u_ViewProjection;
    vec4 u_CameraPosition;
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 3) in vec4 inModelCol0;
layout(location = 4) in vec4 inModelCol1;
layout(location = 5) in vec4 inModelCol2;
layout(location = 6) in vec4 inModelCol3;
layout(location = 7) in vec2 inUVOffset;
layout(location = 8) in vec2 inUVSize;
layout(location = 9) in vec4 inColor;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec4 fragColor;

void main()
{
    mat4 model = mat4(inModelCol0, inModelCol1, inModelCol2, inModelCol3);

    vec4 worldPosition = model * vec4(inPosition, 1.0);

    vec4 clipPos = u_ViewProjection * worldPosition;

    clipPos.y = -clipPos.y;

    gl_Position = clipPos;

    fragUV = inUVOffset + inUV * inUVSize;
    fragColor = inColor;
}
