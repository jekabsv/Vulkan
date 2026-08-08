#version 450

layout(set = 2, binding = 0) uniform MaterialBuffer
{
    vec4 u_Color;
};

layout(set = 2, binding = 1) uniform sampler2D u_Atlas;

layout(location = 0) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

void main()
{
    // Plain bitmap glyph atlas (no SDF): the red channel is used as a coverage/alpha mask.
    float coverage = texture(u_Atlas, fragUV).r;
    outColor = vec4(u_Color.rgb, u_Color.a * coverage);
}
