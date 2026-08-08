#version 450

layout(set = 2, binding = 0) uniform sampler2D u_Atlas;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

void main()
{
    float coverage = texture(u_Atlas, fragUV).r;
    outColor = vec4(fragColor.rgb, fragColor.a * coverage);
}
