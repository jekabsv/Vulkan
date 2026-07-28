#version 450

layout(set = 0, binding = 0) uniform CameraBuffer
{
    mat4 u_View;
    mat4 u_Projection;
    mat4 u_ViewProjection;
    vec4 u_CameraPosition;
};

layout(set = 2, binding = 0) uniform MaterialBuffer
{
    vec3 u_AlbedoColor;
    float u_Roughness;
    float u_Metallic;
    vec2 u_Tiling;
};

layout(set = 2, binding = 1) uniform sampler2D u_AlbedoMap;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragWorldPosition;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 normal = normalize(fragNormal);
    vec3 lightDirection = normalize(vec3(0.4, 0.9, 0.35));
    vec3 viewDirection = normalize(u_CameraPosition.xyz - fragWorldPosition);
    vec3 halfway = normalize(lightDirection + viewDirection);

    vec3 albedo = texture(u_AlbedoMap, fragUV * u_Tiling).rgb * u_AlbedoColor;

    float diffuse = max(dot(normal, lightDirection), 0.0);
    float shininess = mix(4.0, 128.0, 1.0 - u_Roughness);
    float specular = pow(max(dot(normal, halfway), 0.0), shininess) * (1.0 - u_Roughness);

    vec3 ambient = albedo * 0.08;
    vec3 lit = albedo * diffuse + mix(vec3(specular), albedo * specular, u_Metallic);

    outColor = vec4(ambient + lit, 1.0);
}
