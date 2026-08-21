#version 450

// glslc particle.frag -o particle_frag.spv

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
    float u_ParticleScale;
    vec3 u_FastColor;
    float u_SpeedRange;
};

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragWorldPosition;
layout(location = 2) in float fragSpeed01;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 normal = normalize(fragNormal);
    vec3 lightDirection = normalize(vec3(0.4, 0.9, 0.35));
    vec3 viewDirection = normalize(u_CameraPosition.xyz - fragWorldPosition);

    // Shading a particle system by speed makes the simulation readable at a glance: slow cores
    // stay base-coloured while fast ejecta light up.
    vec3 albedo = mix(u_AlbedoColor, u_FastColor, fragSpeed01);

    float diffuse = max(dot(normal, lightDirection), 0.0);
    float rim = pow(1.0 - max(dot(normal, viewDirection), 0.0), 3.0);

    vec3 ambient = albedo * 0.12;
    vec3 lit = albedo * diffuse + albedo * rim * 0.35;

    outColor = vec4(ambient + lit, 1.0);
}
