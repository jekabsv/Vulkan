#version 450

// glslc particle.vert -o particle_vert.spv
//
// Instanced particle rendering with no per-instance vertex buffer and no CPU-side instance list:
// the position comes straight out of the storage buffer the compute pass just wrote, indexed by
// gl_InstanceIndex. One DrawMeshInstanced call draws the whole system, and the data never leaves
// device memory.

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

struct Particle
{
    vec4 positionMass;  // xyz = position, w = mass
    vec4 velocity;      // xyz = velocity, w = unused
};

// Given an instance name on purpose: Shader.cpp reflects a binding's name from the SPIR-V
// variable and only falls back to the block's type name when that is empty, so naming the
// instance is what makes "u_Particles" the stable key for Material::SetStorageBuffer.
layout(std430, set = 2, binding = 1) readonly buffer ParticleBuffer
{
    Particle particles[];
} u_Particles;

// The mesh is the unit sphere from Mesh::CreateSphere. Build it with radius 1.0 for particles:
// whatever radius it was generated at multiplies into u_ParticleScale below.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragWorldPosition;
layout(location = 2) out float fragSpeed01;

void main()
{
    Particle particle = u_Particles.particles[gl_InstanceIndex];

    vec3 center = particle.positionMass.xyz;
    float mass = particle.positionMass.w;

    // Cube root so that drawn volume tracks mass rather than radius tracking it directly, which
    // would make a heavy body swamp the screen.
    float radius = u_ParticleScale * pow(max(mass, 1e-6), 1.0 / 3.0);

    vec3 worldPosition = center + inPosition * radius;

    vec4 clipPosition = u_ViewProjection * vec4(worldPosition, 1.0);
    clipPosition.y = -clipPosition.y;

    gl_Position = clipPosition;

    // Uniform scale, so the normal survives untransformed.
    fragNormal = inNormal;
    fragWorldPosition = worldPosition;
    fragSpeed01 = clamp(length(particle.velocity.xyz) / max(u_SpeedRange, 1e-6), 0.0, 1.0);
}
