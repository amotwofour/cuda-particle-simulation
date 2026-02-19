#include <cuda_runtime.h>
#include "particle.hpp"
#include <crt/math_functions.h>
#include <device_launch_parameters.h>

__global__ void updateParticles(Particle* particles, int n, float dt, float G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle p = particles[i];
    float3 force = make_float3(0.0f, 0.0f, 0.0f);

    for (int j = 0; j < n; ++j) {
        if (i == j) continue;

        Particle other = particles[j];
        
        float3 diff = make_float3(
            other.position.x - p.position.x,
            other.position.y - p.position.y,
            other.position.z - p.position.z
        );

        float distSqr = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + 1e-6f; // prevent division by 0
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        float3 f = make_float3(
            G * other.mass * diff.x * invDist3,
            G * other.mass * diff.y * invDist3,
            G * other.mass * diff.z * invDist3
        );

        force.x += f.x;
        force.y += f.y;
        force.z += f.z;
    }

    p.velocity.x += force.x * dt;
    p.velocity.y += force.y * dt;
    p.velocity.z += force.z * dt;

    p.position.x += p.velocity.x * dt;
    p.position.y += p.velocity.y * dt;
    p.position.z += p.velocity.z * dt;

    particles[i] = p;
}