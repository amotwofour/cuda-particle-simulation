#include "Simulation.cuh"
#include <cuda_runtime.h>

__global__ void updateParticles(
    float2 *positions,
    float2 *velocities,
    float dt,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        // Gravity
        velocities[idx].y += -9.81f * dt;

        // Integrate motion
        positions[idx].x += velocities[idx].x * dt;
        positions[idx].y += velocities[idx].y * dt;

        // Bounce floor
        if (positions[idx].y < -1.0f)
        {
            positions[idx].y = -1.0f;
            velocities[idx].y *= -0.8f;
        }
    }
}

Simulation::Simulation(int n) : n(n)
{
    cudaMalloc(&d_vel, n * sizeof(float2));

    cudaMemset(d_vel, 0, n * sizeof(float2));
}

Simulation::~Simulation()
{
    cudaFree(d_vel);
}

void Simulation::setVBO(cudaGraphicsResource *vboResource)
{
    this->vboResource = vboResource;
}

void Simulation::step(float dt)
{
    float2 *d_positions;

    size_t num_bytes;

    cudaGraphicsMapResources(1, &vboResource, 0);

    cudaGraphicsResourceGetMappedPointer(
        (void **)&d_positions,
        &num_bytes,
        vboResource);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    updateParticles<<<blocks, threads>>>(
        d_positions,
        d_vel,
        dt,
        n);

    cudaGraphicsUnmapResources(1, &vboResource, 0);
}