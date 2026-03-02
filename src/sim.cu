#include "Simulation.cuh"
#include <cuda_runtime.h>
#include <vector>

__global__ void updateParticles(
    float3 *positions,
    float3 *velocities,
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
        positions[idx].z += velocities[idx].z * dt;

        if (positions[idx].x < -1.0f)
        {
            positions[idx].x = -1.0f;
            velocities[idx].x *= -0.8f;
        }
        if (positions[idx].x > 1.0f)
        {
            positions[idx].x = 1.0f;
            velocities[idx].x *= -0.8f;
        }

        // Bounce floor
        if (positions[idx].y < -1.0f)
        {
            positions[idx].y = -1.0f;
            velocities[idx].y *= -0.8f;
        }
        if (positions[idx].y > 1.0f)
        {
            positions[idx].y = 1.0f;
            velocities[idx].y *= -0.8f;
        }

        if (positions[idx].z < -1.0f)
        {
            positions[idx].z = -1.0f;
            velocities[idx].z *= -0.8f;
        }
        if (positions[idx].z > 1.0f)
        {
            positions[idx].z = 1.0f;
            velocities[idx].z *= -0.8f;
        }
    }
}

Simulation::Simulation(int n) : n(n), vboResource(nullptr)
{
    cudaMalloc(&d_vel, n * sizeof(float3));

    cudaMemset(d_vel, 0, n * sizeof(float3));
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
    if (!vboResource)
    {
        return;
    }

    float3 *d_positions;

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

void Simulation::downloadPositions(float *posX, float *posY, float *posZ)
{
    if (!vboResource || !posX || !posY || !posZ)
    {
        return;
    }

    float3 *d_positions;
    size_t num_bytes;

    cudaGraphicsMapResources(1, &vboResource, 0);
    cudaGraphicsResourceGetMappedPointer(
        (void **)&d_positions,
        &num_bytes,
        vboResource);

    std::vector<float3> h_positions(n);
    cudaMemcpy(h_positions.data(), d_positions, n * sizeof(float3), cudaMemcpyDeviceToHost);

    cudaGraphicsUnmapResources(1, &vboResource, 0);

    for (int i = 0; i < n; ++i)
    {
        posX[i] = h_positions[i].x;
        posY[i] = h_positions[i].y;
        posZ[i] = h_positions[i].z;
    }
}