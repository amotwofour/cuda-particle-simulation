#include "../include/Simulation.cuh"

#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>

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

        // Integrate
        positions[idx].x += velocities[idx].x * dt;
        positions[idx].y += velocities[idx].y * dt;

        // Floor bounce
        if (positions[idx].y < -1.0f)
        {
            positions[idx].y = -1.0f;
            velocities[idx].y *= -0.8f;
        }
    }
}

Simulation::Simulation(int n)
    : n(n)
{

    cudaMalloc(&d_pos, n * sizeof(float2));
    cudaMalloc(&d_vel, n * sizeof(float2));

    std::vector<float2> init(n);

    for (int i = 0; i < n; ++i)
    {
        init[i].x = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        init[i].y = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    cudaMemcpy(d_pos, init.data(), n * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemset(d_vel, 0, n * sizeof(float2));
}

Simulation::~Simulation()
{

    cudaFree(d_pos);
    cudaFree(d_vel);
}

void Simulation::step(float dt)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    updateParticles<<<blocks, threads>>>(
        d_pos,
        d_vel,
        dt,
        n);

    cudaDeviceSynchronize();
}

void Simulation::downloadPositions(
    float *posX,
    float *posY)
{
    std::vector<float2> h_positions(n);

    cudaMemcpy(
        h_positions.data(),
        d_pos,
        n * sizeof(float2),
        cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
    {

        posX[i] = h_positions[i].x;
        posY[i] = h_positions[i].y;
    }
}