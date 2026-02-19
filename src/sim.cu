#include "Simulation.cuh"
#include <cuda_runtime.h>
#include <vector>

__global__ void updateParticles(float *posX, float *posY,
                                float *velX, float *velY,
                                float dt, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float g = -9.81f; // Gravitational acceleration
        velY[idx] += g * dt; // Update velocity with gravity
        posX[idx] += velX[idx] * dt; // Update position X
        posY[idx] += velY[idx] * dt; // Update position Y
    }
}

Simulation::Simulation(int n) : n(n) {
    std::vector<float> h_posX(n);
    std::vector<float> h_posY(n);
    std::vector<float> h_velX(n);
    std::vector<float> h_velY(n);

    for (int i = 0; i < n; i++) {
        h_posX[i] = 0.0f; // Initial X position
        h_posY[i] = 10.0f; // Initial Y position
        h_velX[i] = 0.0f; // Initial X velocity
        h_velY[i] = 0.0f; // Initial Y velocity
    }
    
    cudaMalloc(&d_posX, n * sizeof(float));
    cudaMalloc(&d_posY, n * sizeof(float));
    cudaMalloc(&d_velX, n * sizeof(float));
    cudaMalloc(&d_velY, n * sizeof(float));

    cudaMemcpy(d_posX, h_posX.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posY, h_posY.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velX, h_velX.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velY, h_velY.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

Simulation::~Simulation() {
    cudaFree(d_posX);
    cudaFree(d_posY);
    cudaFree(d_velX);
    cudaFree(d_velY);
}

void Simulation::step(float dt) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    updateParticles<<<blocks, threads>>>(
        d_posX, d_posY, d_velX, d_velY, dt, n
    );

    cudaDeviceSynchronize(); // Wait for the kernel to finish
}

void Simulation::downloadPositions(float* posX, float* posY) {
    cudaMemcpy(posX, d_posX, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(posY, d_posY, n * sizeof(float), cudaMemcpyDeviceToHost);
}