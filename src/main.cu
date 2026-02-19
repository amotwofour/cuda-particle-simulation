#include <iostream>
#include <cuda_runtime.h>
#include "particle.hpp"

__global__ void updateParticles(Particle* particles, int n, float dt, float G);

void initializeParticles(Particle* particles, int count, float spaceSize, float massRange);

int main() {
    const int numParticles = 1024;
    const float spaceSize = 100.0f;
    const float massRange = 10.0f;
    const float dt = 0.01f;
    const float G = 6.67430e-11f;
    const int steps = 1000;

    Particle* h_particles = new Particle[numParticles];

    initializeParticles(h_particles, numParticles, spaceSize, massRange);

    Particle* d_particles = nullptr;
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));

    //copy particles from host to device
    cudaMemcpy(d_particles, h_particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);

    //sim loop
    int threadsPerBlock = 256;
    int blocks = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    for (int i = 0; i < steps; ++i) {
        updateParticles<<<blocks, threadsPerBlock>>>(d_particles, numParticles, dt, G);
        cudaDeviceSynchronize(); // Ensure each step completes before the next
    }

    //copy final particle states back to host
    cudaMemcpy(h_particles, d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);

    //print the final positions
    std::cout << "Final positions:\n";
    for (int i = 0; i < 10; ++i) { //print first 10 for brevity
        const auto& p = h_particles[i];
        std::cout << "Particle " << i << ": ("
                  << p.position.x << ", "
                  << p.position.y << ", "
                  << p.position.z << ")\n";
    }

    // Cleanup
    cudaFree(d_particles);
    delete[] h_particles;

    return 0;
}
