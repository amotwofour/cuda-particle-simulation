#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

struct Particle {
    float3 position;
    float3 velocity;
    float mass;
};

void initializeParticles(Particle* particles, int count, float spaceSize, float massRange);

#endif