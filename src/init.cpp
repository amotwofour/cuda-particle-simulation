#include "particle.hpp"
#include <cstdlib>
#include <ctime>
#include <cmath>

static float randFloat(float min, float max) {
    return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
}

void initializeParticles(Particle* particles, int count, float spaceSize, float massRange) {
    srand(static_cast<unsigned int>(time(nullptr)));

    for (int i = 0; i < count; ++i) {
        particles[i].position = make_float3(
            randFloat(-spaceSize, spaceSize),
            randFloat(-spaceSize, spaceSize),
            randFloat(-spaceSize, spaceSize)
        );

        particles[i].velocity = make_float3(0.0f, 0.0f, 0.0f);

        particles[i].mass = randFloat(1.0f, massRange);
    }
}