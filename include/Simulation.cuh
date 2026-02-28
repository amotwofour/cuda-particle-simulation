#pragma once
#include <cuda_runtime.h>

class Simulation
{
public:
    Simulation(int n);
    ~Simulation();

    void step(float dt);

    void downloadPositions(float *posX, float *posY);

private:
    int n;

    float2 *d_pos;
    float2 *d_vel;
};