#pragma once
#include <cuda_gl_interop.h>

class Simulation {
    public:
        Simulation(int n);
        ~Simulation();

        void step(float dt);
        void downloadPositions(float* posX, float* posY, float* posZ);
        void setVBO(struct cudaGraphicsResource* vboResource);

        private:
            int n;

            /*
            float* d_posX;
            float* d_posY;
            float* d_velX;
            float* d_velY;
            */
            float3* d_vel;

            cudaGraphicsResource* vboResource;
};