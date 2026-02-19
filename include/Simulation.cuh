#pragma once

class Simulation {
    public:
        Simulation(int n);
        ~Simulation();

        void step(float dt);
        void downloadPositions(float* posX, float* posY);
    
        private:
            int n;

            float* d_posX;
            float* d_posY;
            float* d_velX;
            float* d_velY;
};