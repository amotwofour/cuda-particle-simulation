#include <iostream>
#include <vector>

#include "Simulation.cuh"

int main () {
    const int n = 100000;
    const float dt = 0.01f;
    
    std::vector<float> posX(n), posY(n);
    Simulation sim(n);

    for (int i = 0; i < 100; i++) {
        sim.step(dt);
    }

    sim.downloadPositions(posX.data(), posY.data());
    std::cout << "Particle 0 position: (" << posX[0] << ", " << posY[0] << ")" << std::endl;

    std::cout << "Simulation completed." << std::endl;
    return 0;
}