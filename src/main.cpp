#include <iostream>
#include <vector>
#include <GLFW/glfw3.h>

#include "Simulation.cuh"

int main () {
    const int n = 5000;
    const float dt = 0.01f;
    
    
    Simulation sim(n);
    std::vector<float> posX(n), posY(n);
    
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 600, "CUDA Particle Simulation", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glPointSize(3.0f);

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);
        sim.step(dt);
        sim.downloadPositions(posX.data(), posY.data());

        for (int i = 0; i < n; i++)
            glVertex2f(posX[i], posY[i]);

        glEnd();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    /*
     * sim.downloadPositions(posX.data(), posY.data());
     * std::cout << "Particle 0 position: (" << posX[0] << ", " << posY[0] << ")" << std::endl;
     * std::cout << "Simulation completed." << std::endl;
     * return 0;
    */
    
}