#include <iostream>
#include <vector>
#include <cstdlib>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include "../include/Simulation.cuh"

int main()
{
    const int n = 5000;
    const float dt = 0.01f;

    Simulation sim(n);

    std::vector<float> posX(n);
    std::vector<float> posY(n);

    // Initialize GLFW
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    GLFWwindow *window =
        glfwCreateWindow(
            800,
            600,
            "CUDA Particle Simulation",
            nullptr,
            nullptr);

    if (!window)
    {
        std::cerr << "Failed to create window\n";
        return -1;
    }

    glfwMakeContextCurrent(window);

    //
    // Initialize GLEW (required for VBOs)
    //

    glewInit();

    //
    // OpenGL Setup
    //

    glViewport(0, 0, 800, 600);

    glPointSize(3.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    std::unique_ptr<Simulation> sim(new Simulation(particleCount));
    std::vector<float> posX(particleCount), posY(particleCount), posZ(particleCount);

    auto refreshWindowTitle = [&]() {
        const std::string title = "CUDA Particle Simulation - Count: " + std::to_string(particleCount) + " | +/- to adjust";
        glfwSetWindowTitle(window, title.c_str());
    };

    auto rebuildSimulation = [&]() {
        sim.reset(new Simulation(particleCount));
        posX.assign(particleCount, 0.0f);
        posY.assign(particleCount, 0.0f);
        posZ.assign(particleCount, 0.0f);
        refreshWindowTitle();
    };

    refreshWindowTitle();
    bool increaseKeyHeld = false;
    bool decreaseKeyHeld = false;

    float cameraX = 0.0f;
    float cameraY = 0.0f;
    float cameraZ = 3.5f;
    float yawDegrees = -90.0f;
    float pitchDegrees = 0.0f;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
    glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
    double lastFrameTime = glfwGetTime();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);

    //
    // Main Loop
    //

    while (!glfwWindowShouldClose(window))
    {

        glClear(GL_COLOR_BUFFER_BIT);

        sim.step(dt);

        sim.downloadPositions(
            posX.data(),
            posY.data());

        glColor3f(1.0f, 1.0f, 1.0f);

        glBegin(GL_POINTS);

        for (int i = 0; i < n; i++)
        {
            glVertex2f(
                posX[i],
                posY[i]);
        }

        glEnd();

        glfwSwapBuffers(window);

        glfwPollEvents();
    }

    glfwTerminate();

    return 0;
}