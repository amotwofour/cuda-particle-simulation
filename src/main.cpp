#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <string>
#include <GLFW/glfw3.h>

#include "Simulation.cuh"

int main (int argc, char** argv) {
    int particleCount = 5000;
    if (argc > 1) {
        try {
            const int parsedValue = std::stoi(argv[1]);
            if (parsedValue > 0) {
                particleCount = parsedValue;
            }
        } catch (...) {
            std::cerr << "Invalid particle count argument, using default 5000" << std::endl;
        }
    }

    const int minParticles = 100;
    const int maxParticles = 200000;
    const int particleStep = 1000;
    const float dt = 0.01f;
    
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_DEPTH_BITS, 24);
    GLFWwindow* window = glfwCreateWindow(800, 600, "CUDA Particle Simulation", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        std::cerr << "Failed to create GLFW window" << std::endl;
        return -1;
    }

    glfwMakeContextCurrent(window);
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

    while (!glfwWindowShouldClose(window)) {
        const double currentTime = glfwGetTime();
        const float frameDelta = static_cast<float>(currentTime - lastFrameTime);
        lastFrameTime = currentTime;

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        const bool increasePressed =
            glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_KP_ADD) == GLFW_PRESS;
        const bool decreasePressed =
            glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS;

        if (increasePressed && !increaseKeyHeld) {
            if (particleCount < maxParticles) {
                particleCount += particleStep;
                if (particleCount > maxParticles) {
                    particleCount = maxParticles;
                }
                rebuildSimulation();
            }
        }
        if (decreasePressed && !decreaseKeyHeld) {
            if (particleCount > minParticles) {
                particleCount -= particleStep;
                if (particleCount < minParticles) {
                    particleCount = minParticles;
                }
                rebuildSimulation();
            }
        }
        increaseKeyHeld = increasePressed;
        decreaseKeyHeld = decreasePressed;

        double mouseX = 0.0;
        double mouseY = 0.0;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        const float mouseSensitivity = 0.1f;
        const float xOffset = static_cast<float>(mouseX - lastMouseX) * mouseSensitivity;
        const float yOffset = static_cast<float>(lastMouseY - mouseY) * mouseSensitivity;
        lastMouseX = mouseX;
        lastMouseY = mouseY;

        yawDegrees += xOffset;
        pitchDegrees += yOffset;
        if (pitchDegrees > 89.0f) {
            pitchDegrees = 89.0f;
        }
        if (pitchDegrees < -89.0f) {
            pitchDegrees = -89.0f;
        }

        const float yawRadians = yawDegrees * 3.14159265359f / 180.0f;
        const float pitchRadians = pitchDegrees * 3.14159265359f / 180.0f;
        float frontX = cosf(yawRadians) * cosf(pitchRadians);
        float frontY = sinf(pitchRadians);
        float frontZ = sinf(yawRadians) * cosf(pitchRadians);
        const float frontLength = sqrtf(frontX * frontX + frontY * frontY + frontZ * frontZ);
        if (frontLength > 0.0f) {
            frontX /= frontLength;
            frontY /= frontLength;
            frontZ /= frontLength;
        }

        float rightX = -frontZ;
        float rightY = 0.0f;
        float rightZ = frontX;
        const float rightLength = sqrtf(rightX * rightX + rightY * rightY + rightZ * rightZ);
        if (rightLength > 0.0f) {
            rightX /= rightLength;
            rightY /= rightLength;
            rightZ /= rightLength;
        }

        const float movementSpeed = 2.5f * frameDelta;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            cameraX += frontX * movementSpeed;
            cameraY += frontY * movementSpeed;
            cameraZ += frontZ * movementSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            cameraX -= frontX * movementSpeed;
            cameraY -= frontY * movementSpeed;
            cameraZ -= frontZ * movementSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            cameraX -= rightX * movementSpeed;
            cameraY -= rightY * movementSpeed;
            cameraZ -= rightZ * movementSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            cameraX += rightX * movementSpeed;
            cameraY += rightY * movementSpeed;
            cameraZ += rightZ * movementSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            cameraY += movementSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
            cameraY -= movementSpeed;
        }

        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);

        const float aspect = (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
        const float fovYDegrees = 60.0f;
        const float nearPlane = 0.1f;
        const float farPlane = 100.0f;
        const float top = nearPlane * tanf(fovYDegrees * 3.14159265359f / 360.0f);
        const float right = top * aspect;

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glFrustum(-right, right, -top, top, nearPlane, farPlane);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glRotatef(-pitchDegrees, 1.0f, 0.0f, 0.0f);
        glRotatef(-yawDegrees, 0.0f, 1.0f, 0.0f);
        glTranslatef(-cameraX, -cameraY, -cameraZ);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        sim->step(dt);
        sim->downloadPositions(posX.data(), posY.data(), posZ.data());

        glBegin(GL_POINTS);
        for (int i = 0; i < particleCount; i++)
            glVertex3f(posX[i], posY[i], posZ[i]);

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