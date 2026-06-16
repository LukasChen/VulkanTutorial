#include "input.h"

#include <iostream>

Input::Input(GLFWwindow* window) : m_window(window) {}

Input::~Input() = default;

bool Input::isKeyPressed(int key) {
    return glfwGetKey(m_window, key) == GLFW_PRESS;
}

float Input::getAxis(Axis axis) {
    switch(axis) {
        case Axis::Horizontal:
            return (isKeyPressed(GLFW_KEY_D) ? 1.0f : 0.0f) - (isKeyPressed(GLFW_KEY_A) ? 1.0f : 0.0f);
        case Axis::Vertical:
            return (isKeyPressed(GLFW_KEY_W) ? 1.0f : 0.0f) - (isKeyPressed(GLFW_KEY_S) ? 1.0f : 0.0f);
        case Axis::LookHorizontal:
            return (isKeyPressed(GLFW_KEY_RIGHT) ? 1.0f : 0.0f) - (isKeyPressed(GLFW_KEY_LEFT) ? 1.0f : 0.0f);
        case Axis::LookVertical:
            return (isKeyPressed(GLFW_KEY_UP) ? 1.0f : 0.0f) - (isKeyPressed(GLFW_KEY_DOWN) ? 1.0f : 0.0f);
        default:
            return 0.0f;
    }
}
