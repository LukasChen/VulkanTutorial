#pragma once

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#define NOMINMAX
#include <GLFW/glfw3native.h>

enum class Axis {
    Horizontal,
    Vertical,
    Elevation,
    LookHorizontal,
    LookVertical
};

class Input {
public:
    Input(GLFWwindow* window);
    ~Input();

    bool isKeyPressed(int key);
    float getAxis(Axis axis);
private:
    GLFWwindow* m_window;
};
