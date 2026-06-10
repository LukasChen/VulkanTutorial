#pragma once

#include "renderer.h"

class Engine {
public:
	Engine(Registry& registry);
	~Engine();

	void run();

	inline Renderer* getRenderer() { return &m_renderer; }

private:
	GLFWwindow* m_window = nullptr;
	bool m_glfwInitialized = false;

	Registry& m_registry;
	Entity m_camera;
	Renderer m_renderer;

	void initWindow();
	void mainLoop();
	void cleanup();
	Entity addCamera();
	void onFramebufferResized();
	static GLFWwindow* createWindow();

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
};
