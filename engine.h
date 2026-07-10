#pragma once

#include <vector>
#include <memory>
#include "ecs/ecs.hpp"
#include "renderer.h"
#include "input.h"

class Engine {
public:
	Engine(Registry& registry);
	~Engine();

	void run();
	inline Renderer* getRenderer() { return &m_renderer; }
	inline Input* getInput() { return &m_input; }

	template<typename T>
	void BindSystem() {
		m_systems.push_back(std::make_unique<T>(*this, m_registry));
	}

private:
	GLFWwindow* m_window = nullptr;
	bool m_glfwInitialized = false;

	Registry& m_registry;

	Entity m_camera;
	Renderer m_renderer;
	Input m_input;

	void initWindow();
	void mainLoop();
	void cleanup();
	Entity addCamera();
	void onFramebufferResized();
	static GLFWwindow* createWindow();
	static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

	// ecs
	std::vector<std::unique_ptr<BaseSystem>> m_systems;

	std::chrono::time_point<std::chrono::high_resolution_clock> m_lastTime;
};
