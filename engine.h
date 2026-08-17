#pragma once

#include <vector>
#include <memory>
#include "ecs/ecs.hpp"
#include "renderer.h"
#include "input.h"
#include "scene.h"
#include "components/components_common.h"
#include "transformAccess.h"

class Engine {
public:
	Engine(Registry& registry);
	~Engine();

	void run();
	inline Renderer* getRenderer() { return &m_renderer; }
	inline Input* getInput() { return &m_input; }
	inline TransformAccess* getTransforms() { return &m_transforms; }

	template<typename T>
	void BindSystem() {
		m_systems.push_back(std::make_unique<T>(*this, m_registry));
	}

	void SetParent(Entity parent, Entity child);

	void addTransform(Entity entity, TransData transform, Entity parent = INVALID_ENTITY);

private:
	GLFWwindow* m_window = nullptr;
	bool m_glfwInitialized = false;

	Registry& m_registry;
	TransformAccess m_transforms;

	Entity m_camera;
	Entity m_directionalLight;
	Scene m_mainScene;
	Renderer m_renderer;
	Input m_input;

	void initWindow();
	void mainLoop();
	void cleanup();
	Entity addCamera();
	Entity addLight();
	void onFramebufferResized();
	static GLFWwindow* createWindow();
	static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

	// ecs
	std::vector<std::unique_ptr<BaseSystem>> m_systems;

	std::chrono::time_point<std::chrono::high_resolution_clock> m_lastTime;
};
