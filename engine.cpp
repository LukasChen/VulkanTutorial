#include "engine.h"

#include "components/components_common.h"

Engine::Engine(Registry& registry)
	: m_window(createWindow()),
	  m_glfwInitialized(true),
	  m_registry(registry),
	  m_camera(addCamera()),
	  m_renderer(m_window, m_registry, m_camera) {
	initWindow();
}

Engine::~Engine() {
	cleanup();
}

void Engine::run() {
	mainLoop();
}

GLFWwindow* Engine::createWindow() {
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	return glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
}

void Engine::initWindow() {
	glfwSetWindowUserPointer(m_window, this);
	glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);
}

void Engine::framebufferResizeCallback(GLFWwindow* window, int, int) {
	auto* app = reinterpret_cast<Engine*>(glfwGetWindowUserPointer(window));
	if (app != nullptr) {
		app->onFramebufferResized();
	}
}

void Engine::mainLoop() {
	while (!glfwWindowShouldClose(m_window)) {
		glfwPollEvents();
		m_renderer.drawFrame();
	}
}

Entity Engine::addCamera() {
	const Entity camera = m_registry.create();
	m_registry.get<Transform>().addComponent(camera, Transform{
		glm::vec3(0.0f, 0.0f, -2.0f),
		glm::vec3(glm::radians(30.0f), 0.0f, 0.0f)
	});
	return camera;
}

void Engine::onFramebufferResized() {
	m_renderer.onFramebufferResized();
}

void Engine::cleanup() {
	if (m_window != nullptr) {
		glfwDestroyWindow(m_window);
		m_window = nullptr;
	}

	if (m_glfwInitialized) {
		glfwTerminate();
		m_glfwInitialized = false;
	}
}
