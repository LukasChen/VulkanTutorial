#pragma once
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#include <glm/glm.hpp>
#include <array>


struct Vertex {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 uv;

	Vertex(glm::vec3 p, glm::vec3 n, glm::vec2 t = {0.0f, 0.0f}) : pos(p), normal(n), uv(t) {}

	static vk::VertexInputBindingDescription getBindingDescription() {
		return {
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = vk::VertexInputRate::eVertex
		};
	}

	static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
		return {{
			{.location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, pos)},
			{.location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, normal)},
			{.location = 2, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(Vertex, uv)}
		}};
	}
};
