#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include "engine.h"
#include "model.h"

const std::vector<Vertex> vertices = {
	{{-0.5f, 0.0f, -0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, 0.0f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
	{{0.5f, 0.0f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.0f, 0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}
};

const std::vector<uint16_t> indices = {
	0, 1, 2, 2, 3, 0
};

std::pair<std::vector<Vertex>, std::vector<uint16_t>> loadModel(const char* path) {
	Model model(path);
	std::vector<Vertex> modelVertices;
	std::vector<uint16_t> modelIndices;
	modelVertices.reserve(model.nfaces() * 3);
	modelIndices.reserve(model.nfaces() * 3);

	std::map<std::pair<int, int>, uint16_t> uniqueVertices;
	for (int i = 0; i < model.nfaces(); i++) {
		for (int j = 0; j < 3; j++) {
			int vertIndex = model.vertIndex(i, j);
			int normalIndex = model.normalIndex(i, j);
			auto key = std::make_pair(vertIndex, normalIndex);
			auto it = uniqueVertices.find(key);

			if (it == uniqueVertices.end()) {
				float t = static_cast<float>(vertIndex) / static_cast<float>(model.nverts());
				uint16_t newIndex = static_cast<uint16_t>(modelVertices.size());
				modelVertices.emplace_back(model.vert(i, j), model.normal(i, j), glm::vec3{t, t, t});
				it = uniqueVertices.emplace(key, newIndex).first;
			}

			modelIndices.push_back(it->second);
		}
	}

	return {std::move(modelVertices), std::move(modelIndices)};
}

int main() {
	try {
		auto [modelVertices, modelIndices] = loadModel("donut.obj");
		Engine app(std::move(modelVertices), std::move(modelIndices));
		app.run();
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
