#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

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

#include "ecs/ecs.hpp"

inline constexpr uint32_t WIDTH = 800;
inline constexpr uint32_t HEIGHT = 600;
inline constexpr int MAX_FRAMES_IN_FLIGHT = 2;
inline const std::vector<char const*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
inline constexpr bool enableValidationLayers = false;
#else
inline constexpr bool enableValidationLayers = true;
#endif

struct Vertex {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec3 color;

	Vertex(glm::vec3 p, glm::vec3 n, glm::vec3 c) : pos(p), normal(n), color(c) {}

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
			{.location = 2, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, color)}
		}};
	}
};

struct UniformBufferObject {
	glm::mat4 m;
	glm::mat4 vp;
	glm::mat3 n;
};

struct EntityResources {
	vk::raii::Buffer uniformBuffer = nullptr;
	vk::raii::DeviceMemory uniformBufferMemory = nullptr;
	void* uniformBufferMapped = nullptr;
	vk::raii::DescriptorSet descriptorSet = nullptr;
};

struct MeshResources {
	vk::raii::Buffer vertexBuffer = nullptr;
	vk::raii::DeviceMemory vertexBufferMemory = nullptr;
	vk::raii::Buffer indexBuffer = nullptr;
	vk::raii::DeviceMemory indexBufferMemory = nullptr;
	uint16_t indiceSize;
};

struct FrameData {
	vk::raii::CommandBuffer commandBuffer = nullptr;
	vk::raii::Semaphore presentCompleteSemaphore = nullptr;
	vk::raii::Fence inFlightFence = nullptr;
	std::unordered_map<Entity, EntityResources> entityResources;
};

struct SwapchainData {
	vk::Image image = nullptr;
	vk::raii::ImageView imageView = nullptr;
	vk::raii::Image depthImage = nullptr;
	vk::raii::DeviceMemory depthImageMemory = nullptr;
	vk::raii::ImageView depthImageView = nullptr;
	vk::raii::Semaphore renderFinishedSemaphore = nullptr;
};

class Renderer {
public:
	Renderer(GLFWwindow* window, Registry& registry, Entity camera);
	~Renderer();

	void createMeshEntity(Entity entity, size_t meshHandle);
	size_t uploadMesh(const std::pair<std::vector<Vertex>, std::vector<uint16_t>>& meshData);
	void drawFrame();
	void onFramebufferResized();

private:
	GLFWwindow* m_window = nullptr;

	vk::raii::Context m_context;
	vk::raii::Instance m_instance = nullptr;
	vk::raii::PhysicalDevice m_physicalDevice = nullptr;
	vk::raii::Device m_device = nullptr;
	vk::raii::Queue m_graphicsQueue = nullptr;
	vk::raii::Queue m_transferQueue = nullptr;
	vk::raii::SurfaceKHR m_surface = nullptr;
	vk::raii::SwapchainKHR m_swapChain = nullptr;
	uint32_t m_graphicsQueueIndex = ~0u;
	uint32_t m_transferQueueIndex = ~0u;
	vk::Extent2D m_swapChainExtent;
	vk::SurfaceFormatKHR m_swapChainSurfaceFormat;
	std::vector<SwapchainData> m_swapchainData;
	vk::raii::PipelineLayout m_pipelineLayout = nullptr;
	vk::raii::Pipeline m_graphicsPipeline = nullptr;
	vk::raii::CommandPool m_commandPool = nullptr;
	vk::raii::CommandPool m_transferCommandPool = nullptr;
	vk::raii::DescriptorSetLayout m_descriptorSetLayout = nullptr;
	vk::raii::DescriptorPool m_descriptorPool = nullptr;
	std::vector<FrameData> m_frames;
	uint32_t m_frameIndex = 0;
	bool m_framebufferResized = false;
	bool m_swapChainInitialized = false;

	Registry& m_registry;
	Entity m_camera;
	std::vector<MeshResources> m_meshResources;
	std::unordered_map<Entity, size_t> m_entityToMesh;


	bool hasDedicatedTransferQueueFamily() const;
	vk::raii::Queue& transferQueue();
	vk::raii::CommandPool& transferCommandPool();
	FrameData& currentFrame();
	SwapchainData& currentSwapchainData(uint32_t imageIndex);

	void initVulkan();
	void cleanup();

	void createFrameResources();
	void updateUniformBuffers();

	void createInstance();
	void createSurface();
	void pickPhysicalDevice();
	void createLogicalDevice();
	bool isDeviceSuitable(vk::raii::PhysicalDevice const& physicalDevice);

	void createSwapChain();
	void createDescriptorSetLayout();
	void createSwapImageViews();
	void createGraphicsPipeline();
	void createCommandPool();
	void createDepthResources();
	void createDescriptorPool();
	void createUniformDescriptors(Entity entity);
	void createCommandBuffer();
	void createSyncObjects();
	void recordCommandBuffer(uint32_t imageIndex);
	void transitionImageLayout(
		vk::Image image,
		vk::ImageLayout oldLayout,
		vk::ImageLayout newLayout,
		vk::AccessFlags2 srcAccessMask,
		vk::AccessFlags2 dstAccessMask,
		vk::PipelineStageFlags2 srcStageMask,
		vk::PipelineStageFlags2 dstStageMask,
		vk::ImageAspectFlags imageAspectFlags
	);

	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
	uint32_t findQueueIndex(
		const std::vector<vk::QueueFamilyProperties>& queueFamilyProperties,
		vk::QueueFlagBits queueFlags,
		bool requireSurfaceSupport = false,
		vk::QueueFlags discouragedFlags = {}
	);

	std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(
		vk::DeviceSize size,
		vk::BufferUsageFlags usage,
		vk::MemoryPropertyFlags properties
	);
	std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createMeshBuffer(
		vk::DeviceSize bufferSize,
		const void* data,
		vk::BufferUsageFlagBits bufferType
	);
	void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size);
	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& availableFormats);
	vk::Format findSupportedFormat(std::span<const vk::Format> candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features);
	vk::Format findDepthFormat();
	vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const& availablePresentModes);
	vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const& capabilities);
	uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities);
	std::vector<char> readFile(const std::string& filename);

	void cleanupSwapChain();
	void recreateSwapChain();

	std::pair<vk::raii::Image, vk::raii::DeviceMemory> createImage(
		uint32_t width,
		uint32_t height,
		vk::Format format,
		vk::ImageTiling tiling,
		vk::ImageUsageFlags usage,
		vk::MemoryPropertyFlags properties
	);
	vk::raii::ImageView createImageView(const vk::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags);
	vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const;
	std::vector<const char*> getRequiredInstanceExtensions();
};
