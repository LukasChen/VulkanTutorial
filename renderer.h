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

#include "model.h"
#include "vertex_layout.h"
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

inline const uint32_t MAX_ENTITY_COUNT = 4096;

struct InstanceData {
	glm::mat4 model;
	glm::mat3 normal;

	static vk::VertexInputBindingDescription getBindingDescription() {
		return {
			.binding = 1,
			.stride = sizeof(InstanceData),
			.inputRate = vk::VertexInputRate::eInstance
		};
	}

	static std::array<vk::VertexInputAttributeDescription, 7> getAttributeDescriptions() {
		return {{
			{.location = 4, .binding = 1, .format = vk::Format::eR32G32B32A32Sfloat, .offset = offsetof(InstanceData, model) + sizeof(glm::vec4) * 0},
			{.location = 5, .binding = 1, .format = vk::Format::eR32G32B32A32Sfloat, .offset = offsetof(InstanceData, model) + sizeof(glm::vec4) * 1},
			{.location = 6, .binding = 1, .format = vk::Format::eR32G32B32A32Sfloat, .offset = offsetof(InstanceData, model) + sizeof(glm::vec4) * 2},
			{.location = 7, .binding = 1, .format = vk::Format::eR32G32B32A32Sfloat, .offset = offsetof(InstanceData, model) + sizeof(glm::vec4) * 3},
			{.location = 8, .binding = 1, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(InstanceData, normal) + sizeof(glm::vec3) * 0},
			{.location = 9, .binding = 1, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(InstanceData, normal) + sizeof(glm::vec3) * 1},
			{.location = 10, .binding = 1, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(InstanceData, normal) + sizeof(glm::vec3) * 2}
		}};
	}
};

struct FrameUniformBufferObject {
	glm::mat4 vp;
};

struct InstanceBatch {
	size_t meshHandle;
	uint32_t firstInstance;
	uint32_t instanceCount;
};

struct FrameResources {
	vk::raii::Buffer uniformBuffer = nullptr;
	vk::raii::DeviceMemory uniformBufferMemory = nullptr;
	void* uniformBufferMapped = nullptr;
	vk::raii::DescriptorSet descriptorSet = nullptr;
	vk::raii::Buffer instanceBuffer = nullptr;
	vk::raii::DeviceMemory instanceBufferMemory = nullptr;
	void* instanceBufferMapped = nullptr;
	size_t instanceCapacity = 0;
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
	FrameResources resources;
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

	void createMeshEntity(Entity entity);
	void rebuildInstanceBatches();
	size_t uploadMesh(const Model& meshData);
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
	std::vector<InstanceBatch> m_instanceBatches;
	std::unordered_map<size_t, uint32_t> m_meshToBatchIndex;
	size_t m_instanceCount = 0;

	vk::raii::Image m_textureImage = nullptr;
	vk::raii::DeviceMemory m_textureImageMemory = nullptr;

	vk::raii::ImageView m_textureImageView = nullptr;
	vk::raii::Sampler m_textureSampler = nullptr;


	bool hasDedicatedTransferQueueFamily() const;
	vk::raii::Queue& transferQueue();
	vk::raii::CommandPool& transferCommandPool();
	FrameData& currentFrame();
	SwapchainData& currentSwapchainData(uint32_t imageIndex);

	void initVulkan();
	void cleanup();

	void createFrameResources();
	void updateFrameResources();

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
	void createTextureImage();
	void createTextureImageView();
	void createTextureSampler();
	void createDepthResources();
	void createDescriptorPool();
	void createFrameDescriptors();
	void createCommandBuffer();
	void createSyncObjects();
	void recordCommandBuffer(uint32_t imageIndex);
	void transition_image_layout(
		vk::Image image,
		vk::ImageLayout oldLayout,
		vk::ImageLayout newLayout,
		vk::AccessFlags2 srcAccessMask,
		vk::AccessFlags2 dstAccessMask,
		vk::PipelineStageFlags2 srcStageMask,
		vk::PipelineStageFlags2 dstStageMask,
		vk::ImageAspectFlags imageAspectFlags
	);
	void transitionImageLayout(vk::raii::CommandBuffer& commandBuffer, const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);

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
	void copyBufferToImage(vk::raii::CommandBuffer& commandBuffer, const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height);
	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& availableFormats);
	vk::Format findSupportedFormat(std::span<const vk::Format> candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features);
	vk::Format findDepthFormat();
	vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const& availablePresentModes);
	vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const& capabilities);
	uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities);
	vk::raii::CommandBuffer beginSingleTimeCommands();
	void endSingleTimeCommands(vk::raii::CommandBuffer&& commandBuffer);
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
