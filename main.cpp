#include <memory>
#include <vector>
#include <cstdint>
#include <limits>
#include <fstream>
#include <iostream>
#include <map>
#include <array>
#include <chrono>

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

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "model.h"

constexpr uint32_t WIDTH  = 800;
constexpr uint32_t HEIGHT = 600;
const std::vector<char const*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex {
	glm::vec3 pos;
	glm::vec3 color;

	Vertex(glm::vec3 p, glm::vec3 c) : pos(p), color(c) {}

	static vk::VertexInputBindingDescription getBindingDescription() {
		return {
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = vk::VertexInputRate::eVertex
		};
	}

	static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
		return {{
			{.location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, pos)},
			{.location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, color)}
		}};
	}
};

struct UniformInfo {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

struct UniformBufferObject {
	glm::mat4 mvp;
};

const std::vector<Vertex> vertices = {
    {{-0.5f, 0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, 0.0f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.0f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.0f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};

const std::vector<uint16_t> indices = {
	0, 1, 2, 2, 3, 0
};

class HelloTriangleApplication
{
  public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

  private:
	GLFWwindow *window = nullptr;
	vk::raii::Context m_context;
	vk::raii::Instance m_instance = nullptr;
	vk::raii::PhysicalDevice m_physicalDevice = nullptr;
	vk::raii::Device m_device = nullptr;
	vk::raii::Queue m_graphicsQueue = nullptr;
	vk::raii::Queue m_transferQueue = nullptr;
	vk::raii::SurfaceKHR m_surface = nullptr;
	vk::raii::SwapchainKHR m_swapChain = nullptr;
	uint32_t m_graphicsQueueIndex = ~0;
	uint32_t m_transferQueueIndex = ~0;
	vk::Extent2D m_swapChainExtent;
	vk::SurfaceFormatKHR m_swapChainSurfaceFormat;
	std::vector<vk::Image> m_swapChainImages;
	std::vector<vk::raii::ImageView> m_swapChainImageViews;
	vk::raii::PipelineLayout m_pipelineLayout = nullptr;
	vk::raii::Pipeline m_graphicsPipeline = nullptr;
	vk::raii::CommandPool m_commandPool = nullptr;
	vk::raii::CommandPool m_transferCommandPool = nullptr;
	std::vector<vk::raii::CommandBuffer> m_commandBuffers;
	std::vector<vk::raii::Semaphore> m_presentCompleteSemaphores;
	std::vector<vk::raii::Semaphore> m_renderFinishedSemaphores;
	std::vector<vk::raii::Fence> m_inFlightFences;
	vk::raii::Buffer m_vertexBuffer = nullptr;
	vk::raii::DeviceMemory m_vertexBufferMemory = nullptr;
	vk::raii::Buffer m_indexBuffer = nullptr;
	vk::raii::DeviceMemory m_indexBufferMemory = nullptr;
	std::vector<vk::raii::Buffer> m_uniformBuffers;
	std::vector<vk::raii::DeviceMemory> m_uniformBuffersMemory;
	std::vector<void*> m_uniformBuffersMapped;
	uint32_t m_frameIndex = 0;
	bool m_framebufferResized = false;
	vk::raii::DescriptorSetLayout m_descriptorSetLayout = nullptr;
	vk::raii::DescriptorPool m_descriptorPool = nullptr;
	std::vector<vk::raii::DescriptorSet> m_descriptorSets;
	std::vector<Vertex> m_vertices;
	std::vector<uint16_t> m_indices;

	bool hasDedicatedTransferQueueFamily() const {
		return m_transferQueueIndex != m_graphicsQueueIndex;
	}

	vk::raii::Queue& transferQueue() {
		return hasDedicatedTransferQueueFamily() ? m_transferQueue : m_graphicsQueue;
	}

	vk::raii::CommandPool& transferCommandPool() {
		return hasDedicatedTransferQueueFamily() ? m_transferCommandPool : m_commandPool;
	}

	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		// the callback has to be static so it can be called from the callback
		// we set this as window
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
		app->m_framebufferResized = true;
	}

	void initVulkan()
	{
		createInstance();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		loadModel();
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffer();
		createSyncObjects();
	}

	void drawFrame() {
		auto fenceResult = m_device.waitForFences(*m_inFlightFences[m_frameIndex], VK_TRUE, std::numeric_limits<uint64_t>::max());
		if (fenceResult != vk::Result::eSuccess) {
			throw std::runtime_error("Failed to wait for fence");
		}

		auto [result, imageIndex] = m_swapChain.acquireNextImage(std::numeric_limits<uint64_t>::max(), *m_presentCompleteSemaphores[m_frameIndex], nullptr);

		if (result == vk::Result::eErrorOutOfDateKHR) {
			recreateSwapChain();
			m_framebufferResized = true;
			return;
		}

		if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
			assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		m_device.resetFences(*m_inFlightFences[m_frameIndex]);

		m_commandBuffers[m_frameIndex].reset();
		updateUniformBuffer(m_frameIndex);
		recordCommandBuffer(imageIndex);

		m_graphicsQueue.waitIdle();

		vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		const vk::SubmitInfo submitInfo {
			.waitSemaphoreCount = 1,
			.pWaitSemaphores    = &*m_presentCompleteSemaphores[m_frameIndex],
			.pWaitDstStageMask  = &waitDestinationStageMask,
			.commandBufferCount = 1,
			.pCommandBuffers    = &*m_commandBuffers[m_frameIndex],
			.signalSemaphoreCount = 1,
			.pSignalSemaphores   = &*m_renderFinishedSemaphores[imageIndex]
		};

		m_graphicsQueue.submit(submitInfo, *m_inFlightFences[m_frameIndex]);

		const vk::PresentInfoKHR presentInfoKHR {
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &*m_renderFinishedSemaphores[imageIndex],
			.swapchainCount      = 1,
			.pSwapchains         = &*m_swapChain,
			.pImageIndices       = &imageIndex
		};

		auto presentResult = m_graphicsQueue.presentKHR(presentInfoKHR);
		if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR || m_framebufferResized) {
			std::cout << "Suboptimal swap chain\n";
			recreateSwapChain();
		} else {
			// There are no other success codes than eSuccess; on any error code, presentKHR already threw an exception.
			assert(result == vk::Result::eSuccess);
		}
		m_frameIndex = (m_frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void updateUniformBuffer(uint32_t currentImage) {
		static auto startTime = std::chrono::high_resolution_clock::now();
		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, sin(time), 0.0f));
		model = glm::rotate(model, time * glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 proj = glm::perspective(glm::radians(45.0f), m_swapChainExtent.width / (float)m_swapChainExtent.height, 0.1f, 10.0f);
		proj[1][1] *= -1;

		UniformBufferObject ubo(proj * view * model);

		memcpy(m_uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
	}

	void createInstance() {
		constexpr vk::ApplicationInfo appInfo{
			.pApplicationName   = "Hello Triangle",
			.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
			.pEngineName        = "No Engine",
			.engineVersion      = VK_MAKE_VERSION(1, 0, 0),
			.apiVersion         = vk::ApiVersion14,
		};

		auto requiredExtensions = getRequiredInstanceExtensions();
		auto extensionProperties = m_context.enumerateInstanceExtensionProperties();

		std::cout << "Available extensions: \n";

		for (const auto& extensionProperty : extensionProperties) {
			std::cout << "\t" << extensionProperty.extensionName << "\n";
		}

		auto unsupportedPropertyIt = 
			std::ranges::find_if(requiredExtensions, [&extensionProperties](auto const& requiredExtension) {
				return std::ranges::none_of(extensionProperties, [requiredExtension](auto const& extensionProperty) {
					return strcmp(extensionProperty.extensionName, requiredExtension) == 0;
				});
			});

		if (unsupportedPropertyIt != requiredExtensions.end()) {
			throw std::runtime_error("Required extension not supported: " + std::string(*unsupportedPropertyIt));
		}

		

		std::vector<char const*> requiredLayers;
		if (enableValidationLayers) {
			requiredLayers.assign(validationLayers.begin(), validationLayers.end());
		}

		auto layerProperties = m_context.enumerateInstanceLayerProperties();
		auto unsupportedLayerIt = std::ranges::find_if(requiredLayers, 
			[&layerProperties](auto const &requiredLayer) {
				return std::ranges::none_of(layerProperties, [requiredLayer](auto const &layerProperty) {
					return strcmp(layerProperty.layerName, requiredLayer) == 0;
				});
			}
		);

		if (unsupportedLayerIt != requiredLayers.end()) {
			throw std::runtime_error("Required layer not supporeted: " + std::string(*unsupportedLayerIt));
		}

		vk::InstanceCreateInfo createInfo{
			.pApplicationInfo = &appInfo,
			.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
			.ppEnabledLayerNames = requiredLayers.data(),
			.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
			.ppEnabledExtensionNames = requiredExtensions.data()
		};

		m_instance = vk::raii::Instance(m_context, createInfo);
	}

	void createSurface() {
		VkSurfaceKHR windowSurface;

		if (glfwCreateWindowSurface(*m_instance, window, nullptr, &windowSurface) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create window surface");
		}
		m_surface = vk::raii::SurfaceKHR(m_instance, windowSurface);

	}

	void pickPhysicalDevice() {
		std::vector<vk::raii::PhysicalDevice> physicalDevices = m_instance.enumeratePhysicalDevices();
		auto const devIter = std::ranges::find_if(physicalDevices, [&](auto const& physicalDevice) { return isDeviceSuitable(physicalDevice); });
		if (devIter == physicalDevices.end()) {
			throw std::runtime_error("Failed to find GPUs with Vulkan support");
		}

		m_physicalDevice = *devIter;
	}

	void createLogicalDevice() {
		std::vector<vk::QueueFamilyProperties> queueFamilyProperties = m_physicalDevice.getQueueFamilyProperties();

		std::cout << queueFamilyProperties.size() << " queue families found\n";

		m_graphicsQueueIndex = findQueueIndex(queueFamilyProperties, vk::QueueFlagBits::eGraphics, true);
		m_transferQueueIndex = findQueueIndex(queueFamilyProperties, vk::QueueFlagBits::eTransfer, false, vk::QueueFlagBits::eGraphics);

		vk::StructureChain<vk::PhysicalDeviceFeatures2,vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> featChain = {
			{},
			{.shaderDrawParameters = true},
			{ .synchronization2 = true, .dynamicRendering = true },
			{.extendedDynamicState = true}
		};

		std::vector<const char*> requiredDeviceExtension = {vk::KHRSwapchainExtensionName};

		float queuePriority = 0.5f;
		uint32_t queueCreateInfoCount = 1;
		std::array<vk::DeviceQueueCreateInfo, 2> deviceQueueCreateInfos {{
			{
				.queueFamilyIndex = m_graphicsQueueIndex,
				.queueCount = 1,
				.pQueuePriorities = &queuePriority
			}
		}};
		if (hasDedicatedTransferQueueFamily()) {
			deviceQueueCreateInfos[1] = {
				.queueFamilyIndex = m_transferQueueIndex,
				.queueCount = 1,
				.pQueuePriorities = &queuePriority
			};
			queueCreateInfoCount = 2;
		}
		vk::DeviceCreateInfo deviceCreateInfo {
			.pNext = &featChain.get<vk::PhysicalDeviceFeatures2>(),
			.queueCreateInfoCount = queueCreateInfoCount,
			.pQueueCreateInfos = deviceQueueCreateInfos.data(),
			.enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
			.ppEnabledExtensionNames = requiredDeviceExtension.data()
		};

		m_device = vk::raii::Device(m_physicalDevice, deviceCreateInfo);
		m_graphicsQueue = vk::raii::Queue(m_device, m_graphicsQueueIndex, 0);
		m_transferQueue = vk::raii::Queue(m_device, m_transferQueueIndex, 0);
	}

	bool isDeviceSuitable(vk::raii::PhysicalDevice const& physicalDevice) {
		bool supportsVulkan1_3 = physicalDevice.getProperties().apiVersion >= VK_API_VERSION_1_3;

		auto queueFamilies = physicalDevice.getQueueFamilyProperties();
		bool supportsGraphics = std::ranges::any_of(queueFamilies, [](auto const& qfp) { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });

		std::vector<const char*> requiredDeviceExtension = {vk::KHRSwapchainExtensionName};

		auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
		bool supportsAllRequiredExtensions = std::ranges::all_of(requiredDeviceExtension, [&availableDeviceExtensions](auto const& requiredDeviceExtension) {
			return std::ranges::any_of(availableDeviceExtensions, [requiredDeviceExtension](auto const& availableDeviceExtension) {
				return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0;
			});
		});

		auto feature = physicalDevice.template getFeatures2<vk::PhysicalDeviceFeatures2,vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
		bool supportsAllRequiredFeatures = 
			feature.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
			feature.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
			feature.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
			feature.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

		return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsAllRequiredFeatures;
	}

	void createSwapChain() {
		vk::SurfaceCapabilitiesKHR surfaceCapabilities = m_physicalDevice.getSurfaceCapabilitiesKHR(*m_surface);
		m_swapChainExtent = chooseSwapExtent(surfaceCapabilities);
		uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

		std::vector<vk::SurfaceFormatKHR> availableSurfaceFormats = m_physicalDevice.getSurfaceFormatsKHR(*m_surface);
		m_swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableSurfaceFormats);

		std::vector<vk::PresentModeKHR> presentModes = m_physicalDevice.getSurfacePresentModesKHR(*m_surface);
		vk::PresentModeKHR swapPresentMode = chooseSwapPresentMode(presentModes);

		vk::SwapchainCreateInfoKHR swapChainCreateInfo {
			.surface = *m_surface,
			.minImageCount = minImageCount,
			.imageFormat = m_swapChainSurfaceFormat.format,
			.imageColorSpace = m_swapChainSurfaceFormat.colorSpace,
			.imageExtent = m_swapChainExtent,
			.imageArrayLayers = 1,
			.imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
			.imageSharingMode = vk::SharingMode::eExclusive,
			.preTransform = surfaceCapabilities.currentTransform,
			.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
			.presentMode = swapPresentMode,
			.clipped = true,
			.oldSwapchain = nullptr
		};

		m_swapChain = vk::raii::SwapchainKHR(m_device, swapChainCreateInfo);
		m_swapChainImages = m_swapChain.getImages();
	}

	void createDescriptorSetLayout() {
		vk::DescriptorSetLayoutBinding uboLayoutBinding {
			.binding = 0,
			.descriptorType = vk::DescriptorType::eUniformBuffer,
			.descriptorCount = 1,
			.stageFlags = vk::ShaderStageFlagBits::eVertex
		};

		vk::DescriptorSetLayoutCreateInfo layoutInfo {
			.bindingCount = 1,
			.pBindings = &uboLayoutBinding,
		};
		m_descriptorSetLayout = m_device.createDescriptorSetLayout(layoutInfo);
	}

	void createImageViews() {
		assert(m_swapChainImageViews.empty());

		vk::ImageViewCreateInfo imageViewCreateInfo {
			.viewType = vk::ImageViewType::e2D,
			.format = m_swapChainSurfaceFormat.format,
			.subresourceRange = {
				vk::ImageAspectFlagBits::eColor,
				0,
				1,
				0,
				1
			}
		};

		for (auto& image : m_swapChainImages) {
			imageViewCreateInfo.image = image;
			m_swapChainImageViews.emplace_back(m_device, imageViewCreateInfo);
		}
	}

	void createGraphicsPipeline() {
		std::vector<char> shaderCode = readFile("shaders/slang.spv");
		vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo {
			.stage = vk::ShaderStageFlagBits::eVertex,
			.module = shaderModule,
			.pName = "vertMain"
		};

		vk::PipelineShaderStageCreateInfo fragShaderStageInfo {
			.stage = vk::ShaderStageFlagBits::eFragment,
			.module = shaderModule,
			.pName = "fragMain"
		};

		vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo {
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &bindingDescription,
			.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
			.pVertexAttributeDescriptions = attributeDescriptions.data(),
		};
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly {.topology = vk::PrimitiveTopology::eTriangleList};
		vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

		vk::PipelineRasterizationStateCreateInfo rasterizer {
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eBack,
			.frontFace = vk::FrontFace::eClockwise,
			.depthBiasEnable = vk::False,
			.lineWidth = 1.0f
		};

		vk::PipelineMultisampleStateCreateInfo multisampling {
			.rasterizationSamples = vk::SampleCountFlagBits::e1,
			.sampleShadingEnable = vk::False
		};

		vk::PipelineColorBlendAttachmentState colorBlendAttachment {
			.blendEnable = vk::False,
			.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
		};

		vk::PipelineColorBlendStateCreateInfo colorBlending {
			.logicOpEnable = vk::False,
			.logicOp = vk::LogicOp::eCopy,
			.attachmentCount = 1,
			.pAttachments = &colorBlendAttachment
		};

		std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dynamicState {
			.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
			.pDynamicStates = dynamicStates.data()
		};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo {
			.setLayoutCount = 1,
			.pSetLayouts = &*m_descriptorSetLayout,
			.pushConstantRangeCount = 0
		};
		m_pipelineLayout = vk::raii::PipelineLayout(m_device, pipelineLayoutInfo);

		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfo {
			{
				.stageCount = 2,
				.pStages = shaderStages,
				.pVertexInputState = &vertexInputInfo,
				.pInputAssemblyState = &inputAssembly,
				.pViewportState = &viewportState,
				.pRasterizationState = &rasterizer,
				.pMultisampleState = &multisampling,
				.pColorBlendState = &colorBlending,
				.pDynamicState = &dynamicState,
				.layout = m_pipelineLayout,
				.renderPass = nullptr
			},
			{
				.colorAttachmentCount = 1,
				.pColorAttachmentFormats = &m_swapChainSurfaceFormat.format
			}
		};

		m_graphicsPipeline = vk::raii::Pipeline(m_device, nullptr, pipelineCreateInfo.get<vk::GraphicsPipelineCreateInfo>());

	}

	void createCommandPool() {
		vk::CommandPoolCreateInfo poolInfo{
			.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			.queueFamilyIndex = m_graphicsQueueIndex
		};
		m_commandPool = vk::raii::CommandPool(m_device, poolInfo);

		vk::CommandPoolCreateInfo transferPoolInfo{
			.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			.queueFamilyIndex = m_transferQueueIndex

		};

		if (hasDedicatedTransferQueueFamily()) {
			m_transferCommandPool = vk::raii::CommandPool(m_device, transferPoolInfo);
		}
	}

	void createVertexBuffer() {
		vk::DeviceSize bufferSize = sizeof(m_vertices[0]) * m_vertices.size();

		auto [stagingBuffer, stagingBufferMemory] = 
			createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		
		void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(dataStaging, m_vertices.data(), bufferSize);
		stagingBufferMemory.unmapMemory();

		std::tie(m_vertexBuffer, m_vertexBufferMemory) = 
			createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);

		copyBuffer(stagingBuffer, m_vertexBuffer, bufferSize);

	}

	void createIndexBuffer() {
		vk::DeviceSize bufferSize = sizeof(m_indices[0]) * m_indices.size();

		auto [stagingBuffer, stagingBufferMemory] =
			createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		
		void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(dataStaging, m_indices.data(), bufferSize);
		stagingBufferMemory.unmapMemory();

		std::tie(m_indexBuffer, m_indexBufferMemory) = createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
		copyBuffer(stagingBuffer, m_indexBuffer, bufferSize);
	}

	void createUniformBuffers() {
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
			auto [buffer, bufferMem] = 
				createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
			m_uniformBuffers.emplace_back(std::move(buffer));
			m_uniformBuffersMemory.emplace_back(std::move(bufferMem));
			m_uniformBuffersMapped.emplace_back(m_uniformBuffersMemory.back().mapMemory(0, bufferSize));
		}
	}

	void createDescriptorPool() {
		vk::DescriptorPoolSize poolSize {
			.type = vk::DescriptorType::eUniformBuffer,
			.descriptorCount = MAX_FRAMES_IN_FLIGHT
		};

		vk::DescriptorPoolCreateInfo poolInfo {
			.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
			.maxSets = MAX_FRAMES_IN_FLIGHT,
			.poolSizeCount = 1,
			.pPoolSizes = &poolSize
		};

		m_descriptorPool = vk::raii::DescriptorPool(m_device, poolInfo);
	}

	void createDescriptorSets() {
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *m_descriptorSetLayout);
		vk::DescriptorSetAllocateInfo allocInfo {
			.descriptorPool = m_descriptorPool,
			.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
			.pSetLayouts = layouts.data()
		};
		m_descriptorSets = m_device.allocateDescriptorSets(allocInfo);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vk::DescriptorBufferInfo bufferInfo {
				.buffer = m_uniformBuffers[i],
				.offset = 0,
				.range = sizeof(UniformBufferObject)
			};
			vk::WriteDescriptorSet descriptorWrite {
				.dstSet = m_descriptorSets[i],
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBuffer,
				.pBufferInfo = &bufferInfo,
			};
			m_device.updateDescriptorSets(descriptorWrite, {});
		}
	}

	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
		vk::PhysicalDeviceMemoryProperties memProperties = m_physicalDevice.getMemoryProperties();

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	uint32_t findQueueIndex(
		const std::vector<vk::QueueFamilyProperties>& queueFamilyProperties,
		vk::QueueFlagBits queueFlags,
		bool requireSurfaceSupport = false,
		vk::QueueFlags discouragedFlags = {}
	) {
		uint32_t fallbackIndex = UINT_MAX;

		for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++) {
			vk::QueueFlags flags = queueFamilyProperties[qfpIndex].queueFlags;
			if (!(flags & queueFlags)) {
				continue;
			}
			if (requireSurfaceSupport && !m_physicalDevice.getSurfaceSupportKHR(qfpIndex, *m_surface)) {
				continue;
			}
			if (fallbackIndex == UINT_MAX) {
				fallbackIndex = qfpIndex;
			}
			if (!(flags & discouragedFlags)) {
				return qfpIndex;
			}
		}

		if (fallbackIndex != UINT_MAX) {
			return fallbackIndex;
		}

		throw std::runtime_error("Failed to find a suitable Queue Family for " + vk::to_string(queueFlags));
	}

	void createCommandBuffer() {
		vk::CommandBufferAllocateInfo allocInfo {
			.commandPool = m_commandPool,
			.level = vk::CommandBufferLevel::ePrimary,
			.commandBufferCount = MAX_FRAMES_IN_FLIGHT
		};

		m_commandBuffers = vk::raii::CommandBuffers(m_device, allocInfo);
	}

	void createSyncObjects() {
		for (size_t i = 0; i < m_swapChainImages.size(); i++) {
			m_renderFinishedSemaphores.emplace_back(m_device, vk::SemaphoreCreateInfo());
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			m_presentCompleteSemaphores.emplace_back(m_device, vk::SemaphoreCreateInfo());
			m_inFlightFences.emplace_back(m_device, vk::FenceCreateInfo{
				.flags = vk::FenceCreateFlagBits::eSignaled
			});
		}
	}

	void recordCommandBuffer(uint32_t index) {
		auto& commandBuffer = m_commandBuffers[m_frameIndex];
		commandBuffer.begin({});

		transition_image_layout(
			index,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eColorAttachmentOptimal,
			{},
			vk::AccessFlagBits2::eColorAttachmentWrite,
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			vk::PipelineStageFlagBits2::eColorAttachmentOutput
		);

		vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
		vk::RenderingAttachmentInfo attachmentInfo = {
			.imageView = m_swapChainImageViews[index],
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eStore,
			.clearValue = clearColor
		};

		vk::RenderingInfo renderingInfo = {
			.renderArea = {.offset = {0, 0}, .extent = m_swapChainExtent },
			.layerCount = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments = &attachmentInfo
		};

		commandBuffer.beginRendering(renderingInfo);
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_graphicsPipeline);

		commandBuffer.bindVertexBuffers(0, *m_vertexBuffer, {0});
		commandBuffer.bindIndexBuffer(*m_indexBuffer, 0, vk::IndexType::eUint16);

		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, *m_descriptorSets[m_frameIndex], nullptr);

		commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(m_swapChainExtent.width), static_cast<float>(m_swapChainExtent.height), 0.0f, 1.0f));
		commandBuffer.setScissor(0, vk::Rect2D({0, 0}, m_swapChainExtent));

		commandBuffer.drawIndexed(static_cast<uint32_t>(m_indices.size()), 1, 0, 0, 0); // HOLY SHIIIT;

		commandBuffer.endRendering();

		transition_image_layout(
			index,
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::ePresentSrcKHR,
			vk::AccessFlagBits2::eColorAttachmentWrite,
			{},
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			vk::PipelineStageFlagBits2::eBottomOfPipe
		);
		
		commandBuffer.end();
	}

	void transition_image_layout(
		uint32_t imageIndex,
		vk::ImageLayout old_layout,
		vk::ImageLayout new_layout,
		vk::AccessFlags2 src_access_mask,
		vk::AccessFlags2 dst_access_mask,
		vk::PipelineStageFlags2 src_stage_mask,
		vk::PipelineStageFlags2 dst_stage_mask
	) {
		vk::ImageMemoryBarrier2 barrier = {
			.srcStageMask = src_stage_mask,
			.srcAccessMask = src_access_mask,
			.dstStageMask = dst_stage_mask,
			.dstAccessMask = dst_access_mask,
			.oldLayout = old_layout,
			.newLayout = new_layout,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = m_swapChainImages[imageIndex],
			.subresourceRange = {
				vk::ImageAspectFlagBits::eColor,
				0,
				1,
				0,
				1
			}
		};
		vk::DependencyInfo dependency_info {
			.dependencyFlags = {},
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &barrier
		};
		m_commandBuffers[m_frameIndex].pipelineBarrier2(dependency_info);
	}

	std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties) {
		vk::BufferCreateInfo bufferInfo {
			.size = size,
			.usage = usage,
			.sharingMode = vk::SharingMode::eExclusive
		};
		std::array<uint32_t, 2> queueFamilies = {m_graphicsQueueIndex, m_transferQueueIndex};
		if (hasDedicatedTransferQueueFamily()) {
			bufferInfo.sharingMode = vk::SharingMode::eConcurrent;
			bufferInfo.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilies.size());
			bufferInfo.pQueueFamilyIndices = queueFamilies.data();
		}

		vk::raii::Buffer buffer = vk::raii::Buffer(m_device, bufferInfo);
		vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
		vk::MemoryAllocateInfo memoryAllocateInfo {
			.allocationSize = memRequirements.size,
			.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
		};
		vk::raii::DeviceMemory bufferMemory = vk::raii::DeviceMemory(m_device, memoryAllocateInfo);
		buffer.bindMemory(bufferMemory, 0);
		return {std::move(buffer), std::move(bufferMemory)};
	}

	void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size) {
		vk::CommandBufferAllocateInfo allocInfo {
			.commandPool = transferCommandPool(),
			.level = vk::CommandBufferLevel::ePrimary,
			.commandBufferCount = 1
		};
		vk::raii::CommandBuffer commandCopyBuffer = std::move(m_device.allocateCommandBuffers(allocInfo).front());
		commandCopyBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy(0, 0, size));
		commandCopyBuffer.end();
		transferQueue().submit(vk::SubmitInfo {.commandBufferCount = 1, .pCommandBuffers = &*commandCopyBuffer}, nullptr);
		transferQueue().waitIdle();
	}

	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& availableFormats) {
		assert(!availableFormats.empty());
		const auto formatIt = std::ranges::find_if(
			availableFormats,
			[](const auto& format){
				return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
			}
		);
		return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
	}


	vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const& availablePresentModes) {
		assert(std::ranges::any_of(availablePresentModes, [](const auto& mode){ return mode == vk::PresentModeKHR::eFifo; }));
		return std::ranges::any_of(availablePresentModes, [](const auto& mode) { return mode == vk::PresentModeKHR::eMailbox; }) ?
			vk::PresentModeKHR::eMailbox :
			vk::PresentModeKHR::eFifo;
	}

	vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const& capabilities) {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		}
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		return {
			std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
			std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
		};
	}

	uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities) {
		uint32_t minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
		if (surfaceCapabilities.maxImageCount > 0 && surfaceCapabilities.maxImageCount < minImageCount) {
			minImageCount = surfaceCapabilities.maxImageCount;
		}
		return minImageCount;
	}

	std::vector<char> readFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error("Failed to open file!");
		}
		std::vector<char> buffer(file.tellg());
		file.seekg(0, std::ios::beg);
		file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
		file.close();
		return buffer;
	}

	void cleanupSwapChain() {
		m_swapChainImageViews.clear();
		m_swapChain = nullptr;
	}

	void recreateSwapChain() {
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		m_device.waitIdle();

		cleanupSwapChain();

		createSwapChain();
		createImageViews();
	}

	vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
		vk::ShaderModuleCreateInfo createInfo {
			.codeSize = code.size() * sizeof(char),
			.pCode = reinterpret_cast<const uint32_t*>(code.data())
		};
		vk::raii::ShaderModule shaderModule(m_device, createInfo);
		return shaderModule;
	}

	std::vector<const char*> getRequiredInstanceExtensions() {
		uint32_t glfwExtensionCount = 0;
		auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		return extensions;
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			drawFrame();
		}
		m_device.waitIdle();        // wait for device to finish operations before destroying resources
	}

	void loadModel() {
		Model model("box.obj");
		for (size_t i = 0; i < model.nverts(); i++) {
			m_vertices.emplace_back(model.vert(i), glm::vec3{1.0f, 1.0f, 1.0f});
		}
		m_indices = model.getIndices();
	}

	void cleanup()
	{
		cleanupSwapChain();

		glfwDestroyWindow(window);
		glfwTerminate();
	}
};

int main()
{
	try
	{
		HelloTriangleApplication app;
		app.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
