#include "engine.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "../components/components_common.h"

Engine::Engine(Registry& registry)
	: m_registry(registry) {
	initWindow();
	initVulkan();
}

Engine::~Engine() {
	cleanup();
}

void Engine::createMeshEntity(Entity entity, std::pair<std::vector<Vertex>, std::vector<uint16_t>>& meshData) {
	vk::DeviceSize vertexBufferSize = meshData.first.size() * sizeof(Vertex);
	auto [vertexBuffer, vertexBufferMemory] = createMeshBuffer(vertexBufferSize, meshData.first.data(), vk::BufferUsageFlagBits::eVertexBuffer);

	vk::DeviceSize indexBufferSize = meshData.second.size() * sizeof(uint16_t);
	auto [indexBuffer, indexBufferMemory] = createMeshBuffer(indexBufferSize, meshData.second.data(), vk::BufferUsageFlagBits::eIndexBuffer);

	m_meshResources.emplace(entity, MeshResources{
		std::move(vertexBuffer),
		std::move(vertexBufferMemory),
		std::move(indexBuffer),
		std::move(indexBufferMemory),
		static_cast<uint16_t>(meshData.second.size())
	});

	createUniformDescriptors(entity);
}

void Engine::createUniformDescriptors(Entity entity) {
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *m_descriptorSetLayout);
	vk::DescriptorSetAllocateInfo allocInfo{
		.descriptorPool = m_descriptorPool,
		.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
		.pSetLayouts = layouts.data()
	};
	auto descriptorSets = m_device.allocateDescriptorSets(allocInfo);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
		auto [buffer, bufferMem] = createBuffer(
			bufferSize,
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
		);
		auto& frame = m_frames[i];
		frame.entityResources[entity].uniformBuffer = std::move(buffer);
		frame.entityResources[entity].uniformBufferMemory = std::move(bufferMem);
		frame.entityResources[entity].uniformBufferMapped = frame.entityResources[entity].uniformBufferMemory.mapMemory(0, bufferSize);

		frame.entityResources[entity].descriptorSet = std::move(descriptorSets[i]);
		vk::DescriptorBufferInfo bufferInfo{
			.buffer = frame.entityResources[entity].uniformBuffer,
			.offset = 0,
			.range = sizeof(UniformBufferObject)
		};
		vk::WriteDescriptorSet descriptorWrite{
			.dstSet = frame.entityResources[entity].descriptorSet,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = vk::DescriptorType::eUniformBuffer,
			.pBufferInfo = &bufferInfo,
		};
		m_device.updateDescriptorSets(descriptorWrite, {});
	}
}

void Engine::updateUniformBuffers(uint32_t currentFrameIndex) {
	auto view = m_registry.view<Transform, Mesh>();
	for (auto it = view.begin(); it != view.end(); ++it) {
		const Entity entity = it.entity();
		auto [transform, mesh] = *it;
		auto& enttResource = currentFrame().entityResources[entity];

		glm::mat4 model = glm::translate(glm::mat4(1.0f), transform.position);
		model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		// model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		// model = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)) * model;
		glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 proj = glm::perspective(glm::radians(45.0f), m_swapChainExtent.width / static_cast<float>(m_swapChainExtent.height), 0.1f, 10.0f);
		proj[1][1] *= -1;

		glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(model)));
		UniformBufferObject ubo{model, proj * view, normalMatrix};
		memcpy(enttResource.uniformBufferMapped, &ubo, sizeof(ubo));
	}
}

void Engine::run() {
	mainLoop();
}

bool Engine::hasDedicatedTransferQueueFamily() const {
	return m_transferQueueIndex != m_graphicsQueueIndex;
}

vk::raii::Queue& Engine::transferQueue() {
	return hasDedicatedTransferQueueFamily() ? m_transferQueue : m_graphicsQueue;
}

vk::raii::CommandPool& Engine::transferCommandPool() {
	return hasDedicatedTransferQueueFamily() ? m_transferCommandPool : m_commandPool;
}

FrameData& Engine::currentFrame() {
	return m_frames[m_frameIndex];
}

SwapchainData& Engine::currentSwapchainData(uint32_t imageIndex) {
	return m_swapchainData[imageIndex];
}

void Engine::initWindow() {
	glfwInit();
	m_glfwInitialized = true;

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	m_window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	glfwSetWindowUserPointer(m_window, this);
	glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);
}

void Engine::framebufferResizeCallback(GLFWwindow* window, int, int) {
	auto* app = reinterpret_cast<Engine*>(glfwGetWindowUserPointer(window));
	app->m_framebufferResized = true;
}

void Engine::initVulkan() {
	createInstance();
	createSurface();
	pickPhysicalDevice();
	createLogicalDevice();
	createSwapChain();
	createDescriptorSetLayout();
	createGraphicsPipeline();
	createCommandPool();
	createFrameResources();
	createDescriptorPool();
	createCommandBuffer();
	createSyncObjects();
}

void Engine::mainLoop() {
	while (!glfwWindowShouldClose(m_window)) {
		glfwPollEvents();
		drawFrame();
	}

	if (m_device != nullptr) {
		m_device.waitIdle();
	}
}

void Engine::drawFrame() {
	auto& frame = currentFrame();
	auto fenceResult = m_device.waitForFences(*frame.inFlightFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
	if (fenceResult != vk::Result::eSuccess) {
		throw std::runtime_error("Failed to wait for fence");
	}

	auto [result, imageIndex] = m_swapChain.acquireNextImage(
		std::numeric_limits<uint64_t>::max(),
		*frame.presentCompleteSemaphore,
		nullptr
	);

	if (result == vk::Result::eErrorOutOfDateKHR) {
		recreateSwapChain();
		m_framebufferResized = true;
		return;
	}

	if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
		assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
		throw std::runtime_error("failed to acquire swap chain image!");
	}

	auto& swapchainData = currentSwapchainData(imageIndex);

	m_device.resetFences(*frame.inFlightFence);
	frame.commandBuffer.reset();
	// updateUniformBuffer(m_frameIndex);
	updateUniformBuffers(m_frameIndex);
	recordCommandBuffer(imageIndex);

	vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
	const vk::SubmitInfo submitInfo{
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &*frame.presentCompleteSemaphore,
		.pWaitDstStageMask = &waitDestinationStageMask,
		.commandBufferCount = 1,
		.pCommandBuffers = &*frame.commandBuffer,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &*swapchainData.renderFinishedSemaphore
	};

	m_graphicsQueue.submit(submitInfo, *frame.inFlightFence);

	const vk::PresentInfoKHR presentInfoKHR{
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &*swapchainData.renderFinishedSemaphore,
		.swapchainCount = 1,
		.pSwapchains = &*m_swapChain,
		.pImageIndices = &imageIndex
	};

	auto presentResult = m_graphicsQueue.presentKHR(presentInfoKHR);
	if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR || m_framebufferResized) {
		recreateSwapChain();
	} else {
		assert(result == vk::Result::eSuccess);
	}

	m_frameIndex = (m_frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Engine::createFrameResources() {
	m_frames.resize(MAX_FRAMES_IN_FLIGHT);
}

void Engine::updateUniformBuffer(uint32_t currentFrameIndex) {
	static auto startTime = std::chrono::high_resolution_clock::now();
	auto currentTime = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

	glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
	model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	model = glm::rotate(model, time * glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)) * model;
	glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 proj = glm::perspective(glm::radians(45.0f), m_swapChainExtent.width / static_cast<float>(m_swapChainExtent.height), 0.1f, 10.0f);
	proj[1][1] *= -1;

	glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(model)));
	UniformBufferObject ubo{model, proj * view, normalMatrix};
	memcpy(m_frames[currentFrameIndex].uniformBufferMapped, &ubo, sizeof(ubo));
}

void Engine::createInstance() {
	constexpr vk::ApplicationInfo appInfo{
		.pApplicationName = "Hello Triangle",
		.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
		.pEngineName = "No Engine",
		.engineVersion = VK_MAKE_VERSION(1, 0, 0),
		.apiVersion = vk::ApiVersion14,
	};

	auto requiredExtensions = getRequiredInstanceExtensions();
	auto extensionProperties = m_context.enumerateInstanceExtensionProperties();

	for (const auto& requiredExtension : requiredExtensions) {
		auto it = std::ranges::find_if(extensionProperties, [requiredExtension](const auto& extensionProperty) {
			return strcmp(extensionProperty.extensionName, requiredExtension) == 0;
		});
		if (it == extensionProperties.end()) {
			throw std::runtime_error("Required extension not supported: " + std::string(requiredExtension));
		}
	}

	std::vector<char const*> requiredLayers;
	if (enableValidationLayers) {
		requiredLayers.assign(validationLayers.begin(), validationLayers.end());
	}

	auto layerProperties = m_context.enumerateInstanceLayerProperties();
	for (const auto& requiredLayer : requiredLayers) {
		auto it = std::ranges::find_if(layerProperties, [requiredLayer](const auto& layerProperty) {
			return strcmp(layerProperty.layerName, requiredLayer) == 0;
		});
		if (it == layerProperties.end()) {
			throw std::runtime_error("Required layer not supported: " + std::string(requiredLayer));
		}
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

void Engine::createSurface() {
	VkSurfaceKHR windowSurface;
	if (glfwCreateWindowSurface(*m_instance, m_window, nullptr, &windowSurface) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create window surface");
	}
	m_surface = vk::raii::SurfaceKHR(m_instance, windowSurface);
}

void Engine::pickPhysicalDevice() {
	std::vector<vk::raii::PhysicalDevice> physicalDevices = m_instance.enumeratePhysicalDevices();
	auto devIter = std::ranges::find_if(physicalDevices, [&](auto const& physicalDevice) {
		return isDeviceSuitable(physicalDevice);
	});
	if (devIter == physicalDevices.end()) {
		throw std::runtime_error("Failed to find GPUs with Vulkan support");
	}

	m_physicalDevice = *devIter;
}

void Engine::createLogicalDevice() {
	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = m_physicalDevice.getQueueFamilyProperties();

	m_graphicsQueueIndex = findQueueIndex(queueFamilyProperties, vk::QueueFlagBits::eGraphics, true);
	m_transferQueueIndex = findQueueIndex(queueFamilyProperties, vk::QueueFlagBits::eTransfer, false, vk::QueueFlagBits::eGraphics);

	vk::StructureChain<
		vk::PhysicalDeviceFeatures2,
		vk::PhysicalDeviceVulkan11Features,
		vk::PhysicalDeviceVulkan13Features,
		vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
	> featChain = {
		{},
		{.shaderDrawParameters = true},
		{.synchronization2 = true, .dynamicRendering = true},
		{.extendedDynamicState = true}
	};

	std::vector<const char*> requiredDeviceExtension = {vk::KHRSwapchainExtensionName};

	float queuePriority = 0.5f;
	uint32_t queueCreateInfoCount = 1;
	std::array<vk::DeviceQueueCreateInfo, 2> deviceQueueCreateInfos{{
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

	vk::DeviceCreateInfo deviceCreateInfo{
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

bool Engine::isDeviceSuitable(vk::raii::PhysicalDevice const& physicalDevice) {
	bool supportsVulkan13 = physicalDevice.getProperties().apiVersion >= VK_API_VERSION_1_3;
	auto queueFamilies = physicalDevice.getQueueFamilyProperties();
	bool supportsGraphics = std::ranges::any_of(queueFamilies, [](auto const& qfp) {
		return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
	});

	std::vector<const char*> requiredDeviceExtension = {vk::KHRSwapchainExtensionName};
	auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
	bool supportsAllRequiredExtensions = std::ranges::all_of(requiredDeviceExtension, [&availableDeviceExtensions](auto const& requiredDeviceExtensionName) {
		return std::ranges::any_of(availableDeviceExtensions, [requiredDeviceExtensionName](auto const& availableDeviceExtension) {
			return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtensionName) == 0;
		});
	});

	auto feature = physicalDevice.template getFeatures2<
		vk::PhysicalDeviceFeatures2,
		vk::PhysicalDeviceVulkan11Features,
		vk::PhysicalDeviceVulkan13Features,
		vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
	>();
	bool supportsAllRequiredFeatures =
		feature.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
		feature.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
		feature.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
		feature.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

	return supportsVulkan13 && supportsGraphics && supportsAllRequiredExtensions && supportsAllRequiredFeatures;
}

void Engine::createSwapChain() {
	vk::SurfaceCapabilitiesKHR surfaceCapabilities = m_physicalDevice.getSurfaceCapabilitiesKHR(*m_surface);
	m_swapChainExtent = chooseSwapExtent(surfaceCapabilities);
	uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

	std::vector<vk::SurfaceFormatKHR> availableSurfaceFormats = m_physicalDevice.getSurfaceFormatsKHR(*m_surface);
	m_swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableSurfaceFormats);

	std::vector<vk::PresentModeKHR> presentModes = m_physicalDevice.getSurfacePresentModesKHR(*m_surface);
	vk::PresentModeKHR swapPresentMode = chooseSwapPresentMode(presentModes);

	vk::SwapchainCreateInfoKHR swapChainCreateInfo{
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
		.clipped = true
	};

	m_swapChain = vk::raii::SwapchainKHR(m_device, swapChainCreateInfo);
	auto swapChainImages = m_swapChain.getImages();
	if (!m_swapChainInitialized) {
		m_swapchainData.resize(swapChainImages.size());
		m_swapChainInitialized = true;
	} else {
		for (auto& swapchainData : m_swapchainData) {
			swapchainData.image = nullptr;
			swapchainData.imageView = nullptr;
			swapchainData.depthImage = nullptr;
			swapchainData.depthImageMemory = nullptr;
			swapchainData.depthImageView = nullptr;
		}
	}

	for (size_t i = 0; i < swapChainImages.size(); i++) {
		m_swapchainData[i].image = swapChainImages[i];
	}

	createSwapImageViews();
	createDepthResources();
}

void Engine::createDescriptorSetLayout() {
	vk::DescriptorSetLayoutBinding uboLayoutBinding{
		.binding = 0,
		.descriptorType = vk::DescriptorType::eUniformBuffer,
		.descriptorCount = 1,
		.stageFlags = vk::ShaderStageFlagBits::eVertex
	};

	vk::DescriptorSetLayoutCreateInfo layoutInfo{
		.bindingCount = 1,
		.pBindings = &uboLayoutBinding,
	};
	m_descriptorSetLayout = m_device.createDescriptorSetLayout(layoutInfo);
}

void Engine::createSwapImageViews() {
	vk::ImageViewCreateInfo imageViewCreateInfo{
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

	for (auto& swapchainData : m_swapchainData) {
		imageViewCreateInfo.image = swapchainData.image;
		swapchainData.imageView = vk::raii::ImageView(m_device, imageViewCreateInfo);
	}
}

void Engine::createGraphicsPipeline() {
	std::vector<char> shaderCode = readFile("shaders/slang.spv");
	vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

	vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
		.stage = vk::ShaderStageFlagBits::eVertex,
		.module = shaderModule,
		.pName = "vertMain"
	};

	vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
		.stage = vk::ShaderStageFlagBits::eFragment,
		.module = shaderModule,
		.pName = "fragMain"
	};

	std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {vertShaderStageInfo, fragShaderStageInfo};
	auto bindingDescription = Vertex::getBindingDescription();
	auto attributeDescriptions = Vertex::getAttributeDescriptions();

	vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &bindingDescription,
		.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
		.pVertexAttributeDescriptions = attributeDescriptions.data(),
	};
	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
	vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

	vk::PipelineRasterizationStateCreateInfo rasterizer{
		.depthClampEnable = vk::False,
		.rasterizerDiscardEnable = vk::False,
		.polygonMode = vk::PolygonMode::eFill,
		.cullMode = vk::CullModeFlagBits::eBack,
		.frontFace = vk::FrontFace::eCounterClockwise,
		.depthBiasEnable = vk::False,
		.lineWidth = 1.0f
	};

	vk::PipelineMultisampleStateCreateInfo multisampling{
		.rasterizationSamples = vk::SampleCountFlagBits::e1,
		.sampleShadingEnable = vk::False
	};

	vk::PipelineColorBlendAttachmentState colorBlendAttachment{
		.blendEnable = vk::False,
		.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
	};

	vk::PipelineColorBlendStateCreateInfo colorBlending{
		.logicOpEnable = vk::False,
		.logicOp = vk::LogicOp::eCopy,
		.attachmentCount = 1,
		.pAttachments = &colorBlendAttachment
	};

	std::array<vk::DynamicState, 2> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
	vk::PipelineDynamicStateCreateInfo dynamicState{
		.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
		.pDynamicStates = dynamicStates.data()
	};

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
		.setLayoutCount = 1,
		.pSetLayouts = &*m_descriptorSetLayout
	};
	m_pipelineLayout = vk::raii::PipelineLayout(m_device, pipelineLayoutInfo);

	vk::PipelineDepthStencilStateCreateInfo depthStencil{
		.depthTestEnable = vk::True,
		.depthWriteEnable = vk::True,
		.depthCompareOp = vk::CompareOp::eLess,
		.depthBoundsTestEnable = vk::False,
		.stencilTestEnable = vk::False
	};

	vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfo{
		{
			.stageCount = static_cast<uint32_t>(shaderStages.size()),
			.pStages = shaderStages.data(),
			.pVertexInputState = &vertexInputInfo,
			.pInputAssemblyState = &inputAssembly,
			.pViewportState = &viewportState,
			.pRasterizationState = &rasterizer,
			.pMultisampleState = &multisampling,
			.pDepthStencilState = &depthStencil,
			.pColorBlendState = &colorBlending,
			.pDynamicState = &dynamicState,
			.layout = m_pipelineLayout
		},
		{
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &m_swapChainSurfaceFormat.format,
			.depthAttachmentFormat = findDepthFormat()
		}
	};

	m_graphicsPipeline = vk::raii::Pipeline(m_device, nullptr, pipelineCreateInfo.get<vk::GraphicsPipelineCreateInfo>());
}

void Engine::createCommandPool() {
	vk::CommandPoolCreateInfo poolInfo{
		.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
		.queueFamilyIndex = m_graphicsQueueIndex
	};
	m_commandPool = vk::raii::CommandPool(m_device, poolInfo);

	if (hasDedicatedTransferQueueFamily()) {
		vk::CommandPoolCreateInfo transferPoolInfo{
			.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			.queueFamilyIndex = m_transferQueueIndex
		};
		m_transferCommandPool = vk::raii::CommandPool(m_device, transferPoolInfo);
	}
}

void Engine::createDepthResources() {
	vk::Format depthFormat = findDepthFormat();
	for (auto& swapchainData : m_swapchainData) {
		std::tie(swapchainData.depthImage, swapchainData.depthImageMemory) = createImage(
			m_swapChainExtent.width,
			m_swapChainExtent.height,
			depthFormat,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eDepthStencilAttachment,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);
		swapchainData.depthImageView = createImageView(*swapchainData.depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);
	}
}

void Engine::createVertexBuffer() {
	vk::DeviceSize bufferSize = sizeof(m_vertices[0]) * m_vertices.size();
	auto [stagingBuffer, stagingBufferMemory] = createBuffer(
		bufferSize,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
	);

	void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
	memcpy(dataStaging, m_vertices.data(), static_cast<size_t>(bufferSize));
	stagingBufferMemory.unmapMemory();

	std::tie(m_vertexBuffer, m_vertexBufferMemory) = createBuffer(
		bufferSize,
		vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal
	);

	copyBuffer(stagingBuffer, m_vertexBuffer, bufferSize);
}

void Engine::createIndexBuffer() {
	vk::DeviceSize bufferSize = sizeof(m_indices[0]) * m_indices.size();
	auto [stagingBuffer, stagingBufferMemory] = createBuffer(
		bufferSize,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
	);

	void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
	memcpy(dataStaging, m_indices.data(), static_cast<size_t>(bufferSize));
	stagingBufferMemory.unmapMemory();

	std::tie(m_indexBuffer, m_indexBufferMemory) = createBuffer(
		bufferSize,
		vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal
	);

	copyBuffer(stagingBuffer, m_indexBuffer, bufferSize);
}

std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> Engine::createMeshBuffer(vk::DeviceSize bufferSize, void* data, vk::BufferUsageFlagBits bufferType) {
	auto [stagingBuffer, stagingBufferMemory] = createBuffer(
		bufferSize,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
	);

	void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
	memcpy(dataStaging, data, static_cast<size_t>(bufferSize));
	stagingBufferMemory.unmapMemory();

	auto [meshBuffer, meshBufferMemory] = createBuffer(
		bufferSize,
		vk::BufferUsageFlagBits::eTransferDst | bufferType,
		vk::MemoryPropertyFlagBits::eDeviceLocal
	);

	copyBuffer(stagingBuffer, meshBuffer, bufferSize);
	return {std::move(meshBuffer), std::move(meshBufferMemory)};
}


void Engine::createUniformBuffers() {
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
		auto [buffer, bufferMem] = createBuffer(
			bufferSize,
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
		);
		auto& frame = m_frames[i];
		frame.uniformBuffer = std::move(buffer);
		frame.uniformBufferMemory = std::move(bufferMem);
		frame.uniformBufferMapped = frame.uniformBufferMemory.mapMemory(0, bufferSize);
	}
}

void Engine::createDescriptorPool() {
	vk::DescriptorPoolSize poolSize{
		.type = vk::DescriptorType::eUniformBuffer,
		.descriptorCount = MAX_FRAMES_IN_FLIGHT
	};

	vk::DescriptorPoolCreateInfo poolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = 4096,
		.poolSizeCount = 1,
		.pPoolSizes = &poolSize
	};

	m_descriptorPool = vk::raii::DescriptorPool(m_device, poolInfo);
}

void Engine::createDescriptorSets() {
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *m_descriptorSetLayout);
	vk::DescriptorSetAllocateInfo allocInfo{
		.descriptorPool = m_descriptorPool,
		.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
		.pSetLayouts = layouts.data()
	};
	auto descriptorSets = m_device.allocateDescriptorSets(allocInfo);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		auto& frame = m_frames[i];
		frame.descriptorSet = std::move(descriptorSets[i]);
		vk::DescriptorBufferInfo bufferInfo{
			.buffer = frame.uniformBuffer,
			.offset = 0,
			.range = sizeof(UniformBufferObject)
		};
		vk::WriteDescriptorSet descriptorWrite{
			.dstSet = frame.descriptorSet,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = vk::DescriptorType::eUniformBuffer,
			.pBufferInfo = &bufferInfo,
		};
		m_device.updateDescriptorSets(descriptorWrite, {});
	}
}

uint32_t Engine::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
	vk::PhysicalDeviceMemoryProperties memProperties = m_physicalDevice.getMemoryProperties();
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");
}

uint32_t Engine::findQueueIndex(
	const std::vector<vk::QueueFamilyProperties>& queueFamilyProperties,
	vk::QueueFlagBits queueFlags,
	bool requireSurfaceSupport,
	vk::QueueFlags discouragedFlags
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

void Engine::createCommandBuffer() {
	vk::CommandBufferAllocateInfo allocInfo{
		.commandPool = m_commandPool,
		.level = vk::CommandBufferLevel::ePrimary,
		.commandBufferCount = MAX_FRAMES_IN_FLIGHT
	};

	auto commandBuffers = vk::raii::CommandBuffers(m_device, allocInfo);
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		m_frames[i].commandBuffer = std::move(commandBuffers[i]);
	}
}

void Engine::createSyncObjects() {
	for (auto& swapchainData : m_swapchainData) {
		swapchainData.renderFinishedSemaphore = vk::raii::Semaphore(m_device, vk::SemaphoreCreateInfo());
	}

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		auto& frame = m_frames[i];
		frame.presentCompleteSemaphore = vk::raii::Semaphore(m_device, vk::SemaphoreCreateInfo());
		frame.inFlightFence = vk::raii::Fence(m_device, vk::FenceCreateInfo{
			.flags = vk::FenceCreateFlagBits::eSignaled
		});
	}
}

void Engine::recordCommandBuffer(uint32_t index) {
	auto& commandBuffer = currentFrame().commandBuffer;
	auto& swapchainData = currentSwapchainData(index);
	commandBuffer.begin({});

	transitionImageLayout(
		swapchainData.image,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eColorAttachmentOptimal,
		{},
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::ImageAspectFlagBits::eColor
	);
	transitionImageLayout(
		*swapchainData.depthImage,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eDepthAttachmentOptimal,
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::ImageAspectFlagBits::eDepth
	);

	vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
	vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
	vk::RenderingAttachmentInfo attachmentInfo{
		.imageView = swapchainData.imageView,
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = clearColor
	};

	vk::RenderingAttachmentInfo depthAttachmentInfo{
		.imageView = swapchainData.depthImageView,
		.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eDontCare,
		.clearValue = clearDepth
	};

	vk::RenderingInfo renderingInfo{
		.renderArea = {.offset = {0, 0}, .extent = m_swapChainExtent},
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &attachmentInfo,
		.pDepthAttachment = &depthAttachmentInfo
	};

	commandBuffer.beginRendering(renderingInfo);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_graphicsPipeline);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(m_swapChainExtent.width), static_cast<float>(m_swapChainExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D({0, 0}, m_swapChainExtent));


	for (const auto& [entity, meshResource] : m_meshResources) {
		commandBuffer.bindVertexBuffers(0, *meshResource.vertexBuffer, {0});
		commandBuffer.bindIndexBuffer(*meshResource.indexBuffer, 0, vk::IndexType::eUint16);
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, *currentFrame().entityResources[entity].descriptorSet, nullptr);
		commandBuffer.drawIndexed(meshResource.indiceSize, 1, 0, 0, 0);
	}

	// commandBuffer.bindVertexBuffers(0, *m_vertexBuffer, {0});
	// commandBuffer.bindIndexBuffer(*m_indexBuffer, 0, vk::IndexType::eUint16);
	// commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, *currentFrame().descriptorSet, nullptr);
	// commandBuffer.drawIndexed(static_cast<uint32_t>(m_indices.size()), 1, 0, 0, 0);
	commandBuffer.endRendering();

	transitionImageLayout(
		swapchainData.image,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::ePresentSrcKHR,
		vk::AccessFlagBits2::eColorAttachmentWrite,
		{},
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::PipelineStageFlagBits2::eBottomOfPipe,
		vk::ImageAspectFlagBits::eColor
	);

	commandBuffer.end();
}

void Engine::transitionImageLayout(
	vk::Image image,
	vk::ImageLayout oldLayout,
	vk::ImageLayout newLayout,
	vk::AccessFlags2 srcAccessMask,
	vk::AccessFlags2 dstAccessMask,
	vk::PipelineStageFlags2 srcStageMask,
	vk::PipelineStageFlags2 dstStageMask,
	vk::ImageAspectFlags imageAspectFlags
) {
	vk::ImageMemoryBarrier2 barrier{
		.srcStageMask = srcStageMask,
		.srcAccessMask = srcAccessMask,
		.dstStageMask = dstStageMask,
		.dstAccessMask = dstAccessMask,
		.oldLayout = oldLayout,
		.newLayout = newLayout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			imageAspectFlags,
			0,
			1,
			0,
			1
		}
	};
	vk::DependencyInfo dependencyInfo{
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &barrier
	};
	currentFrame().commandBuffer.pipelineBarrier2(dependencyInfo);
}

std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> Engine::createBuffer(
	vk::DeviceSize size,
	vk::BufferUsageFlags usage,
	vk::MemoryPropertyFlags properties
) {
	vk::BufferCreateInfo bufferInfo{
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
	vk::MemoryAllocateInfo memoryAllocateInfo{
		.allocationSize = memRequirements.size,
		.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
	};
	vk::raii::DeviceMemory bufferMemory = vk::raii::DeviceMemory(m_device, memoryAllocateInfo);
	buffer.bindMemory(bufferMemory, 0);
	return {std::move(buffer), std::move(bufferMemory)};
}

void Engine::copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size) {
	vk::CommandBufferAllocateInfo allocInfo{
		.commandPool = transferCommandPool(),
		.level = vk::CommandBufferLevel::ePrimary,
		.commandBufferCount = 1
	};
	vk::raii::CommandBuffer commandCopyBuffer = std::move(m_device.allocateCommandBuffers(allocInfo).front());
	commandCopyBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
	commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy(0, 0, size));
	commandCopyBuffer.end();
	transferQueue().submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandCopyBuffer}, nullptr);
	transferQueue().waitIdle();
}

vk::SurfaceFormatKHR Engine::chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& availableFormats) {
	assert(!availableFormats.empty());
	auto formatIt = std::ranges::find_if(availableFormats, [](const auto& format) {
		return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
	});
	return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
}

vk::Format Engine::findSupportedFormat(std::span<const vk::Format> candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) {
	for (const auto format : candidates) {
		vk::FormatProperties props = m_physicalDevice.getFormatProperties(format);
		if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
			return format;
		}
		if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
			return format;
		}
	}

	throw std::runtime_error("failed to find supported format");
}

vk::Format Engine::findDepthFormat() {
	return findSupportedFormat(
		std::array{vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
		vk::ImageTiling::eOptimal,
		vk::FormatFeatureFlagBits::eDepthStencilAttachment
	);
}

vk::PresentModeKHR Engine::chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const& availablePresentModes) {
	assert(std::ranges::any_of(availablePresentModes, [](const auto& mode) {
		return mode == vk::PresentModeKHR::eFifo;
	}));
	return std::ranges::any_of(availablePresentModes, [](const auto& mode) {
		return mode == vk::PresentModeKHR::eMailbox;
	}) ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
}

vk::Extent2D Engine::chooseSwapExtent(vk::SurfaceCapabilitiesKHR const& capabilities) {
	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
		return capabilities.currentExtent;
	}

	int width;
	int height;
	glfwGetFramebufferSize(m_window, &width, &height);
	return {
		std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
		std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
	};
}

uint32_t Engine::chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities) {
	uint32_t minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
	if (surfaceCapabilities.maxImageCount > 0 && surfaceCapabilities.maxImageCount < minImageCount) {
		minImageCount = surfaceCapabilities.maxImageCount;
	}
	return minImageCount;
}

std::vector<char> Engine::readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file!");
	}

	std::vector<char> buffer(file.tellg());
	file.seekg(0, std::ios::beg);
	file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
	return buffer;
}

void Engine::cleanupSwapChain() {
	m_swapChain = nullptr;
}

void Engine::recreateSwapChain() {
	int width = 0;
	int height = 0;
	glfwGetFramebufferSize(m_window, &width, &height);
	while (width == 0 || height == 0) {
		glfwGetFramebufferSize(m_window, &width, &height);
		glfwWaitEvents();
	}

	m_device.waitIdle();
	cleanupSwapChain();
	createSwapChain();
}

std::pair<vk::raii::Image, vk::raii::DeviceMemory> Engine::createImage(
	uint32_t width,
	uint32_t height,
	vk::Format format,
	vk::ImageTiling tiling,
	vk::ImageUsageFlags usage,
	vk::MemoryPropertyFlags properties
) {
	vk::ImageCreateInfo imageInfo{
		.imageType = vk::ImageType::e2D,
		.format = format,
		.extent = {width, height, 1},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = vk::SampleCountFlagBits::e1,
		.tiling = tiling,
		.usage = usage,
		.sharingMode = vk::SharingMode::eExclusive
	};

	vk::raii::Image image = vk::raii::Image(m_device, imageInfo);
	vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
	vk::MemoryAllocateInfo allocInfo{
		.allocationSize = memRequirements.size,
		.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
	};

	vk::raii::DeviceMemory imageMemory = vk::raii::DeviceMemory(m_device, allocInfo);
	image.bindMemory(imageMemory, 0);
	return {std::move(image), std::move(imageMemory)};
}

vk::raii::ImageView Engine::createImageView(const vk::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags) {
	vk::ImageViewCreateInfo viewInfo{
		.image = image,
		.viewType = vk::ImageViewType::e2D,
		.format = format,
		.subresourceRange = {
			.aspectMask = aspectFlags,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1
		}
	};
	return vk::raii::ImageView(m_device, viewInfo);
}

vk::raii::ShaderModule Engine::createShaderModule(const std::vector<char>& code) const {
	vk::ShaderModuleCreateInfo createInfo{
		.codeSize = code.size() * sizeof(char),
		.pCode = reinterpret_cast<const uint32_t*>(code.data())
	};
	return vk::raii::ShaderModule(m_device, createInfo);
}

std::vector<const char*> Engine::getRequiredInstanceExtensions() {
	uint32_t glfwExtensionCount = 0;
	auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
	return std::vector<const char*>(glfwExtensions, glfwExtensions + glfwExtensionCount);
}

void Engine::cleanup() {
	if (m_device != nullptr) {
		m_device.waitIdle();
	}

	cleanupSwapChain();

	if (m_window != nullptr) {
		glfwDestroyWindow(m_window);
		m_window = nullptr;
	}

	if (m_glfwInitialized) {
		glfwTerminate();
		m_glfwInitialized = false;
	}
}
