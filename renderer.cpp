#include "renderer.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "components/components_common.h"

Renderer::Renderer(GLFWwindow* window, Registry& registry, Entity camera)
	: m_window(window),
	  m_registry(registry),
	  m_camera(camera) {
	initVulkan();
}

Renderer::~Renderer() {
	cleanup();
}

void Renderer::createMeshEntity(Entity entity, size_t meshHandle) {
	m_entityToMesh[entity] = meshHandle;
	createUniformDescriptors(entity);
}

size_t Renderer::uploadMesh(const std::pair<std::vector<Vertex>, std::vector<uint16_t>>& meshData) {
	const vk::DeviceSize vertexBufferSize = meshData.first.size() * sizeof(Vertex);
	auto [vertexBuffer, vertexBufferMemory] = createMeshBuffer(
		vertexBufferSize,
		meshData.first.data(),
		vk::BufferUsageFlagBits::eVertexBuffer
	);

	const vk::DeviceSize indexBufferSize = meshData.second.size() * sizeof(uint16_t);
	auto [indexBuffer, indexBufferMemory] = createMeshBuffer(
		indexBufferSize,
		meshData.second.data(),
		vk::BufferUsageFlagBits::eIndexBuffer
	);

	m_meshResources.emplace_back(
		std::move(vertexBuffer),
		std::move(vertexBufferMemory),
		std::move(indexBuffer),
		std::move(indexBufferMemory),
		static_cast<uint16_t>(meshData.second.size())
	);

	return m_meshResources.size() - 1;
}

void Renderer::createUniformDescriptors(Entity entity) {
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *m_descriptorSetLayout);
	vk::DescriptorSetAllocateInfo allocInfo{
		.descriptorPool = m_descriptorPool,
		.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
		.pSetLayouts = layouts.data()
	};
	auto descriptorSets = m_device.allocateDescriptorSets(allocInfo);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		const vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
		auto [buffer, bufferMem] = createBuffer(
			bufferSize,
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
		);
		auto& frame = m_frames[i];
		frame.entityResources[entity].uniformBuffer = std::move(buffer);
		frame.entityResources[entity].uniformBufferMemory = std::move(bufferMem);
		frame.entityResources[entity].uniformBufferMapped =
			frame.entityResources[entity].uniformBufferMemory.mapMemory(0, bufferSize);

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

void Renderer::updateUniformBuffers() {
	static auto startTime = std::chrono::high_resolution_clock::now();
	const auto currentTime = std::chrono::high_resolution_clock::now();
	const float time =
		std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

	const Transform& cameraTransform = m_registry.get<Transform>().get(m_camera);
	glm::mat4 viewMat = glm::translate(glm::mat4(1.0f), cameraTransform.position);
	viewMat = glm::rotate(viewMat, cameraTransform.rotation.x, glm::vec3(1, 0, 0));
	viewMat = glm::rotate(viewMat, cameraTransform.rotation.y, glm::vec3(0, 1, 0));
	viewMat = glm::rotate(viewMat, cameraTransform.rotation.z, glm::vec3(0, 0, 1));

	auto view = m_registry.view<Transform, Mesh>();
	for (auto it = view.begin(); it != view.end(); ++it) {
		const Entity entity = it.entity();
		auto [transform, mesh] = *it;
		(void)mesh;
		auto& entityResources = currentFrame().entityResources[entity];

		glm::mat4 model = glm::translate(glm::mat4(1.0f), transform.position);
		model = glm::rotate(model, time * glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));

		glm::mat4 proj = glm::perspective(
			glm::radians(45.0f),
			m_swapChainExtent.width / static_cast<float>(m_swapChainExtent.height),
			0.1f,
			10.0f
		);
		proj[1][1] *= -1;

		const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(model)));
		const UniformBufferObject ubo{model, proj * viewMat, normalMatrix};
		memcpy(entityResources.uniformBufferMapped, &ubo, sizeof(ubo));
	}
}

void Renderer::drawFrame() {
	auto& frame = currentFrame();
	const auto fenceResult =
		m_device.waitForFences(*frame.inFlightFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
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
	updateUniformBuffers();
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

	const auto presentResult = m_graphicsQueue.presentKHR(presentInfoKHR);
	if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR ||
		m_framebufferResized) {
		recreateSwapChain();
	} else {
		assert(result == vk::Result::eSuccess);
	}

	m_frameIndex = (m_frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Renderer::onFramebufferResized() {
	m_framebufferResized = true;
}

bool Renderer::hasDedicatedTransferQueueFamily() const {
	return m_transferQueueIndex != m_graphicsQueueIndex;
}

vk::raii::Queue& Renderer::transferQueue() {
	return hasDedicatedTransferQueueFamily() ? m_transferQueue : m_graphicsQueue;
}

vk::raii::CommandPool& Renderer::transferCommandPool() {
	return hasDedicatedTransferQueueFamily() ? m_transferCommandPool : m_commandPool;
}

FrameData& Renderer::currentFrame() {
	return m_frames[m_frameIndex];
}

SwapchainData& Renderer::currentSwapchainData(uint32_t imageIndex) {
	return m_swapchainData[imageIndex];
}

void Renderer::initVulkan() {
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

void Renderer::createFrameResources() {
	m_frames.resize(MAX_FRAMES_IN_FLIGHT);
}

void Renderer::createInstance() {
	constexpr vk::ApplicationInfo appInfo{
		.pApplicationName = "Hello Triangle",
		.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
		.pEngineName = "No Engine",
		.engineVersion = VK_MAKE_VERSION(1, 0, 0),
		.apiVersion = vk::ApiVersion14,
	};

	auto requiredExtensions = getRequiredInstanceExtensions();
	const auto extensionProperties = m_context.enumerateInstanceExtensionProperties();

	for (const auto& requiredExtension : requiredExtensions) {
		const auto it = std::ranges::find_if(extensionProperties, [requiredExtension](const auto& extensionProperty) {
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

	const auto layerProperties = m_context.enumerateInstanceLayerProperties();
	for (const auto& requiredLayer : requiredLayers) {
		const auto it = std::ranges::find_if(layerProperties, [requiredLayer](const auto& layerProperty) {
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

void Renderer::createSurface() {
	VkSurfaceKHR windowSurface;
	if (glfwCreateWindowSurface(*m_instance, m_window, nullptr, &windowSurface) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create window surface");
	}
	m_surface = vk::raii::SurfaceKHR(m_instance, windowSurface);
}

void Renderer::pickPhysicalDevice() {
	std::vector<vk::raii::PhysicalDevice> physicalDevices = m_instance.enumeratePhysicalDevices();
	const auto deviceIt = std::ranges::find_if(physicalDevices, [&](auto const& physicalDevice) {
		return isDeviceSuitable(physicalDevice);
	});
	if (deviceIt == physicalDevices.end()) {
		throw std::runtime_error("Failed to find GPUs with Vulkan support");
	}

	m_physicalDevice = *deviceIt;
}

void Renderer::createLogicalDevice() {
	const std::vector<vk::QueueFamilyProperties> queueFamilyProperties = m_physicalDevice.getQueueFamilyProperties();

	m_graphicsQueueIndex = findQueueIndex(queueFamilyProperties, vk::QueueFlagBits::eGraphics, true);
	m_transferQueueIndex =
		findQueueIndex(queueFamilyProperties, vk::QueueFlagBits::eTransfer, false, vk::QueueFlagBits::eGraphics);

	vk::StructureChain<
		vk::PhysicalDeviceFeatures2,
		vk::PhysicalDeviceVulkan11Features,
		vk::PhysicalDeviceVulkan13Features,
		vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
		featChain = {
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

bool Renderer::isDeviceSuitable(vk::raii::PhysicalDevice const& physicalDevice) {
	const bool supportsVulkan13 = physicalDevice.getProperties().apiVersion >= VK_API_VERSION_1_3;
	const auto queueFamilies = physicalDevice.getQueueFamilyProperties();
	const bool supportsGraphics = std::ranges::any_of(queueFamilies, [](auto const& qfp) {
		return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
	});

	std::vector<const char*> requiredDeviceExtension = {vk::KHRSwapchainExtensionName};
	const auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
	const bool supportsAllRequiredExtensions =
		std::ranges::all_of(requiredDeviceExtension, [&availableDeviceExtensions](auto const& requiredDeviceExtensionName) {
			return std::ranges::any_of(availableDeviceExtensions, [requiredDeviceExtensionName](auto const& availableDeviceExtension) {
				return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtensionName) == 0;
			});
		});

	const auto feature = physicalDevice.template getFeatures2<
		vk::PhysicalDeviceFeatures2,
		vk::PhysicalDeviceVulkan11Features,
		vk::PhysicalDeviceVulkan13Features,
		vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
	const bool supportsAllRequiredFeatures =
		feature.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
		feature.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
		feature.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
		feature.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

	return supportsVulkan13 && supportsGraphics && supportsAllRequiredExtensions && supportsAllRequiredFeatures;
}

void Renderer::createSwapChain() {
	const vk::SurfaceCapabilitiesKHR surfaceCapabilities = m_physicalDevice.getSurfaceCapabilitiesKHR(*m_surface);
	m_swapChainExtent = chooseSwapExtent(surfaceCapabilities);
	const uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

	const std::vector<vk::SurfaceFormatKHR> availableSurfaceFormats = m_physicalDevice.getSurfaceFormatsKHR(*m_surface);
	m_swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableSurfaceFormats);

	const std::vector<vk::PresentModeKHR> presentModes = m_physicalDevice.getSurfacePresentModesKHR(*m_surface);
	const vk::PresentModeKHR swapPresentMode = chooseSwapPresentMode(presentModes);

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
	const auto swapChainImages = m_swapChain.getImages();
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

void Renderer::createDescriptorSetLayout() {
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

void Renderer::createSwapImageViews() {
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

void Renderer::createGraphicsPipeline() {
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
		.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
						  vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
	};

	vk::PipelineColorBlendStateCreateInfo colorBlending{
		.logicOpEnable = vk::False,
		.logicOp = vk::LogicOp::eCopy,
		.attachmentCount = 1,
		.pAttachments = &colorBlendAttachment
	};

	std::array<vk::DynamicState, 2> dynamicStates = {
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor
	};
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

	m_graphicsPipeline = vk::raii::Pipeline(
		m_device,
		nullptr,
		pipelineCreateInfo.get<vk::GraphicsPipelineCreateInfo>()
	);
}

void Renderer::createCommandPool() {
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

void Renderer::createDepthResources() {
	const vk::Format depthFormat = findDepthFormat();
	for (auto& swapchainData : m_swapchainData) {
		std::tie(swapchainData.depthImage, swapchainData.depthImageMemory) = createImage(
			m_swapChainExtent.width,
			m_swapChainExtent.height,
			depthFormat,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eDepthStencilAttachment,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);
		swapchainData.depthImageView =
			createImageView(*swapchainData.depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);
	}
}

std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> Renderer::createMeshBuffer(
	vk::DeviceSize bufferSize,
	const void* data,
	vk::BufferUsageFlagBits bufferType
) {
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

void Renderer::createDescriptorPool() {
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

uint32_t Renderer::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
	const vk::PhysicalDeviceMemoryProperties memProperties = m_physicalDevice.getMemoryProperties();
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");
}

uint32_t Renderer::findQueueIndex(
	const std::vector<vk::QueueFamilyProperties>& queueFamilyProperties,
	vk::QueueFlagBits queueFlags,
	bool requireSurfaceSupport,
	vk::QueueFlags discouragedFlags
) {
	uint32_t fallbackIndex = UINT_MAX;
	for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++) {
		const vk::QueueFlags flags = queueFamilyProperties[qfpIndex].queueFlags;
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

void Renderer::createCommandBuffer() {
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

void Renderer::createSyncObjects() {
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

void Renderer::recordCommandBuffer(uint32_t imageIndex) {
	auto& commandBuffer = currentFrame().commandBuffer;
	auto& swapchainData = currentSwapchainData(imageIndex);
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
	commandBuffer.setViewport(
		0,
		vk::Viewport(
			0.0f,
			0.0f,
			static_cast<float>(m_swapChainExtent.width),
			static_cast<float>(m_swapChainExtent.height),
			0.0f,
			1.0f
		)
	);
	commandBuffer.setScissor(0, vk::Rect2D({0, 0}, m_swapChainExtent));

	auto view = m_registry.view<Transform, Mesh>();
	for (auto it = view.begin(); it != view.end(); ++it) {
		const Entity entity = it.entity();
		const auto& meshResource = m_meshResources[m_entityToMesh[entity]];
		commandBuffer.bindVertexBuffers(0, *meshResource.vertexBuffer, {0});
		commandBuffer.bindIndexBuffer(*meshResource.indexBuffer, 0, vk::IndexType::eUint16);
		commandBuffer.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			m_pipelineLayout,
			0,
			*currentFrame().entityResources[entity].descriptorSet,
			nullptr
		);
		commandBuffer.drawIndexed(meshResource.indiceSize, 1, 0, 0, 0);
	}

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

void Renderer::transitionImageLayout(
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

std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> Renderer::createBuffer(
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
	const vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
	vk::MemoryAllocateInfo memoryAllocateInfo{
		.allocationSize = memRequirements.size,
		.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
	};
	vk::raii::DeviceMemory bufferMemory = vk::raii::DeviceMemory(m_device, memoryAllocateInfo);
	buffer.bindMemory(bufferMemory, 0);
	return {std::move(buffer), std::move(bufferMemory)};
}

void Renderer::copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size) {
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

vk::SurfaceFormatKHR Renderer::chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& availableFormats) {
	assert(!availableFormats.empty());
	const auto formatIt = std::ranges::find_if(availableFormats, [](const auto& format) {
		return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
	});
	return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
}

vk::Format Renderer::findSupportedFormat(
	std::span<const vk::Format> candidates,
	vk::ImageTiling tiling,
	vk::FormatFeatureFlags features
) {
	for (const auto format : candidates) {
		const vk::FormatProperties props = m_physicalDevice.getFormatProperties(format);
		if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
			return format;
		}
		if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
			return format;
		}
	}

	throw std::runtime_error("failed to find supported format");
}

vk::Format Renderer::findDepthFormat() {
	return findSupportedFormat(
		std::array{vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
		vk::ImageTiling::eOptimal,
		vk::FormatFeatureFlagBits::eDepthStencilAttachment
	);
}

vk::PresentModeKHR Renderer::chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const& availablePresentModes) {
	assert(std::ranges::any_of(availablePresentModes, [](const auto& mode) {
		return mode == vk::PresentModeKHR::eFifo;
	}));
	return std::ranges::any_of(availablePresentModes, [](const auto& mode) {
		return mode == vk::PresentModeKHR::eMailbox;
	})
		? vk::PresentModeKHR::eMailbox
		: vk::PresentModeKHR::eFifo;
}

vk::Extent2D Renderer::chooseSwapExtent(vk::SurfaceCapabilitiesKHR const& capabilities) {
	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
		return capabilities.currentExtent;
	}

	int width = 0;
	int height = 0;
	glfwGetFramebufferSize(m_window, &width, &height);
	return {
		std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
		std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
	};
}

uint32_t Renderer::chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities) {
	uint32_t minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
	if (surfaceCapabilities.maxImageCount > 0 && surfaceCapabilities.maxImageCount < minImageCount) {
		minImageCount = surfaceCapabilities.maxImageCount;
	}
	return minImageCount;
}

std::vector<char> Renderer::readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file!");
	}

	std::vector<char> buffer(file.tellg());
	file.seekg(0, std::ios::beg);
	file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
	return buffer;
}

void Renderer::cleanupSwapChain() {
	m_swapChain = nullptr;
}

void Renderer::recreateSwapChain() {
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
	m_framebufferResized = false;
}

std::pair<vk::raii::Image, vk::raii::DeviceMemory> Renderer::createImage(
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
	const vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
	vk::MemoryAllocateInfo allocInfo{
		.allocationSize = memRequirements.size,
		.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
	};

	vk::raii::DeviceMemory imageMemory = vk::raii::DeviceMemory(m_device, allocInfo);
	image.bindMemory(imageMemory, 0);
	return {std::move(image), std::move(imageMemory)};
}

vk::raii::ImageView Renderer::createImageView(
	const vk::Image& image,
	vk::Format format,
	vk::ImageAspectFlags aspectFlags
) {
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

vk::raii::ShaderModule Renderer::createShaderModule(const std::vector<char>& code) const {
	vk::ShaderModuleCreateInfo createInfo{
		.codeSize = code.size() * sizeof(char),
		.pCode = reinterpret_cast<const uint32_t*>(code.data())
	};
	return vk::raii::ShaderModule(m_device, createInfo);
}

std::vector<const char*> Renderer::getRequiredInstanceExtensions() {
	uint32_t glfwExtensionCount = 0;
	const auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
	return std::vector<const char*>(glfwExtensions, glfwExtensions + glfwExtensionCount);
}

void Renderer::cleanup() {
	if (m_device != nullptr) {
		m_device.waitIdle();
	}

	cleanupSwapChain();
}
