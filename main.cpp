#include <memory>
#include <vector>
#include <cstdint>
#include <limits>
#include <fstream>
#include <iostream>
#include <map>
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

const uint32_t WIDTH  = 800;
const uint32_t HEIGHT = 600;
const std::vector<char const*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

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
	vk::raii::SurfaceKHR m_surface = nullptr;
	vk::raii::SwapchainKHR m_swapChain = nullptr;
	uint32_t m_graphicsIndex = ~0;
	vk::Extent2D m_swapChainExtent;
	vk::SurfaceFormatKHR m_swapChainSurfaceFormat;
	std::vector<vk::Image> m_swapChainImages;
	std::vector<vk::raii::ImageView> m_swapChainImageViews;
	vk::raii::PipelineLayout m_pipelineLayout = nullptr;
	vk::raii::Pipeline m_graphicsPipeline = nullptr;
	vk::raii::CommandPool m_commandPool = nullptr;
	vk::raii::CommandBuffer m_commandBuffer = nullptr;
	vk::raii::Semaphore m_presentCompleteSemaphore = nullptr;
	vk::raii::Semaphore m_renderFinishedSemaphore = nullptr;
	vk::raii::Fence m_drawFence = nullptr;


	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	}

	void initVulkan()
	{
		createInstance();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createGraphicsPipeline();
		createCommandPool();
		createCommandBuffer();
		createSyncObjects();
	}

	void drawFrame() {
		auto fenceResult = m_device.waitForFences(*m_drawFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
		if (fenceResult != vk::Result::eSuccess) {
			throw std::runtime_error("Failed to wait for fence");
		}
		m_device.resetFences(*m_drawFence);

		auto [result, imageIndex] = m_swapChain.acquireNextImage(std::numeric_limits<uint64_t>::max(), *m_presentCompleteSemaphore, nullptr);
		recordCommandBuffer(imageIndex);

		m_graphicsQueue.waitIdle();

		vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		const vk::SubmitInfo submitInfo {
			.waitSemaphoreCount = 1,
			.pWaitSemaphores    = &*m_presentCompleteSemaphore,
			.pWaitDstStageMask  = &waitDestinationStageMask,
			.commandBufferCount = 1,
			.pCommandBuffers    = &*m_commandBuffer,
			.signalSemaphoreCount = 1,
			.pSignalSemaphores   = &*m_renderFinishedSemaphore
		};

		m_graphicsQueue.submit(submitInfo, *m_drawFence);

		const vk::PresentInfoKHR presentInfoKHR {
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &*m_renderFinishedSemaphore,
			.swapchainCount      = 1,
			.pSwapchains         = &*m_swapChain,
			.pImageIndices       = &imageIndex
		};

		auto presentResult = m_graphicsQueue.presentKHR(presentInfoKHR);
		switch(presentResult) {
			case vk::Result::eSuccess:
				break;
			case vk::Result::eSuboptimalKHR:
				std::cout << "Suboptimal swap chain\n";
				break;
			default:
				throw std::runtime_error("Failed to present swap chain image");
		}
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

		for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++) {
			if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
				m_physicalDevice.getSurfaceSupportKHR(qfpIndex, *m_surface)) {
					m_graphicsIndex = qfpIndex;
					break;
				}
		}

		if (m_graphicsIndex == UINT_MAX) {
			throw std::runtime_error("Failed to find a suitable GPU");
		}

		vk::StructureChain<vk::PhysicalDeviceFeatures2,vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> featChain = {
			{},
			{.shaderDrawParameters = true},
			{ .synchronization2 = true, .dynamicRendering = true },
			{.extendedDynamicState = true}
		};

		std::vector<const char*> requiredDeviceExtension = {vk::KHRSwapchainExtensionName};

		float queuePriority = 0.5f;
		vk::DeviceQueueCreateInfo deviceQueueCreateInfo { .queueFamilyIndex = m_graphicsIndex, .queueCount = 1, .pQueuePriorities = &queuePriority};
		vk::DeviceCreateInfo deviceCreateInfo {
			.pNext = &featChain.get<vk::PhysicalDeviceFeatures2>(),
			.queueCreateInfoCount = 1,
			.pQueueCreateInfos = &deviceQueueCreateInfo,
			.enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
			.ppEnabledExtensionNames = requiredDeviceExtension.data()
		};

		m_device = vk::raii::Device(m_physicalDevice, deviceCreateInfo);
		m_graphicsQueue = vk::raii::Queue(m_device, m_graphicsIndex, 0);
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

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
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

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo {.setLayoutCount = 0, .pushConstantRangeCount = 0};
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
			.queueFamilyIndex = m_graphicsIndex
		};
		m_commandPool = vk::raii::CommandPool(m_device, poolInfo);
	}

	void createCommandBuffer() {
		vk::CommandBufferAllocateInfo allocInfo {
			.commandPool = m_commandPool,
			.level = vk::CommandBufferLevel::ePrimary,
			.commandBufferCount = 1
		};

		m_commandBuffer = std::move(vk::raii::CommandBuffers(m_device, allocInfo).front());
	}

	void createSyncObjects() {
		m_presentCompleteSemaphore = vk::raii::Semaphore(m_device, vk::SemaphoreCreateInfo());
		m_renderFinishedSemaphore = vk::raii::Semaphore(m_device, vk::SemaphoreCreateInfo());
		m_drawFence = vk::raii::Fence(m_device, {
			.flags = vk::FenceCreateFlagBits::eSignaled
		});
	}

	void recordCommandBuffer(uint32_t index) {
		m_commandBuffer.begin({});

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

		m_commandBuffer.beginRendering(renderingInfo);
		m_commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_graphicsPipeline);

		m_commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(m_swapChainExtent.width), static_cast<float>(m_swapChainExtent.height), 0.0f, 1.0f));
		m_commandBuffer.setScissor(0, vk::Rect2D({0, 0}, m_swapChainExtent));

		m_commandBuffer.draw(3, 1, 0, 0); // HOLY SHIIIT;

		m_commandBuffer.endRendering();

		transition_image_layout(
			index,
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::ePresentSrcKHR,
			vk::AccessFlagBits2::eColorAttachmentWrite,
			{},
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			vk::PipelineStageFlagBits2::eBottomOfPipe
		);
		
		m_commandBuffer.end();
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
		m_commandBuffer.pipelineBarrier2(dependency_info);
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

	void cleanup()
	{
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
