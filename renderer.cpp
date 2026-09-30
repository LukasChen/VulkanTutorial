#include "renderer.h"
#include "resourceUtils.h"

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <utility>


#include "components/components_common.h"
#include "transformAccess.h"

Renderer::Renderer(GLFWwindow* window, Registry& registry, ResourceManager& resource)
	: m_window(window),
	  m_registry(registry),
	  m_resource(resource) {
	initVulkan();
}

Renderer::~Renderer() {
	cleanup();
}

void Renderer::createMeshEntity(Entity entity) {
	const MeshRenderer* mesh = m_registry.get<MeshRenderer>().tryGet(entity);
	if (mesh == nullptr) {
		return;
	}

	const InstanceBatchKey batchKey {
		.meshHandle = mesh->meshHandle,
		.meshRenderer = mesh->matHandle
	};
	auto batchIt = m_instanceBatchToIndex.find(batchKey);
	if (batchIt == m_instanceBatchToIndex.end()) {
		const uint32_t batchIndex = static_cast<uint32_t>(m_instanceBatches.size());
		batchIt = m_instanceBatchToIndex.emplace(batchKey, batchIndex).first;
		m_instanceBatches.push_back({
			.meshHandle = mesh->meshHandle,
			.matHandle = mesh->matHandle,
			.firstInstance = static_cast<uint32_t>(m_instanceCount),
			.instanceCount = 0
		});
	}

	InstanceBatch& batch = m_instanceBatches[batchIt->second];
	batch.instanceCount++;
	m_instanceCount++;

	for (uint32_t batchIndex = batchIt->second + 1; batchIndex < m_instanceBatches.size(); batchIndex++) {
		m_instanceBatches[batchIndex].firstInstance++;
	}
}

void Renderer::createSkinMeshEntity(Entity entity) {
	SkinMeshRenderer* skinMeshRend = m_registry.get<SkinMeshRenderer>().tryGet(entity);

	if (skinMeshRend == nullptr) {
		return;
	}

	const SkinMeshResources& skinMesh = m_skinMeshes[skinMeshRend->skinMeshHandle];

	skinMeshRend->skinResourceHandle = m_skinResourceCount++;

	for (FrameData& frame : m_frames) {
		SkinFrameResource frameResource = createSkinFrameResource(skinMesh);
		createSkinDescriptorSet(frameResource, skinMesh);

		// Mock Anim data
		std::vector<glm::mat4> jointMatrices(skinMesh.jointCount, glm::mat4(1.0f));
		std::memcpy(frameResource.jointMatrixMapped, jointMatrices.data(), jointMatrices.size() * sizeof(glm::mat4));

		frame.resources.skinInstances.push_back(std::move(frameResource));
	}
}

void Renderer::setSkinPalette() {
	FrameResources& frameResources = currentFrame().resources;

	for (auto [animator, skinMeshRenderer] : m_registry.view<Animator, SkinMeshRenderer>()) {
		const SkinMeshResources& skinMesh = m_skinMeshes[skinMeshRenderer.skinMeshHandle];
		SkinFrameResource& skinFrame = frameResources.skinInstances[skinMeshRenderer.skinResourceHandle];

		if (animator.jointPalette.size() != skinMesh.jointCount) {
			throw std::runtime_error("Animator palette does not match the skinned mesh skeleton");
		}

		std::memcpy(
			skinFrame.jointMatrixMapped,
			animator.jointPalette.data(),
			animator.jointPalette.size() * sizeof(glm::mat4)
		);
	}
}

SkinFrameResource Renderer::createSkinFrameResource(const SkinMeshResources& skinMesh) {
	SkinFrameResource skinFrame;

	const vk::DeviceSize outputSize = skinMesh.vertexCount * sizeof(Vertex);

	std::tie(skinFrame.outputVertexBuffer, skinFrame.outputVertexMemory) = createBuffer(
		outputSize,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer,
		vk::MemoryPropertyFlagBits::eDeviceLocal
	);

	const vk::DeviceSize jointMatrixSize = skinMesh.jointCount * sizeof(glm::mat4);
	std::tie(skinFrame.jointMatrixBuffer, skinFrame.jointMatrixMemory) = createBuffer(
		jointMatrixSize,
		vk::BufferUsageFlagBits::eStorageBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
	);

	skinFrame.jointMatrixMapped = skinFrame.jointMatrixMemory.mapMemory(0, jointMatrixSize);

	return skinFrame;
}

void Renderer::createSkinDescriptorSet(SkinFrameResource& frame, const SkinMeshResources& mesh) {
	vk::DescriptorSetAllocateInfo allocInfo {
		.descriptorPool = m_descriptorPool,
		.descriptorSetCount = 1,
		.pSetLayouts = &*m_skinDescriptorSetLayout
	};

	auto descriptorSets = m_device.allocateDescriptorSets(allocInfo);
	frame.descriptorSet = std::move(descriptorSets.front());

	vk::DescriptorBufferInfo inputVertexInfo {
		.buffer = mesh.inputVertexBuffer,
		.offset = 0,
		.range = mesh.vertexCount * sizeof(Vertex)
	};

	vk::DescriptorBufferInfo outputVertexInfo {
		.buffer = frame.outputVertexBuffer,
		.offset = 0,
		.range = mesh.vertexCount * sizeof(Vertex)
	};

	vk::DescriptorBufferInfo jointMatrixInfo {
		.buffer = frame.jointMatrixBuffer,
		.offset = 0,
		.range = mesh.jointCount * sizeof(glm::mat4)
	};

	vk::DescriptorBufferInfo jointIndexInfo {
		.buffer = mesh.jointIndexBuffer,
		.offset = 0,
		.range = mesh.vertexCount * sizeof(glm::uvec4)
	};

	vk::DescriptorBufferInfo weightInfo {
		.buffer = mesh.weightBuffer,
		.offset = 0,
		.range = mesh.vertexCount * sizeof(glm::vec4)
	};

	std::array<vk::WriteDescriptorSet, 5> writes = {{
		{
			.dstSet = frame.descriptorSet,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = vk::DescriptorType::eStorageBuffer,
			.pBufferInfo = &inputVertexInfo
		},
		{
			.dstSet = frame.descriptorSet,
			.dstBinding = 1,
			.descriptorCount = 1,
			.descriptorType = vk::DescriptorType::eStorageBuffer,
			.pBufferInfo = &outputVertexInfo
		},
		{
			.dstSet = frame.descriptorSet,
			.dstBinding = 2,
			.descriptorCount = 1,
			.descriptorType = vk::DescriptorType::eStorageBuffer,
			.pBufferInfo = &jointMatrixInfo
		},
		{
			.dstSet = frame.descriptorSet,
			.dstBinding = 3,
			.descriptorCount = 1,
			.descriptorType = vk::DescriptorType::eStorageBuffer,
			.pBufferInfo = &jointIndexInfo
		},
		{
			.dstSet = frame.descriptorSet,
			.dstBinding = 4,
			.descriptorCount = 1,
			.descriptorType = vk::DescriptorType::eStorageBuffer,
			.pBufferInfo = &weightInfo
		}
	}};

	m_device.updateDescriptorSets(writes, {});
}

size_t Renderer::uploadMesh(const Model& model) {
	const Mesh& mesh = model.mesh;
	const vk::DeviceSize vertexBufferSize = mesh.vertices.size() * sizeof(Vertex);
	auto [vertexBuffer, vertexBufferMemory] = createMeshBuffer(
		vertexBufferSize,
		mesh.vertices.data(),
		vk::BufferUsageFlagBits::eVertexBuffer
	);

	const vk::DeviceSize indexBufferSize = mesh.indices.size() * sizeof(uint16_t);
	auto [indexBuffer, indexBufferMemory] = createMeshBuffer(
		indexBufferSize,
		mesh.indices.data(),
		vk::BufferUsageFlagBits::eIndexBuffer
	);

	m_meshResources.emplace_back(
		std::move(vertexBuffer),
		std::move(vertexBufferMemory),
		std::move(indexBuffer),
		std::move(indexBufferMemory),
		static_cast<uint32_t>(mesh.indices.size())
	);

	return m_meshResources.size() - 1;
}

size_t Renderer::uploadSkinnedMesh(const Model& model) {
	if (!model.isSkinned()) {
		return -1;
	}

	const Mesh& mesh = model.mesh;
	const AnimationAsset& animationAsset = m_resource.getAnimationAsset(model.animaitonAssetHandle);
	const vk::DeviceSize vertexBufferSize = mesh.vertices.size() * sizeof(Vertex);
	auto [vertexBuffer, vertexBufferMemory] = createMeshBuffer(
		vertexBufferSize,
		mesh.vertices.data(),
		vk::BufferUsageFlagBits::eStorageBuffer
	);

	const vk::DeviceSize indexBufferSize = mesh.indices.size() * sizeof(uint16_t);
	auto [indexBuffer, indexBufferMemory] = createMeshBuffer(
		indexBufferSize,
		mesh.indices.data(),
		vk::BufferUsageFlagBits::eIndexBuffer
	);

	const vk::DeviceSize weightBufferSize = model.skinning->weights.size() * sizeof(glm::vec4);
	auto [weightBuffer, weightBufferMemory] = createMeshBuffer(
		weightBufferSize,
		model.skinning->weights.data(),
		vk::BufferUsageFlagBits::eStorageBuffer
	);

	const vk::DeviceSize jointIndexBufferSize = model.skinning->jointIndices.size() * sizeof(glm::uvec4);
	auto [jointIndexBuffer, jointIndexBufferMemory] = createMeshBuffer(
		jointIndexBufferSize,
		model.skinning->jointIndices.data(),
		vk::BufferUsageFlagBits::eStorageBuffer
	);

	SkinMeshResources skinMesh {
		.inputVertexBuffer = std::move(vertexBuffer),
		.inputVertexMemory = std::move(vertexBufferMemory),
		.jointIndexBuffer = std::move(jointIndexBuffer),
		.jointIndexMemory = std::move(jointIndexBufferMemory),
		.weightBuffer = std::move(weightBuffer),
		.weightMemory = std::move(weightBufferMemory),
		.indexBuffer = std::move(indexBuffer),
		.indexMemory = std::move(indexBufferMemory),
		.vertexCount = static_cast<uint32_t>(mesh.vertices.size()),
		.indexCount = static_cast<uint32_t>(mesh.indices.size()),
		.jointCount = static_cast<uint32_t>(animationAsset.skeleton.joints.size()),
	};

	m_skinMeshes.push_back(std::move(skinMesh));

	return m_skinMeshes.size() - 1;
}


TextureHandle Renderer::uploadTexture(const stbi_uc* pixels, int width, int height) {
	const vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(width) * height * STBI_rgb_alpha;
	return uploadTextureData(pixels, width, height, imageSize, vk::Format::eR8G8B8A8Srgb);
}

TextureHandle Renderer::uploadHDRTexture(const float* pixels, int width, int height) {
	const vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(width) * height * STBI_rgb_alpha * sizeof(float);
	return uploadTextureData(pixels, width, height, imageSize, vk::Format::eR32G32B32A32Sfloat);
}

TextureHandle Renderer::uploadTextureData(const void* pixels, int width, int height, vk::DeviceSize imageSize, vk::Format textureFormat) {
	auto [stagingBuffer, stagingBufferMemory] =
		createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

	void* data = stagingBufferMemory.mapMemory(0, imageSize);
	memcpy(data, pixels, imageSize);
	stagingBufferMemory.unmapMemory();

	auto [textureImage, textureImageMemory] = createImage(width,
		 height,
		 textureFormat,
		 vk::ImageTiling::eOptimal,
		 vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
		 vk::MemoryPropertyFlagBits::eDeviceLocal);
	vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();
	transitionImageLayout(commandBuffer, textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
	copyBufferToImage(commandBuffer, stagingBuffer, textureImage, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
	transitionImageLayout(commandBuffer, textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
	endSingleTimeCommands(std::move(commandBuffer));

	vk::raii::ImageView textureImageView = createImageView(*textureImage, textureFormat, vk::ImageAspectFlagBits::eColor);
	vk::raii::Sampler textureSampler = createImageSampler();

	vk::DescriptorSetAllocateInfo allocInfo {
		.descriptorPool = m_descriptorPool,
		.descriptorSetCount = 1,
		.pSetLayouts = &*m_materialDescriptorSetLayout
	};

	auto descriptorSets = m_device.allocateDescriptorSets(allocInfo);

	TextureResources material {
		std::move(textureImage),
		std::move(textureImageMemory),
		std::move(textureImageView),
		std::move(textureSampler),
		std::move(descriptorSets.front())
	};

	vk::DescriptorImageInfo imageInfo {
		.sampler = material.sampler,
		.imageView = material.imageView,
		.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
	};

	vk::WriteDescriptorSet descriptorWrite {
		.dstSet = material.descriptorSet,
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = vk::DescriptorType::eCombinedImageSampler,
		.pImageInfo = &imageInfo
	};

	m_device.updateDescriptorSets(descriptorWrite, {});

	m_textureResources.push_back(std::move(material));
	return m_textureResources.size() - 1;
}

void Renderer::rebuildInstanceBatches() {
	m_instanceBatches.clear();
	m_instanceBatchToIndex.clear();
	m_instanceCount = 0;

	auto view = m_registry.view<Transform, MeshRenderer>();
	for (auto it = view.begin(); it != view.end(); ++it) {
		createMeshEntity(it.entity());
	}
}

void Renderer::createFrameDescriptors() {
	std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *m_descriptorSetLayout);
	vk::DescriptorSetAllocateInfo allocInfo{
		.descriptorPool = m_descriptorPool,
		.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
		.pSetLayouts = layouts.data()
	};
	auto descriptorSets = m_device.allocateDescriptorSets(allocInfo);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		const vk::DeviceSize bufferSize = sizeof(FrameUniformBufferObject);
		auto [buffer, bufferMem] = createBuffer(
			bufferSize,
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
		);
		auto& frame = m_frames[i];
		frame.resources.uniformBuffer = std::move(buffer);
		frame.resources.uniformBufferMemory = std::move(bufferMem);
		frame.resources.uniformBufferMapped = frame.resources.uniformBufferMemory.mapMemory(0, bufferSize);

		frame.resources.descriptorSet = std::move(descriptorSets[i]);

		vk::DescriptorBufferInfo bufferInfo{
			.buffer = frame.resources.uniformBuffer,
			.offset = 0,
			.range = sizeof(FrameUniformBufferObject)
		};

		vk::DescriptorImageInfo shadowImageInfo {
			.sampler = frame.resources.shadow.sampler,
			.imageView = frame.resources.shadow.view,
			.imageLayout = vk::ImageLayout::eDepthReadOnlyOptimal
		};

		std::array<vk::WriteDescriptorSet, 2> descriptorWrites {{
			{
				.dstSet = frame.resources.descriptorSet,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBuffer,
				.pBufferInfo = &bufferInfo,
			},
			{
				.dstSet = frame.resources.descriptorSet,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo = &shadowImageInfo
			}
		}};
		m_device.updateDescriptorSets(descriptorWrites, {});
	}
}

void Renderer::updateFrameResources(const Scene& scene) {
	TransformAccess transforms(m_registry);
	const Transform& cameraTransform = m_registry.get<Transform>(scene.camera);
	const glm::vec3 cameraPosition = transforms.position(scene.camera);
	const glm::vec3 camRotation = transforms.rotation(scene.camera);
	glm::mat4 viewMat = glm::translate(glm::mat4(1.0f), cameraPosition);
	viewMat = glm::rotate(viewMat, camRotation.y, glm::vec3(0, 1, 0));
	viewMat = glm::rotate(viewMat, camRotation.x, glm::vec3(1, 0, 0));
	viewMat = glm::rotate(viewMat, camRotation.z, glm::vec3(0, 0, 1));

	viewMat = glm::inverse(viewMat);

	glm::mat4 proj = glm::perspectiveLH_ZO(
		glm::radians(90.0f),
		m_swapChainExtent.width / static_cast<float>(m_swapChainExtent.height),
		0.1f,
		100.0f
	);
	proj[1][1] *= -1;

	const auto& lightTrans = m_registry.get<Transform>(scene.sun);

	// const glm::vec3 lightDir = glm::normalize(glm::vec3(-3.0f, 3.0f, -3.0f));
	const glm::vec3 lightDir = lightTrans.forward();

	constexpr float cameraShadowDistance = 8.0f;
	const glm::vec3 shadowCenter = cameraPosition + cameraTransform.forward() * cameraShadowDistance;

	constexpr float lightDistance = 30.0f;
	constexpr float shadowExtent = 10.0f;

	const glm::vec3 lightPosition = shadowCenter + lightDir * lightDistance;

	const glm::mat4 lightView = glm::lookAtLH(
		lightPosition,
		shadowCenter,
		glm::vec3(0, 1, 0)
	);

	glm::mat4 lightProjection = glm::orthoLH_ZO(
		-shadowExtent,
		shadowExtent,
		-shadowExtent,
		shadowExtent,
		0.1f,
		lightDistance * 2.0f
	);

	lightProjection[1][1] *= -1.0f;

	const glm::mat4 lightViewProjection = lightProjection * lightView;

	auto& frameResources = currentFrame().resources;
	const FrameUniformBufferObject ubo{
		proj,
		viewMat,
		lightViewProjection,
		glm::vec4(lightDir, 0.0f)
	};
	memcpy(frameResources.uniformBufferMapped, &ubo, sizeof(ubo));

	auto skinView = m_registry.view<SkinMeshRenderer, Transform>();
	size_t skinInstanceCount = 0;
	for (auto it = skinView.begin(); it != skinView.end(); ++it) {
		skinInstanceCount++;
	}

	const size_t totalInstanceCount = m_instanceCount + skinInstanceCount;
	if (totalInstanceCount == 0) {
		return;
	}

	if (frameResources.instanceCapacity < totalInstanceCount) {
		if (frameResources.instanceBufferMemory != nullptr && frameResources.instanceBufferMapped != nullptr) {
			frameResources.instanceBufferMemory.unmapMemory();
			frameResources.instanceBufferMapped = nullptr;
		}

		const vk::DeviceSize bufferSize = sizeof(InstanceData) * totalInstanceCount;
		auto [buffer, bufferMem] = createBuffer(
			bufferSize,
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
		);
		frameResources.instanceBuffer = std::move(buffer);
		frameResources.instanceBufferMemory = std::move(bufferMem);
		frameResources.instanceBufferMapped = frameResources.instanceBufferMemory.mapMemory(0, bufferSize);
		frameResources.instanceCapacity = totalInstanceCount;
	}

	std::vector<uint32_t> instanceWriteOffsets(m_instanceBatches.size(), 0);

	InstanceData* instances = static_cast<InstanceData*>(frameResources.instanceBufferMapped);
	auto view = m_registry.view<Transform, MeshRenderer>();
	for (auto it = view.begin(); it != view.end(); ++it) {
		auto [transform, mesh] = *it;
		const MeshRenderer* meshRend = m_registry.get<MeshRenderer>().tryGet(it.entity());
		// Keep the per-frame batch lookup consistent with createMeshEntity().
		// Untextured entities are assigned the uploaded white default material
		// when their batches are created, so they must use that same handle when
		// writing instance data here.
		const InstanceBatchKey batchKey {
			.meshHandle = meshRend->meshHandle,
			.meshRenderer = meshRend->matHandle
		};

		const auto batchIt = m_instanceBatchToIndex.find(batchKey);
		if (batchIt == m_instanceBatchToIndex.end()) {
			continue;
		}

		const uint32_t batchIndex = batchIt->second;
		InstanceBatch& batch = m_instanceBatches[batchIndex];
		const uint32_t batchWriteIndex = instanceWriteOffsets[batchIndex];
		instanceWriteOffsets[batchIndex]++;
		if (batchWriteIndex >= batch.instanceCount) {
			continue;
		}

		const uint32_t instanceIndex = batch.firstInstance + batchWriteIndex;
		glm::mat4 model = transforms.matrix(it.entity());
		// glm::vec3 eulerAngles = transforms.rotation(it.entity());
		// model = glm::rotate(model, eulerAngles.y, glm::vec3(0.0f, 1.0f, 0.0f));
		// model = glm::rotate(model, eulerAngles.x, glm::vec3(1.0f, 0.0f, 0.0f));
		// model = glm::rotate(model, eulerAngles.z, glm::vec3(0.0f, 0.0f, 1.0f));
		// model = glm::scale(model, transform.scale);

		const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(model)));
		instances[instanceIndex] = {model, normalMatrix};
	}


	uint32_t skinInstanceIndex = static_cast<uint32_t>(m_instanceCount);

	for (auto it = skinView.begin(); it != skinView.end(); ++it) {
		auto [skinMesh, transform] = *it;

		const glm::mat4 model = transforms.matrix(it.entity());
		const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(model)));

		instances[skinInstanceIndex] = {model, normalMatrix};
		skinInstanceIndex++;
	}
}

void Renderer::drawFrame(const Scene& scene) {
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
	updateFrameResources(scene);
	setSkinPalette();
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
		assert(presentResult == vk::Result::eSuccess);
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
	createGraphicsPipelines();
	createSkinPipeline();
	createCommandPool();
	// createTextureImage();
	// createTextureImageView();
	// createTextureSampler();
	createFrameResources();
	createShadowResources();
	createDescriptorPool();
	createFrameDescriptors();
	createCommandBuffer();
	createSyncObjects();

	createSkybox();
	createDefaultTexture();
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
			{.features = {.samplerAnisotropy = true}},
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
		feature.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
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

	vk::SwapchainKHR oldSwapChain = m_swapChain != nullptr ? *m_swapChain : nullptr;
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
		.clipped = true,
		.oldSwapchain = oldSwapChain
	};

	m_swapChain = vk::raii::SwapchainKHR(m_device, swapChainCreateInfo);
	const auto swapChainImages = m_swapChain.getImages();
	m_swapchainData.resize(swapChainImages.size());

	for (size_t i = 0; i < swapChainImages.size(); i++) {
		m_swapchainData[i].image = swapChainImages[i];
	}

	createSwapImageViews();
	createDepthResources();
	for (auto& swapchainData : m_swapchainData) {
		if (swapchainData.renderFinishedSemaphore == nullptr) {
			swapchainData.renderFinishedSemaphore = vk::raii::Semaphore(m_device, vk::SemaphoreCreateInfo());
		}
	}
	m_swapChainInitialized = true;
}

void Renderer::createDescriptorSetLayout() {
	std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {{
		{
			.binding = 0,
			.descriptorType = vk::DescriptorType::eUniformBuffer,
			.descriptorCount = 1,
			.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
		},
		{
			.binding = 1,
			.descriptorType = vk::DescriptorType::eCombinedImageSampler,
			.descriptorCount = 1,
			.stageFlags = vk::ShaderStageFlagBits::eFragment
		}
	}};

	vk::DescriptorSetLayoutCreateInfo layoutInfo {
		.bindingCount = static_cast<uint32_t>(bindings.size()),
		.pBindings = bindings.data(),
	};
	m_descriptorSetLayout = m_device.createDescriptorSetLayout(layoutInfo);

	vk::DescriptorSetLayoutBinding textureBinding {
		.binding = 0,
		.descriptorType = vk::DescriptorType::eCombinedImageSampler,
		.descriptorCount = 1,
		.stageFlags = vk::ShaderStageFlagBits::eFragment
	};

	vk::DescriptorSetLayoutCreateInfo texLayoutInfo {
		.bindingCount = 1,
		.pBindings = &textureBinding
	};

	m_materialDescriptorSetLayout = m_device.createDescriptorSetLayout(texLayoutInfo);
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

void Renderer::createGraphicsPipelines() {
	std::vector<char> shaderCode = readFile("shaders/slang.spv");
	vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

	m_graphicsPipelines.resize(static_cast<size_t>(GraphicsPipelineId::Count));

	std::array bindingDescriptions = {
		Vertex::getBindingDescription(),
		InstanceData::getBindingDescription()
	};
	const auto vertexAttributeDescriptions = Vertex::getAttributeDescriptions();
	const auto instanceAttributeDescriptions = InstanceData::getAttributeDescriptions();
	std::array<vk::VertexInputAttributeDescription, vertexAttributeDescriptions.size() + instanceAttributeDescriptions.size()>
		attributeDescriptions{};
	std::ranges::copy(vertexAttributeDescriptions, attributeDescriptions.begin());
	std::ranges::copy(instanceAttributeDescriptions, attributeDescriptions.begin() + vertexAttributeDescriptions.size());

	std::array<vk::DescriptorSetLayout, 2> descriptorSetLayouts = {
		*m_descriptorSetLayout, *m_materialDescriptorSetLayout
	};

	m_graphicsPipelines[static_cast<size_t>(GraphicsPipelineId::Mesh)] = createGraphicsPipeline(
		shaderModule,
		"vertMain",
		"fragMain",
		bindingDescriptions,
		attributeDescriptions,
		descriptorSetLayouts,
		vk::CullModeFlagBits::eBack,
		vk::True,
		vk::CompareOp::eLess
	);

	std::vector<char> skyboxShaderCode = readFile("shaders/skybox.spv");
	vk::raii::ShaderModule skyboxShaderModule = createShaderModule(skyboxShaderCode);

	std::array skyboxBindingDescriptions = {
		Vertex::getBindingDescription()
	};


	m_graphicsPipelines[static_cast<size_t>(GraphicsPipelineId::Skybox)] = createGraphicsPipeline(
		skyboxShaderModule,
		"vertMain",
		"fragMain",
		skyboxBindingDescriptions,
		vertexAttributeDescriptions,
		descriptorSetLayouts,
		vk::CullModeFlagBits::eNone,
		vk::False,
		vk::CompareOp::eLessOrEqual
	);


	std::vector<char> shadowShaderCode = readFile("shaders/shadow.spv");
	vk::raii::ShaderModule shadowShaderModule = createShaderModule(shadowShaderCode);


	std::array<vk::PipelineShaderStageCreateInfo, 1> shadowStages = {{
		{
			.stage = vk::ShaderStageFlagBits::eVertex,
			.module = shadowShaderModule,
			.pName = "vertMain"
		}
	}};

	std::array shadowSetLayouts {
		*m_descriptorSetLayout
	};

	m_graphicsPipelines[static_cast<size_t>(GraphicsPipelineId::Shadow)] = createGraphicsPipeline(
		shadowStages,
		bindingDescriptions,
		attributeDescriptions,
		shadowSetLayouts,
		vk::Format::eUndefined,
		findShadowFormat(),
		vk::CullModeFlagBits::eBack,
		vk::True,
		vk::CompareOp::eLess,
		vk::True
	);
}

void Renderer::createSkinPipeline() {
	const std::array<vk::DescriptorSetLayoutBinding, 5> bindings {{
		{
			.binding = 0,
			.descriptorType = vk::DescriptorType::eStorageBuffer,
			.descriptorCount = 1,
			.stageFlags = vk::ShaderStageFlagBits::eCompute
		},
		{
			.binding = 1,
			.descriptorType = vk::DescriptorType::eStorageBuffer,
			.descriptorCount = 1,
			.stageFlags = vk::ShaderStageFlagBits::eCompute
		},
		{
			.binding = 2,
			.descriptorType = vk::DescriptorType::eStorageBuffer,
			.descriptorCount = 1,
			.stageFlags = vk::ShaderStageFlagBits::eCompute
		}, 
		{
			.binding = 3,
			.descriptorType = vk::DescriptorType::eStorageBuffer,
			.descriptorCount = 1,
			.stageFlags = vk::ShaderStageFlagBits::eCompute
		}, 
		{
			.binding = 4,
			.descriptorType = vk::DescriptorType::eStorageBuffer,
			.descriptorCount = 1,
			.stageFlags = vk::ShaderStageFlagBits::eCompute
		}
	}};

	const vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutInfo {
		.bindingCount = static_cast<uint32_t>(bindings.size()),
		.pBindings = bindings.data()
	};

	m_skinDescriptorSetLayout = m_device.createDescriptorSetLayout(descriptorSetLayoutInfo);

	const vk::PushConstantRange pushConstantRange {
		.stageFlags = vk::ShaderStageFlagBits::eCompute,
		.offset = 0,
		.size = sizeof(SkinPushConstants)
	};

	const vk::PipelineLayoutCreateInfo pipelineLayoutInfo {
		.setLayoutCount = 1,
		.pSetLayouts = &*m_skinDescriptorSetLayout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange
	};


	m_skinPipeline.layout = m_device.createPipelineLayout(pipelineLayoutInfo);

	// Skin Compute Shader
	std::vector<char> skinShaderCode = readFile("shaders/skin.spv");
	vk::raii::ShaderModule skinShaderModule = createShaderModule(skinShaderCode);

	const vk::PipelineShaderStageCreateInfo skinShaderStage {
		.stage = vk::ShaderStageFlagBits::eCompute,
		.module = skinShaderModule,
		.pName = "main"
	};

	const vk::ComputePipelineCreateInfo pipelineInfo {
		.stage = skinShaderStage,
		.layout = *m_skinPipeline.layout
	};

	m_skinPipeline.pipeline = vk::raii::Pipeline(m_device, nullptr, pipelineInfo);
}

void Renderer::createShadowResources() {
	vk::Format shadowFormat = findShadowFormat();
	for (auto& frame : m_frames) {
		auto [image, memory] = createImage(
			SHADOW_MAP_SIZE,
			SHADOW_MAP_SIZE,
			shadowFormat,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);

		frame.resources.shadow.image = std::move(image);
		frame.resources.shadow.memory = std::move(memory);
		frame.resources.shadow.view = createImageView(*frame.resources.shadow.image, shadowFormat, vk::ImageAspectFlagBits::eDepth);
		frame.resources.shadow.sampler = createShadowSampler();
		frame.resources.shadow.layout = vk::ImageLayout::eUndefined;
	}
}

GraphicsPipelineResources Renderer::createGraphicsPipeline(
	const vk::raii::ShaderModule& shaderModule,
	const char* vertexEntryPoint,
	const char* fragmentEntryPoint,
	std::span<const vk::VertexInputBindingDescription> bindingDescriptions,
	std::span<const vk::VertexInputAttributeDescription> attributeDescriptions,
	std::span<const vk::DescriptorSetLayout> descriptorSetLayouts,
	vk::CullModeFlags cullMode,
	vk::Bool32 depthWriteEnable,
	vk::CompareOp depthCompareOp
) {
	vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
		.stage = vk::ShaderStageFlagBits::eVertex,
		.module = shaderModule,
		.pName = vertexEntryPoint
	};

	vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
		.stage = vk::ShaderStageFlagBits::eFragment,
		.module = shaderModule,
		.pName = fragmentEntryPoint
	};

	std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {vertShaderStageInfo, fragShaderStageInfo};

	vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
		.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size()),
		.pVertexBindingDescriptions = bindingDescriptions.data(),
		.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
		.pVertexAttributeDescriptions = attributeDescriptions.data(),
	};
	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
	vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

	vk::PipelineRasterizationStateCreateInfo rasterizer{
		.depthClampEnable = vk::False,
		.rasterizerDiscardEnable = vk::False,
		.polygonMode = vk::PolygonMode::eFill,
		.cullMode = cullMode,
		.frontFace = vk::FrontFace::eClockwise,
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

	constexpr vk::PushConstantRange materialPushConstantRange {
		.stageFlags = vk::ShaderStageFlagBits::eFragment,
		.offset = 0,
		.size = sizeof(MaterialPushConstants)
	};

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
		.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size()),
		.pSetLayouts = descriptorSetLayouts.data(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &materialPushConstantRange
	};

	GraphicsPipelineResources pipelineResources{
		.layout = vk::raii::PipelineLayout(m_device, pipelineLayoutInfo)
	};

	vk::PipelineDepthStencilStateCreateInfo depthStencil{
		.depthTestEnable = vk::True,
		.depthWriteEnable = depthWriteEnable,
		.depthCompareOp = depthCompareOp,
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
			.layout = pipelineResources.layout
		},
		{
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &m_swapChainSurfaceFormat.format,
			.depthAttachmentFormat = findDepthFormat()
		}
	};

	pipelineResources.pipeline = vk::raii::Pipeline(
		m_device,
		nullptr,
		pipelineCreateInfo.get<vk::GraphicsPipelineCreateInfo>()
	);
	return pipelineResources;
}

GraphicsPipelineResources Renderer::createGraphicsPipeline(
	std::span<const vk::PipelineShaderStageCreateInfo> shaderStages,
	std::span<const vk::VertexInputBindingDescription> bindingDescriptions,
	std::span<const vk::VertexInputAttributeDescription> attributeDescriptions,
	std::span<const vk::DescriptorSetLayout> descriptorSetLayouts,
	vk::Format colorAttachmentFormat,
	vk::Format depthAttachmentFormat,
	vk::CullModeFlags cullMode,
	vk::Bool32 depthWriteEnable,
	vk::CompareOp depthCompareOp,
	vk::Bool32 depthBiasEnable
) {
	vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
		.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size()),
		.pVertexBindingDescriptions = bindingDescriptions.data(),
		.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
		.pVertexAttributeDescriptions = attributeDescriptions.data(),
	};
	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
	vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

	vk::PipelineRasterizationStateCreateInfo rasterizer{
		.depthClampEnable = vk::False,
		.rasterizerDiscardEnable = vk::False,
		.polygonMode = vk::PolygonMode::eFill,
		.cullMode = cullMode,
		.frontFace = vk::FrontFace::eClockwise,
		.depthBiasEnable = depthBiasEnable,
		.lineWidth = 1.0f
	};

	vk::PipelineMultisampleStateCreateInfo multisampling{
		.rasterizationSamples = vk::SampleCountFlagBits::e1,
		.sampleShadingEnable = vk::False
	};

	vk::PipelineColorBlendAttachmentState colorAttachment {
		.blendEnable = vk::False,
		.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
						  vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
	};

	bool enableColorAttachment = colorAttachmentFormat != vk::Format::eUndefined;

	vk::PipelineColorBlendStateCreateInfo colorBlending{
		.logicOpEnable = vk::False,
		.logicOp = vk::LogicOp::eCopy,
		.attachmentCount = enableColorAttachment ? 1u : 0u,
		.pAttachments = enableColorAttachment ? &colorAttachment : nullptr
	};

	std::array<vk::DynamicState, 3> dynamicStates = {
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor,
		vk::DynamicState::eDepthBias
	};
	vk::PipelineDynamicStateCreateInfo dynamicState{
		.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
		.pDynamicStates = dynamicStates.data()
	};

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
		.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size()),
		.pSetLayouts = descriptorSetLayouts.data()
	};
	GraphicsPipelineResources pipelineResources{
		.layout = vk::raii::PipelineLayout(m_device, pipelineLayoutInfo)
	};

	vk::PipelineDepthStencilStateCreateInfo depthStencil{
		.depthTestEnable = vk::True,
		.depthWriteEnable = depthWriteEnable,
		.depthCompareOp = depthCompareOp,
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
			.layout = pipelineResources.layout
		},
		{
			.colorAttachmentCount = enableColorAttachment ? 1u : 0,
			.pColorAttachmentFormats = enableColorAttachment ? &colorAttachmentFormat : nullptr,
			.depthAttachmentFormat = depthAttachmentFormat 
		}
	};

	pipelineResources.pipeline = vk::raii::Pipeline(
		m_device,
		nullptr,
		pipelineCreateInfo.get<vk::GraphicsPipelineCreateInfo>()
	);
	return pipelineResources;
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
	std::array<vk::DescriptorPoolSize, 3> poolSizes = {{
		{
			.type = vk::DescriptorType::eUniformBuffer,
			.descriptorCount = MAX_ENTITY_COUNT
		},
		{
			.type = vk::DescriptorType::eCombinedImageSampler,
			.descriptorCount = MAX_ENTITY_COUNT
		},
		{
			.type = vk::DescriptorType::eStorageBuffer,
			.descriptorCount = MAX_ENTITY_COUNT * MAX_FRAMES_IN_FLIGHT
		}
	}};

	vk::DescriptorPoolCreateInfo poolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = 4096,
		.poolSizeCount = poolSizes.size(),
		.pPoolSizes = poolSizes.data()
	};

	m_descriptorPool = vk::raii::DescriptorPool(m_device, poolInfo);
}

void Renderer::createTextureImage() {
	int texWidth, texHeight, texChannels;
	stbi_uc* pixels = stbi_load("../textures/viking_room.png", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

	if (!pixels) {
		throw std::runtime_error("failed to load texture image!");
	}

	vk::DeviceSize imageSize = texWidth * texHeight * 4;
	auto [stagingBuffer, stagingBufferMemory] =
		createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

	void* data = stagingBufferMemory.mapMemory(0, imageSize);
	memcpy(data, pixels, imageSize);
	stagingBufferMemory.unmapMemory();

	stbi_image_free(pixels);

	std::tie(m_textureImage, m_textureImageMemory) = createImage(texWidth,
		 texHeight,
		 vk::Format::eR8G8B8A8Srgb,
		 vk::ImageTiling::eOptimal,
		 vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
		 vk::MemoryPropertyFlagBits::eDeviceLocal);
	vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();
	transitionImageLayout(commandBuffer, m_textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
	copyBufferToImage(commandBuffer, stagingBuffer, m_textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
	transitionImageLayout(commandBuffer, m_textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
	endSingleTimeCommands(std::move(commandBuffer));
}


void Renderer::createTextureImageView() {
	m_textureImageView = createImageView(*m_textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor);
}

void Renderer::createTextureSampler() {
	vk::PhysicalDeviceProperties properties = m_physicalDevice.getProperties();
	vk::SamplerCreateInfo samplerInfo {
		.magFilter = vk::Filter::eLinear,
		.minFilter = vk::Filter::eLinear,
		.mipmapMode = vk::SamplerMipmapMode::eLinear,
		.addressModeU = vk::SamplerAddressMode::eRepeat,
		.addressModeV = vk::SamplerAddressMode::eRepeat,
		.addressModeW = vk::SamplerAddressMode::eRepeat,
		.mipLodBias = 0.0f,
		.anisotropyEnable = vk::True,
		.maxAnisotropy = properties.limits.maxSamplerAnisotropy,
		.compareEnable = vk::False,
		.compareOp = vk::CompareOp::eAlways,
		.minLod = 0.0f,
		.maxLod = 0.0f,
	};
	m_textureSampler = vk::raii::Sampler(m_device, samplerInfo);
}


vk::raii::Sampler Renderer::createImageSampler() {
	vk::PhysicalDeviceProperties properties = m_physicalDevice.getProperties();
	vk::SamplerCreateInfo samplerInfo {
		.magFilter = vk::Filter::eLinear,
		.minFilter = vk::Filter::eLinear,
		.mipmapMode = vk::SamplerMipmapMode::eLinear,
		.addressModeU = vk::SamplerAddressMode::eRepeat,
		.addressModeV = vk::SamplerAddressMode::eRepeat,
		.addressModeW = vk::SamplerAddressMode::eRepeat,
		.mipLodBias = 0.0f,
		.anisotropyEnable = vk::True,
		.maxAnisotropy = properties.limits.maxSamplerAnisotropy,
		.compareEnable = vk::False,
		.compareOp = vk::CompareOp::eAlways,
		.minLod = 0.0f,
		.maxLod = 0.0f,
	};
	return vk::raii::Sampler(m_device, samplerInfo);
}

vk::raii::Sampler Renderer::createShadowSampler() {
	vk::SamplerCreateInfo samplerInfo {
		.magFilter = vk::Filter::eNearest,
		.minFilter = vk::Filter::eNearest,
		.mipmapMode = vk::SamplerMipmapMode::eNearest,
		.addressModeU = vk::SamplerAddressMode::eClampToBorder,
		.addressModeV = vk::SamplerAddressMode::eClampToBorder,
		.addressModeW = vk::SamplerAddressMode::eClampToBorder,
		.mipLodBias = 0.0f,
		.anisotropyEnable = vk::False,
		.maxAnisotropy = 1.0f,
		.compareEnable = vk::False,
		.compareOp = vk::CompareOp::eAlways,
		.minLod = 0.0f,
		.maxLod = 0.0f,
		.borderColor = vk::BorderColor::eFloatOpaqueWhite,
		.unnormalizedCoordinates = vk::False
	};

	return vk::raii::Sampler(m_device, samplerInfo);
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
		if (swapchainData.renderFinishedSemaphore == nullptr) {
			swapchainData.renderFinishedSemaphore = vk::raii::Semaphore(m_device, vk::SemaphoreCreateInfo());
		}
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

	auto& frameResources = currentFrame().resources;

	recordSkinningDispatches(commandBuffer, frameResources.skinInstances);

	recordShadowPass(commandBuffer, frameResources);

	transition_image_layout(
		swapchainData.image,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eColorAttachmentOptimal,
		{},
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::ImageAspectFlagBits::eColor
	);
	transition_image_layout(
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
	const auto& meshPipeline = m_graphicsPipelines[static_cast<size_t>(GraphicsPipelineId::Mesh)];
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *meshPipeline.pipeline);
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

	commandBuffer.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		meshPipeline.layout,
		0,
		*frameResources.descriptorSet,
		nullptr
	);

	for (const auto& batch : m_instanceBatches) {
		const auto& meshResource = m_meshResources[batch.meshHandle];
		const auto& material = m_resource.getMaterial(batch.matHandle);
		const auto& textureResource =  material.textureHandle != INVALID_TEXTURE ? 
			m_textureResources.at(material.textureHandle) : m_textureResources.at(m_defaultTextureHandle); 

		commandBuffer.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			meshPipeline.layout,
			1,
			*textureResource.descriptorSet,
			nullptr
		);

		const MaterialPushConstants pushConstants {
			.baseColor = material.baseColor
		};

		commandBuffer.pushConstants(
			meshPipeline.layout,
			vk::ShaderStageFlagBits::eFragment,
			0,
			sizeof(pushConstants),
			&pushConstants
		);

		std::array vertexBuffers = {*meshResource.vertexBuffer, *frameResources.instanceBuffer};
		std::array<vk::DeviceSize, 2> offsets = {0, 0};
		commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
		commandBuffer.bindIndexBuffer(*meshResource.indexBuffer, 0, vk::IndexType::eUint16);
		commandBuffer.drawIndexed(meshResource.indiceSize, batch.instanceCount, 0, 0, batch.firstInstance);
	}

	recordSkinMeshPass(commandBuffer, frameResources);

	// Skybox rendering

	const auto& skyboxPipeline = m_graphicsPipelines[static_cast<size_t>(GraphicsPipelineId::Skybox)];
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *skyboxPipeline.pipeline);

	const auto& skyboxMeshResource = m_meshResources.at(m_skyboxMeshHandle);
	const auto& skyboxMatResource = m_textureResources.at(m_skyboxMaterialHandle);

	commandBuffer.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		skyboxPipeline.layout,
		0,
		*frameResources.descriptorSet,
		nullptr
	);

	commandBuffer.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		skyboxPipeline.layout,
		1,
		*skyboxMatResource.descriptorSet,
		nullptr
	);

	commandBuffer.bindVertexBuffers(0, *skyboxMeshResource.vertexBuffer, {0});
	commandBuffer.bindIndexBuffer(*skyboxMeshResource.indexBuffer, 0, vk::IndexType::eUint16);
	commandBuffer.drawIndexed(skyboxMeshResource.indiceSize, 1, 0, 0, 0);

	commandBuffer.endRendering();

	transition_image_layout(
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

void Renderer::transition_image_layout(
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

void Renderer::transitionImageLayout(vk::raii::CommandBuffer& commandBuffer, const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
	vk::ImageMemoryBarrier barrier {
		.oldLayout = oldLayout,
		.newLayout = newLayout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = vk::ImageAspectFlagBits::eColor,
			.levelCount = 1,
			.layerCount = 1
		}
	};

	vk::PipelineStageFlags sourceStage;
	vk::PipelineStageFlags destStage;

	if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
		barrier.srcAccessMask = {};
		barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

		sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
		destStage = vk::PipelineStageFlagBits::eTransfer;
	} else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

		sourceStage = vk::PipelineStageFlagBits::eTransfer;
		destStage = vk::PipelineStageFlagBits::eFragmentShader;
	} else {
		throw std::invalid_argument("unsupported layout transition!");
	}

	commandBuffer.pipelineBarrier(sourceStage, destStage, {}, nullptr, nullptr, barrier);
}

void Renderer::recordSkinningDispatches(const vk::raii::CommandBuffer& commandBuffer, std::span<const SkinFrameResource> frameSkins) {
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *m_skinPipeline.pipeline);

	auto view = m_registry.get<SkinMeshRenderer>();

	for (int i = 0; i < view.data.size(); i++) {
		const SkinMeshRenderer& skinMeshRenderer = view.getByIndex(i);

		const SkinMeshResources& mesh = m_skinMeshes[skinMeshRenderer.skinMeshHandle];

		const SkinFrameResource& frame = frameSkins[skinMeshRenderer.skinResourceHandle];

		commandBuffer.bindDescriptorSets(
			vk::PipelineBindPoint::eCompute,
			*m_skinPipeline.layout,
			0,
			*frame.descriptorSet,
			nullptr
		);


		SkinPushConstants constants {
			.vertexCount = mesh.vertexCount
		};

		commandBuffer.pushConstants(
			*m_skinPipeline.layout,
			vk::ShaderStageFlagBits::eCompute,
			0,
			sizeof(constants),
			&constants
		);

		const uint32_t groupCount = (mesh.vertexCount + 63) / 64;

		commandBuffer.dispatch(groupCount, 1, 1);
	}

	const vk::MemoryBarrier2 barrier {
		.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
		.srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
		.dstStageMask = vk::PipelineStageFlagBits2::eVertexInput,
		.dstAccessMask = vk::AccessFlagBits2::eVertexAttributeRead
	};

	const vk::DependencyInfo dependencyInfo {
		.memoryBarrierCount = 1,
		.pMemoryBarriers = &barrier
	};

	commandBuffer.pipelineBarrier2(dependencyInfo);
}

void Renderer::recordShadowPass(const vk::raii::CommandBuffer& commandBuffer, FrameResources& frameResources) {

	ShadowResources& shadow = frameResources.shadow;

	vk::AccessFlags2 shadowSourceAccess = shadow.layout == vk::ImageLayout::eUndefined ?
		vk::AccessFlagBits2::eNone : vk::AccessFlagBits2::eShaderSampledRead;
	vk::PipelineStageFlags2 shadowSourceStage = shadow.layout == vk::ImageLayout::eUndefined ?
		vk::PipelineStageFlagBits2::eTopOfPipe : vk::PipelineStageFlagBits2::eFragmentShader;

	
	transition_image_layout(
		*shadow.image,
		shadow.layout,
		vk::ImageLayout::eDepthAttachmentOptimal,
		shadowSourceAccess,
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		shadowSourceStage,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::ImageAspectFlagBits::eDepth
	);

	const vk::ClearValue shadowClear = vk::ClearDepthStencilValue(1.0f, 0.0f);

	vk::RenderingAttachmentInfo shadowDepthAttachment = {
		.imageView = shadow.view,
		.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = shadowClear
	};

	vk::RenderingInfo shadowRenderingInfo{
		.renderArea = {
			.offset = {0,0},
			.extent = {
				SHADOW_MAP_SIZE,
				SHADOW_MAP_SIZE
			}
		},
		.layerCount = 1,
		.colorAttachmentCount = 0,
		.pColorAttachments = nullptr,
		.pDepthAttachment = &shadowDepthAttachment
	};

	commandBuffer.beginRendering(shadowRenderingInfo);
	const auto& shadowPipeline = m_graphicsPipelines[static_cast<size_t>(GraphicsPipelineId::Shadow)];

	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *shadowPipeline.pipeline);
	commandBuffer.setViewport(0, vk::Viewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 0, 1));
	commandBuffer.setScissor(
		0,
		vk::Rect2D(
			{0, 0},
			{SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}
		)
	);
	commandBuffer.setDepthBias(1.25f, 0.0f, 1.75f);

	commandBuffer.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		*shadowPipeline.layout,
		0,
		*frameResources.descriptorSet,
		nullptr
	);

	for(const auto& batch : m_instanceBatches) {
		const auto& meshResource = m_meshResources[batch.meshHandle];

		std::array vertexBuffers = {*meshResource.vertexBuffer, *frameResources.instanceBuffer};

		commandBuffer.bindVertexBuffers(0, vertexBuffers, {0,0});

		commandBuffer.bindIndexBuffer(meshResource.indexBuffer, 0, vk::IndexType::eUint16);

		commandBuffer.drawIndexed(meshResource.indiceSize, batch.instanceCount, 0, 0, batch.firstInstance);
	}

	size_t firstInstance = m_instanceCount;
	for (auto [skinMeshRenderer, Transform] : m_registry.view<SkinMeshRenderer, Transform>()) {
		const SkinMeshResources& mesh = m_skinMeshes[skinMeshRenderer.skinMeshHandle];
		const SkinFrameResource& skinFrame = frameResources.skinInstances[skinMeshRenderer.skinResourceHandle];

		std::array vertexBuffers = {*skinFrame.outputVertexBuffer, *frameResources.instanceBuffer};

		commandBuffer.bindVertexBuffers(
			0,
			vertexBuffers,
			std::array<vk::DeviceSize, 2>{0,0}
		);
		commandBuffer.bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexType::eUint16);

		commandBuffer.drawIndexed(mesh.indexCount, 1, 0, 0, firstInstance);
		firstInstance++;
	}

	commandBuffer.endRendering();

	transition_image_layout(
		*shadow.image,
		vk::ImageLayout::eDepthAttachmentOptimal,
		vk::ImageLayout::eDepthReadOnlyOptimal,
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::AccessFlagBits2::eShaderSampledRead,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::PipelineStageFlagBits2::eFragmentShader,
		vk::ImageAspectFlagBits::eDepth
	);

	shadow.layout = vk::ImageLayout::eDepthReadOnlyOptimal;
}

void Renderer::recordSkinMeshPass(const vk::raii::CommandBuffer& commandBuffer, FrameResources& frameResources) {
	const auto& meshPipeline = m_graphicsPipelines[static_cast<size_t>(GraphicsPipelineId::Mesh)];

	auto skinView = m_registry.view<SkinMeshRenderer, Transform>();

	uint32_t firstInstance = static_cast<uint32_t>(m_instanceCount);
	for (auto [skinMeshRenderer, Transform] : skinView) {
		const SkinMeshResources& mesh = m_skinMeshes[skinMeshRenderer.skinMeshHandle];
		const SkinFrameResource& skinFrame = frameResources.skinInstances[skinMeshRenderer.skinResourceHandle];

		const auto& material = m_resource.getMaterial(skinMeshRenderer.matHandle);
		const auto& textureResource =  material.textureHandle != INVALID_TEXTURE ? 
			m_textureResources.at(material.textureHandle) : m_textureResources.at(m_defaultTextureHandle); 

		commandBuffer.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			meshPipeline.layout,
			1,
			*textureResource.descriptorSet,
			nullptr
		);

		const MaterialPushConstants pushConstants {
			.baseColor = material.baseColor
		};

		commandBuffer.pushConstants(
			meshPipeline.layout,
			vk::ShaderStageFlagBits::eFragment,
			0,
			sizeof(pushConstants),
			&pushConstants
		);

		std::array vertexBuffers = {*skinFrame.outputVertexBuffer, *frameResources.instanceBuffer};

		commandBuffer.bindVertexBuffers(
			0,
			vertexBuffers,
			std::array<vk::DeviceSize, 2>{0,0}
		);
		commandBuffer.bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexType::eUint16);

		commandBuffer.drawIndexed(mesh.indexCount, 1, 0, 0, firstInstance);
		firstInstance++;
	}
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

void Renderer::copyBufferToImage(vk::raii::CommandBuffer& commandBuffer, const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height) {
	vk::BufferImageCopy region {
		.bufferOffset = 0,
		.bufferRowLength = 0,
		.bufferImageHeight = 0,
		.imageSubresource = {
			.aspectMask = vk::ImageAspectFlagBits::eColor,
			.mipLevel = 0,
			.baseArrayLayer = 0,
			.layerCount = 1
		},
		.imageOffset = {0, 0, 0},
		.imageExtent = {width, height, 1}
	};

	commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
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

vk::Format Renderer::findShadowFormat() {
	return findSupportedFormat(
		std::array{vk::Format::eD32Sfloat, vk::Format::eD16Unorm},
		vk::ImageTiling::eOptimal,
		vk::FormatFeatureFlagBits::eDepthStencilAttachment | vk::FormatFeatureFlagBits::eSampledImage
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
		throw std::runtime_error("Failed to open file! " + filename);
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

void Renderer::createSkybox() {
	Model skybox = Primitive::createSphere(1.0f, 16, 32, true);
	m_skyboxMeshHandle = uploadMesh(skybox);

	HDRImageTexture image = ResourceUtils::loadHDRTexture("golden_gate_hills_1k.hdr");
	m_skyboxMaterialHandle = uploadHDRTexture(image.pixels, image.width, image.height);
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

void Renderer::createDefaultTexture() {
	const stbi_uc whitePixel[] = {255, 255, 255, 255};
	m_defaultTextureHandle = uploadTexture(whitePixel, 1, 1);
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

vk::raii::CommandBuffer Renderer::beginSingleTimeCommands() {
	vk::CommandBufferAllocateInfo allocInfo {.commandPool = m_commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
	vk::raii::CommandBuffer commandBuffer = std::move(vk::raii::CommandBuffers(m_device, allocInfo).front());

	vk::CommandBufferBeginInfo beginInfo {
		.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
	};
	commandBuffer.begin(beginInfo);

	return commandBuffer;
}

void Renderer::endSingleTimeCommands(vk::raii::CommandBuffer&& commandBuffer) {
	commandBuffer.end();

	vk::SubmitInfo submitInfo {
		.commandBufferCount = 1,
		.pCommandBuffers = &*commandBuffer
	};
	m_graphicsQueue.submit(submitInfo, nullptr);
	m_graphicsQueue.waitIdle();
}

void Renderer::cleanup() {
	if (m_device != nullptr) {
		m_device.waitIdle();
	}

	cleanupSwapChain();
}
