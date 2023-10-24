#define GLFW_INCLUDE_VULKAN


#include <GLFW/glfw3.h>
#include<array>
#include <chrono>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include<glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include<glm/gtx/hash.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include<fstream>
#include<algorithm>
#include<iostream>
#include<set>
#include<stdexcept>
#include<functional>
#include<unordered_map>
#include<cstdlib>
#include<vector>

const int WIDTH = 800;
const int HEIGHT = 600;

const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";
//���д���֡��
const int MAX_FRAMES_IN_FLIGHT = 2;
//ָ��У�������
const std::vector<const char*>validationLayers = { "VK_LAYER_KHRONOS_validation" };
//�Ƿ���У���
bool enableValidationLayers = true;
struct Vertex {
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;
	static VkVertexInputBindingDescription getBindingDescription()
	{
		VkVertexInputBindingDescription bindingDescription = {};
		bindingDescription.binding = 0;
		//����
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;
	}
	static std::array<VkVertexInputAttributeDescription, 3>getAttributeDescriptions()
	{
		std::array<VkVertexInputAttributeDescription, 3>attributeDescriptions;
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

		return attributeDescriptions;
	}
	bool operator==(const Vertex& other) const {
		return pos == other.pos && color == other.color && texCoord == other.texCoord;
	}
};
namespace std {
	template<> struct hash<Vertex> {
		size_t operator()(Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}
struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR>formats;
	std::vector<VkPresentModeKHR>presentModes;
};

struct QueueFamilyIndices {
	uint32_t graphicsFamily=-1;//֧�ֻ���ָ��Ķ�����
	uint32_t presentFamily = -1;//֧�ֱ��ֵĶ�����

	bool isComplete() {
		return graphicsFamily>=0&&presentFamily>=0;
	}
};
struct UniformBufferObject {
	//��ǰ���һ������С��16�ֽڵ�����Ҳû��ϵ����Ϊ�������������offset�ᱻ���뵽16�ֽڵı߽�
	alignas(16) glm::mat4 model;//�����뵽16�ֽڵı߽磬��Ϊvulkan��offsetһ��Ҫ��Ϊ16�ֽڵ������� 
	alignas(16) glm::mat4 view;//offsetΪ64
	alignas(16) glm::mat4 proj;//offsetΪ128
}ubo;
class HelloTriangleApplication {
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}
private:

	GLFWwindow* window;//GLFW����ָ��
	VkInstance instance;//vulkanʵ��
	VkSurfaceKHR surface;
	//�����߼��豸ʱָ���Ķ��л������߼��豸һͬ��������������߼��豸ʱ����ӵ�еĶ���Ҳ��
	// �Զ�����������Բ�����cleanup���������
	VkQueue graphicsQueue;
	VkQueue presentQueue;

	VkPhysicalDevice physicalDevice= VK_NULL_HANDLE;//�����豸
	VkDevice device;//�����豸

	VkSwapchainKHR swapChain;
	//������ͼ���ɽ������Լ��������Զ����������Ҫ�����ֶ����������
	//�κ�VkImage����Ҫ��һ��VkImageView����������
	std::vector<VkImage>swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	//VkImageView������ͼ����ʵķ�ʽ���Լ�ͼ�����һ���ֿ��Ա�����
	std::vector<VkImageView>swapChainImageViews;

	std::vector<VkFramebuffer>swapChainFramebuffers;
	VkRenderPass renderPass;
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;
	VkCommandPool commandPool;
	std::vector<VkCommandBuffer>commandBuffers;
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;

	std::vector<VkFence>inFlightFences;
	//��־֡�����С�Ƿ�ı�
	bool framebufferResized = false;
	size_t currentFrame = 0;

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;

	std::vector<VkBuffer>uniformBuffers;
	std::vector<VkDeviceMemory>uniformBuffersMemory;
	std::vector<void*>uniformBuffersMapped;
	
	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;
	uint32_t mipLevels;
	VkImage textureImage;
	VkDeviceMemory textureImageMemory;

	VkImageView textureImageView;
	VkSampler textureSampler;

	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;
	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
	
	//���ɶ��ز����õ���������
	VkImage colorImage;
	VkDeviceMemory colorImageMemory;
	VkImageView colorImageView;

	VkDebugUtilsMessengerEXT debugMessenger;
	void initWindow()
	{
		glfwInit();
		//��ʾ������GLFW��ֹ���Զ�����openGL������
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		//ǰ���������ֱ�Ϊ��ȣ��߶ȣ����⣬
		//���ĸ�����ָ�����ĸ���ʾ������ʾ���������openGL��أ������vulkanû��
		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow* window,int width,int height)
	{
		auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}
	void initVulkan()
	{
		createinstance();
		setupDebugMessenger();
		//���ڴ��ڱ�����������豸��ѡ����һ��Ӱ�죬����ֻ����vulkanʵ������֮��
		createSurface();
		//��������豸
		pickPhysicalDevice();
		//�߼��豸����ֱ����vulkanʵ�����������Դ����߼��豸ʱ����Ҫʹ��vulkanʵ����Ϊ����
		createLogicalDevice();
		//����������
		createSwapChain();
		createImageViews();
		//��Ⱦ����,��Ҫָ��ʹ�õ���ɫ����Ȼ��壬�Լ�����������Ⱦ������δ����������
		createRenderPass();
		createDescriptorSetLayout();
		//������ǰ�������й��ߣ��������ܸ�������������Ż��ռ�
		createGraphicsPipeline();


		//ָ���
		createCommandPool();
		//����Ҫ�õ�ָ�������Ӧ����ָ��ش���֮��
		createColorResources();
		createDepthResources();
		//�����������
		createFramebuffers();
		createTextureImage();
		createTextureImageView();
		//�����������
		createTextureSampler();

		loadModel();
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		//ָ���
		createCommandBuffers();
		createSyncObjects();
	}
	void createColorResources()
	{
		VkFormat colorFormat = swapChainImageFormat;
		createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
			, colorImage, colorImageMemory);
		colorImageView = createImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
	}
	VkSampleCountFlagBits getMaxUsableSampleCount()
	{
		VkPhysicalDeviceProperties physicalDeviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
		VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts &
			physicalDeviceProperties.limits.framebufferDepthSampleCounts;
		if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
		if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
		if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
		if (counts & VK_SAMPLE_COUNT_8_BIT)  { return VK_SAMPLE_COUNT_8_BIT;  }
		if (counts & VK_SAMPLE_COUNT_4_BIT)  { return VK_SAMPLE_COUNT_4_BIT;  }
		if (counts & VK_SAMPLE_COUNT_2_BIT)  { return VK_SAMPLE_COUNT_2_BIT;  }

		return VK_SAMPLE_COUNT_1_BIT;
	}
	void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
		// Check if image format supports linear blitting
		VkFormatProperties formatProperties;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);

		if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
			throw std::runtime_error("texture image format does not support linear blitting!");
		}

		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.image = image;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.subresourceRange.levelCount = 1;

		int32_t mipWidth = texWidth;
		int32_t mipHeight = texHeight;

		for (uint32_t i = 1; i < mipLevels; i++) {
			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			VkImageBlit blit{};
			blit.srcOffsets[0] = { 0, 0, 0 };
			blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
			blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.srcSubresource.mipLevel = i - 1;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 1;
			blit.dstOffsets[0] = { 0, 0, 0 };
			blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
			blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.dstSubresource.mipLevel = i;
			blit.dstSubresource.baseArrayLayer = 0;
			blit.dstSubresource.layerCount = 1;

			vkCmdBlitImage(commandBuffer,
				image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &blit,
				VK_FILTER_LINEAR);

			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			if (mipWidth > 1) mipWidth /= 2;
			if (mipHeight > 1) mipHeight /= 2;
		}

		barrier.subresourceRange.baseMipLevel = mipLevels - 1;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		endSingleTimeCommands(commandBuffer);
	}
	void loadModel()
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t>shapes;
		std::vector<tinyobj::material_t>materials;
		std::string warn, err;
		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str()))
		{
			throw std::runtime_error(warn + err);
		}
		std::unordered_map<Vertex, uint32_t>uniqueVertices{};
		for (const auto& shape : shapes)
		{
			for (const auto& index : shape.mesh.indices)
			{
				Vertex vertex{};
				vertex.pos = { attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2]};
				vertex.texCoord = { attrib.texcoords[2 * index.texcoord_index + 0],
					1.0-attrib.texcoords[2 * index.texcoord_index + 1] };
				vertex.color = { 1.0,1.0,1.0 };
				if (uniqueVertices.count(vertex) == 0)
				{
					uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
					vertices.push_back(vertex);
				}
				indices.push_back(uniqueVertices[vertex]);
			}
		}
	}
	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
	{
		for (VkFormat format : candidates)
		{
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
				return format;
			}
		}

		throw std::runtime_error("failed to find supported format!");
	}
	VkFormat findDepthFormat()
	{
		return findSupportedFormat({ VK_FORMAT_D32_SFLOAT,VK_FORMAT_D32_SFLOAT_S8_UINT,VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
	}
	bool hasStencilComponent(VkFormat format)
	{
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}
	void createDepthResources()
	{
		VkFormat depthFormat = findDepthFormat();
		createImage(swapChainExtent.width, swapChainExtent.height, 1,msaaSamples,depthFormat, VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
		depthImageView = createImageView(depthImage, depthFormat,VK_IMAGE_ASPECT_DEPTH_BIT,1);
	}
	void createTextureSampler()
	{
		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);
		VkSamplerCreateInfo samplerInfo = {};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
		//clamp to border addressing mode�£��᷵�غ�ɫ
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		//��ΪVK_FALSE�Ļ�����normalized�ڡ�0��1���������ǡ�0��heigth����
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = static_cast<float>(mipLevels);
		if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create texture sampler");
		}
	}
	void createTextureImageView()
	{
		textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB,VK_IMAGE_ASPECT_COLOR_BIT,mipLevels);
	}
	VkImageView createImageView(VkImage image, VkFormat format,VkImageAspectFlags aspectFlags,uint32_t mipLevels) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = aspectFlags;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = mipLevels;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView imageView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}

		return imageView;
	}

	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout,uint32_t mipLevels) 
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();
		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = mipLevels;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;
		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) 
		{
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else
		{
			throw std::invalid_argument("unsupported layout transition!");
		}
		vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
		endSingleTimeCommands(commandBuffer);
	}
	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();
		VkBufferImageCopy region= {};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = { 0,0,0 };
		region.imageExtent = { height,width,1 };
		vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		endSingleTimeCommands(commandBuffer);
	}
	VkCommandBuffer beginSingleTimeCommands()
	{
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;
		
		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo= {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		
		vkBeginCommandBuffer(commandBuffer, &beginInfo);
		return commandBuffer;
	}
	void endSingleTimeCommands(VkCommandBuffer commandbuffer)
	{
		vkEndCommandBuffer(commandbuffer);
		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandbuffer;
		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);
		vkFreeCommandBuffers(device, commandPool, 1, &commandbuffer);
	}
	
	void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples,VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
	{
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.mipLevels = mipLevels;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = numSamples;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create image");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);
		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate image memory");
		}
		vkBindImageMemory(device, image, imageMemory, 0);
	}
	void createTextureImage()
	{
		int texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		VkDeviceSize imageSize = texWidth * texHeight * 4;
		mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;
		if (!pixels)
		{
			throw std::runtime_error("faile to load texture image");
		}
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
			| VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
		void* data;
		//Դ���ݴ浽�ڴ�������ַ
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);
		//����Դ����
		stbi_image_free(pixels);

		createImage(texWidth, texHeight, mipLevels,VK_SAMPLE_COUNT_1_BIT,VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT|
			VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			textureImage, textureImageMemory);
		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,mipLevels);
		copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
		//transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mipLevels);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
		generateMipmaps(textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);
	}
	void createDescriptorSets()
	{
		std::vector<VkDescriptorSetLayout>layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();

		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		//����Ҫ�Լ�����descriptorsets����Ϊ��descriptorPool�����ʱ�����Ҳ���Զ����
		if (vkAllocateDescriptorSets(device,&allocInfo,descriptorSets.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create descriptor sets!");
		}
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			VkDescriptorBufferInfo bufferInfo = {};
			bufferInfo.buffer = uniformBuffers[i];
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(UniformBufferObject);

			VkDescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView = textureImageView;
			imageInfo.sampler = textureSampler;
			
			std::array<VkWriteDescriptorSet,2>descriptorWrites{};
			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = descriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &bufferInfo;

			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = descriptorSets[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pImageInfo = &imageInfo;
			vkUpdateDescriptorSets(device,static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}
	void createDescriptorPool()
	{
		std::array<VkDescriptorPoolSize,2> poolSizes{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);


		VkDescriptorPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create descriptor pool");
		}
	}
	void createDescriptorSetLayout()
	{

		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding samplerLayoutBinding{};
		samplerLayoutBinding.binding = 1;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		std::array<VkDescriptorSetLayoutBinding, 2>bindings = { uboLayoutBinding,samplerLayoutBinding };

		VkDescriptorSetLayoutCreateInfo layoutInfo = {};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();
		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("faile to create descriptor set layout!");
	
		}
	}
	void createUniformBuffers()
	{
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);
		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
		//��������Ƶ�����£�����ʹ���ݴ滺��stagingbuffer��������������½�
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i],
				uniformBuffersMemory[i]);
			
			vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize,
					0, &uniformBuffersMapped[i]);
		}
	}
	void createSurface()
	{
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create window surface!");
		}
	}
	void createBuffer(
		VkDeviceSize size, VkBufferUsageFlags usage,
		VkMemoryPropertyFlags properties, VkBuffer& buffer,
		VkDeviceMemory& bufferMemory)
	{
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate buffer memory!");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}
	void createIndexBuffer()
	{
		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
		//staging buffer in CPU accessible memory to upload the data from the vertex array to

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer,
			stagingBufferMemory);

		//����Ҫ�Ѷ������ݿ�����buffer��,ͨ��ӳ�仺������cpu�ɷ��ʵ��ڴ�����ʵ��
		//data���ڴ���ָ��
		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer,
			indexBufferMemory);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}
	void createVertexBuffer()
	{
		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
		//staging buffer in CPU accessible memory to upload the data from the vertex array to

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer,
			stagingBufferMemory);

		//����Ҫ�Ѷ������ݿ�����buffer��,ͨ��ӳ�仺������cpu�ɷ��ʵ��ڴ�����ʵ��
		//data���ڴ���ָ��
		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0,&data);
		memcpy(data, vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer,
			vertexBufferMemory);

		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}
	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();
			VkBufferCopy copyRegion{};
			copyRegion.size = size;
			vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
		endSingleTimeCommands(commandBuffer);

	}
	//The chrono standard library header exposes functions to do precise timekeeping.
	//We��ll use this to make sure that the geometry rotates 90 degrees per second regardless of frame rate.
	void updateUniformBuffer(uint32_t currentImage) 
	{
		static auto startTime =std::chrono::high_resolution_clock::now();
		auto currentTime = std::chrono::high_resolution_clock::now();
		
		float time = std::chrono::duration<float,
			std::chrono::seconds::period>(currentTime -
				startTime).count();

		UniformBufferObject ubo{};
		//��ת�ľ�����ת�Ƕȣ���ת��
		ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f,1.0f));
		//�۾�λ�ã����ĵ㣬���ϵ���
		ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f),glm::vec3(0.0f,0.0f, 0.0f),glm::vec3(0.0f, 0.0f, 1.0f));
		//��ֱ��fov,aspectRatio,��ƽ�棬Զƽ��
		ubo.proj = glm::perspective(glm::radians(45.0f),
			swapChainExtent.width / (float)swapChainExtent.height, 0.1f , 10.0f);
		//GLM was originally designed for OpenGL, where the Y coordinate of the clip coordinates is inverted.
		ubo.proj[1][1] *= -1;
		//dest,source,size
		memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
	}
	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
	{
		//��������豸���Ի�õ��ڴ�����
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
		{
			if ((typeFilter & (1 << i))&&((memProperties.memoryTypes[i].propertyFlags&properties)==properties))
			{
				return i;
			}
		}
		throw std::runtime_error("failed to find suitable memory type!");

	}
	void pickPhysicalDevice()
	{
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
		if (deviceCount == 0)
		{
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}
		std::vector<VkPhysicalDevice>devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
		for (const auto& device : devices)
		{
			if (isDeviceSuitable(device))
			{
				physicalDevice = device;
				msaaSamples = getMaxUsableSampleCount();
				break;
			}
		}
		if (physicalDevice == VK_NULL_HANDLE)
		{
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}
	void createImageViews()
	{
		swapChainImageViews.resize(swapChainImages.size());

		for (uint32_t i = 0; i < swapChainImages.size(); i++) {
			swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
		}
	}
	//��ȡָ���ļ��������ֽ�
	static std::vector<char>readFile(const std::string& filename)
	{
		//ate���ļ�β����ʼ��ȡ(Ϊ��ȷ������ռ�Ĵ�С)���Զ�������ʽ��ȡ�ļ��������������\n��\r\n��ת����
		std::ifstream file(filename, std::ios::ate | std::ios::binary);
		if (!file.is_open())
		{
			throw std::runtime_error("failed to open file!");
		}
		size_t fileSize = (size_t)file.tellg();
		std::vector<char>buffer(fileSize);
		//�����ļ�ͷ��
		file.seekg(0);
		//����buffer��
		file.read(buffer.data(), fileSize);
		file.close();
		return buffer;
	}
	void reCreateSwapChain()
	{
		int width = 0, height = 0;
		while (width == 0 || height == 0)
		{
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}
		//�ȴ��豸���ڿ���״̬�������ڶ����ʹ�ù����н�������ؽ�
		vkDeviceWaitIdle(device);
		//�ؽ�ǰ�����
		cleanupSwapChain();

		createSwapChain();
		//ͼ����ͼ�����ڽ�����ͼ��������Ҫ�ؽ�
		createImageViews();
		//�ӿںͲü������ڹ��ߴ���ʱ��ָ�������ڴ�С�ı䣬��ЩҲ��Ҫ�޸ģ�������Ҫ�ؽ�����
		//���ǿ��Զ�̬�����ӿںͲü���������������ؽ�
		//Ϊ����������ǲ��ı䴰�ڴ�С�ͷֱ���
		//createRenderPass();
		//createGraphicsPipeline();
		//֡�����ָ��������ڽ�����ͼ��
		createColorResources();
		createDepthResources();
		createFramebuffers();
	}
	void createRenderPass()
	{
		//��������

		VkAttachmentDescription colorAttachment = {};
		//��ɫ���帽�Ÿ�ʽ
		colorAttachment.format = swapChainImageFormat;
		//������
		colorAttachment.samples = msaaSamples;
		//��Ⱦ֮ǰ�Ը������ݽ��еĲ���
		//VK_ATTACHMENT_LOAD_OP_LOAD ������������ CLEAR ��� DONT_CARE ������
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		//��Ⱦ֮��Ը������ݽ��еĲ���
		//VK_ATTACHMENT_STORE_OP_STORE �����ݴ洢���� DONT_CARE ������
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		//ֻ��ģ�建����Ч��������û����ģ�建�壬��DONT_CARE����
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		//ָ����Ⱦ��ʼǰ��ͼ�񲼾ַ�ʽ��VK_IMAGE_LAYOUT_UNDEFINED��ʾ���ǲ�����֮ǰ��ͼ�񲼾ַ�ʽ����Ϊ����
		//��ÿ����Ⱦǰ��Ҫ���ͼ�������������ø�������������
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		//��Ⱦ�������ͼ�񲼾ַ�ʽ VK_IMAGE_LAYOUT_PRESENT_SRC_KHR��ʾͼ�����ڽ������н��г���
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentDescription depthAttachment{};
		depthAttachment.format = findDepthFormat();
		depthAttachment.samples = msaaSamples;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentDescription colorAttachmentResolve{};
		colorAttachmentResolve.format = swapChainImageFormat;
		colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;



		VkAttachmentReference colorAttachmentResolveRef{};
		colorAttachmentResolveRef.attachment = 2;
		colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		//�����̺͸�������
		//һ����Ⱦ���̿��԰�����������̣���������������һ�����̴�����֡��������
		//ÿ�������̿�������һ����������
		VkAttachmentReference colorAttachmentRef = {};
		//��VkAttachmentDescription�е�����,�ᱻƬ����ɫ��ʹ���� layout(location=0)out vec4 outColor
		colorAttachmentRef.attachment = 0;
		//ָ��ͼ�񲼾֣�OPTIMAL�������
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRef{};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		//ָ������һ����Ⱦ��������
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		subpass.pResolveAttachments = &colorAttachmentResolveRef;
		subpass.pDepthStencilAttachment = &depthAttachmentRef;
		
		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		//֮ǰ�����������̵�����
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;;
		dependency.srcAccessMask = 0;
		//ָ����Ҫ�ȴ��Ĺ��߽׶�
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;;
		////�����̽����еĲ���
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;;
			
		std::array<VkAttachmentDescription, 3>attachments = { colorAttachment,depthAttachment,colorAttachmentResolve };
		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;
		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create render pass!");
		}
	}
	void createGraphicsPipeline()
	{
		auto vertShaderCode = readFile("shaders/vert.spv");
		auto fragShaderCode = readFile("shaders/frag.spv");

		VkShaderModule vertShaderModule;
		VkShaderModule fragShaderModule;

		vertShaderModule = createShaderModule(vertShaderCode);
		fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		//pNameָ����ɫ����ָ���׶ε��õĺ�����������һ����ɫ��������ʵ����Ҫ����ɫ����Ȼ��ͨ����ͬ��pName����
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo,fragShaderStageInfo };

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();
		//��������
		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		//����󶨷�ʽ������֮��ļ��������ǰ��𶥵�ķ�ʽ���ǰ���ʵ���ķ�ʽ
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		//�������������ݸ�������ɫ�����������ͣ����ڽ����԰󶨵�������ɫ���ı���
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
		
		//����װ��
		//�������㶨�����������͵ļ���ͼԪ�Լ��Ƿ����ü���ͼԪ����
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo= {};
		inputAssemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		//ÿ�������㹹��һ��������ͼԪ
		inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		//���ΪTRUE����ʹ�ô���_STRIP��β��ͼԪ ������ͨ��һ����������ֵ0xffff����ͼԪ������������ֵ֮�������
		//����ΪͼԪ�ĵ�һ������
		inputAssemblyInfo.primitiveRestartEnable = VK_FALSE;

		//�ӿںͲü�
		//�ӿڶ�����ͼ��֡�����ӳ���ϵ
		//�ӿڴ�С�ͽ�����ͼ���С���Բ�һ����������������Ϊ������ͼ���С
		//�ü����ζ�������һ����������ر�֡����ʵ�ʴ洢���κ�λ�ڲü�����������ض��ᱻ��դ������ʧ
		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width  = (float)swapChainExtent.width;
		viewport.height = (float)swapChainExtent.height;
		//minDepthֵ���Դ���maxDepth
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		
		//�ڱ��̳��У�����ֱ�ӽ��ü���Χ����Ϊ��֡�����Сһ��
		VkRect2D scissors= {};
		scissors.offset = { 0,0 };
		scissors.extent = swapChainExtent;
		//�ӿںͲü�������Ҫͨ��VkPipelineViewportStateCreateInfo�����һ��
		VkPipelineViewportStateCreateInfo viewportStateInfo = {};
		viewportStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportStateInfo.viewportCount = 1;
		viewportStateInfo.pViewports = &viewport;
		viewportStateInfo.scissorCount = 1;
		viewportStateInfo.pScissors = &scissors;

		//��դ��
		//��դ���������Զ������ɫ���Ķ��㹹�ɵ�ͼԪת��ΪƬ�ν���Ƭ����ɫ����ɫ
		//��Ȳ��ԣ������޳��Ͳü�������ο���Ҳ�ɹ�դ������ִ��
		//�������ù�դ�����������������ͼԪ��ΪƬ�Σ�����ֻ�������ΪƬ�Σ��߿�ģʽ��
		VkPipelineRasterizationStateCreateInfo rasterizationInfo = {};
		rasterizationInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		//����ΪTRUE���Խ���ƽ���Զƽ�����Ƭ�νض��ڽ�ƽ���Զƽ���ϣ�������ֱ�Ӷ���
		rasterizationInfo.depthClampEnable = VK_FALSE;
		//����ΪTRUE�������м���ͼԪ������ͨ����դ���׶Σ���һ���û��ֹһ�������֡����
		rasterizationInfo.rasterizerDiscardEnable = VK_FALSE;
		//����ͼԪ����Ƭ�η�ʽ����������Σ������ڲ�
		rasterizationInfo.polygonMode = VK_POLYGON_MODE_FILL;
		//ָ����դ������߶ο��
		rasterizationInfo.lineWidth = 1.0f;
		//ָ�������޳����ͣ������޳����棬�����Լ�˫��
		rasterizationInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		//����ָ��˳ʱ��Ķ����������滹����ʱ��Ķ�����������
		rasterizationInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizationInfo.depthBiasEnable = VK_FALSE;
		rasterizationInfo.depthBiasConstantFactor = 0.0f;
		rasterizationInfo.depthBiasClamp = 0.0f;
		rasterizationInfo.depthBiasSlopeFactor = 0.0f;

		//���ز���
		VkPipelineMultisampleStateCreateInfo multisampleInfo = {};
		multisampleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		//���������Ƚ��ã������½ڻ���ϸ����
		multisampleInfo.sampleShadingEnable = VK_TRUE;
		multisampleInfo.minSampleShading = .2f;
		multisampleInfo.rasterizationSamples = msaaSamples;


		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;
		
		//��ɫ���
		//��ɫ�����Ҫ���������ṹ�壬һ����VkPipelineColorBlendAttachmentState,��ÿ��������֡���������ɫ
		//���ã���һ����VkPipelineColorBlendStateCreateInfo������������ȫ�ֵ���ɫ�������
		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
			| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		//���������Ȳ����л��
		colorBlendAttachment.blendEnable = VK_FALSE;
		//��֡����ɫȨ��
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		//��֡����ɫȨ��
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		//��֡��͸����Ȩ��
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		//��֡��͸����Ȩ��
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

		
		VkPipelineColorBlendStateCreateInfo colorBlendInfo = {};
		colorBlendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		//�����λ�����ϣ����Զ����õ�һ�ֻ�Ϸ�ʽ��colorWriteMask�ڵڶ��ֻ�Ϸ�ʽ����Ȼ������
		//Ҳ���Խ������ֻ��ģʽ����ʱ��Ƭ����ɫ��ֱ�Ӹ���ԭ���������洢����ɫֵ
		colorBlendInfo.logicOpEnable = VK_FALSE;
		colorBlendInfo.logicOp = VK_LOGIC_OP_COPY;
		colorBlendInfo.attachmentCount = 1;
		colorBlendInfo.pAttachments = &colorBlendAttachment;
		colorBlendInfo.blendConstants[0] = 0.0f;
		colorBlendInfo.blendConstants[1] = 0.0f;
		colorBlendInfo.blendConstants[2] = 0.0f;
		colorBlendInfo.blendConstants[3] = 0.0f;
		
		//��̬״̬
		//ֻ�зǳ����޵Ĺ���״̬�ڿ��Բ��ؽ����ߵ�����¿��Խ��ж�̬�޸ģ�������ӿڴ�С���߿�ͻ�ϳ���
		//�������ú�ᵼ������֮ǰ����ʹ�õĶ�̬״̬�����ö������Ե�����Ҫ�ڻ���ʱ����ָ�����ǵ�ֵ
		std::vector<VkDynamicState>dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicStateInfo = {};
		dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicStateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		//�������Ҫ�ڹ��ߴ�����̬�޸ģ����Խ�ָ����Ϊnullptr
		dynamicStateInfo.pDynamicStates = dynamicStates.data();

		//���߲���
		//���ǿ�������ɫ����ʹ��uniform�������������ڹ��߽�����̬�ر�Ӧ�ó����޸ģ�ʵ�ֶ���ɫ������һ���̶�
		//�Ķ�̬���ã�uniform���������������ݱ任�����������ɫ�����Լ�������������������Ƭ����ɫ��
		//uniform������Ҫ�ڹ��ߴ���ʱʹ��VkPipelineLayout���壬��Ȼ������ʱ������uniform
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout!");
		}

		//�������߶���
		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
		pipelineInfo.pViewportState = &viewportStateInfo;
		pipelineInfo.pRasterizationState = &rasterizationInfo;
		pipelineInfo.pMultisampleState = &multisampleInfo;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlendInfo;
		pipelineInfo.pDynamicState = &dynamicStateInfo;
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		//basePipelineHandle����ָ��һ���Ѿ������õĹ���Ϊ��������һ���µĹ��ߣ���������Ա������ֻ����
		//VkGraphicsPipelineCreateInfo�ṹ��ĳ�Ա����ʹ����VK_PIPELINE_CREATE_DERIVATIVE_BIT����²Ż���Ч
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = 0;
		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, 
			nullptr, &graphicsPipeline)!=VK_SUCCESS)
		{
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
	}
	void createFramebuffers()
	{
		swapChainFramebuffers.resize(swapChainImageViews.size());
		for (size_t i = 0; i < swapChainImageViews.size(); ++i)
		{
			std::array<VkImageView, 3> attachments = {
				colorImageView,
				depthImageView,
				swapChainImageViews[i]

			};

			VkFramebufferCreateInfo framebufferInfo = {};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;


			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create framebuffer!");
			}
		}

	}

	//Vulkan�µ�ָ��������ָ����ڴ洫��ָ�����ֱ��ͨ����������ִ�еģ�����Ҫ��ִ�еĲ�����¼��һ��ָ��
	//�������Ȼ���ύ������ִ����Щ�����Ķ��в���ִ�С���ʹ�����ǿ����ڳ����ʼ��ʱ��׼��������Ҫָ����ָ��
	//���У�����Ⱦʱֱ���ύִ��
	void createCommandPool()
	{
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
		VkCommandPoolCreateInfo commandPoolInfo = {};

		commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		//ָ�������ύ������
		commandPoolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;
		commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		if (vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool)!=VK_SUCCESS)
		{
			throw std::runtime_error("failed to create command pool!");
		}

	}

	//���ڻ�������֡�����Ͻ��еģ�������ҪΪ�������е�ÿһ��ͼ�����һ��ָ������
	void createCommandBuffers()
	{
		commandBuffers.resize(swapChainFramebuffers.size());
		VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
		commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		commandBufferAllocateInfo.commandPool = commandPool;
		//����ָ���ķ����ָ����������Ҫָ��廹�Ǹ���ָ������
		//Primary��Ҫָ��� ���Ա��ύ�����н���ִ�У������ܱ�����ָ���������
		//Secondary����ָ��� ����ֱ�ӱ��ύ�����н���ִ�У������Ա���Ҫָ������ִ��
		//���ǿ��԰�һЩ���õ�ָ����ڸ���ָ����Ȼ������Ҫָ�������е���ִ��
		commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		commandBufferAllocateInfo.commandBufferCount = (uint32_t)commandBuffers.size();
		if (vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, commandBuffers.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate command buffers!");
		}
	}
	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
	{
		//��¼ָ�ָ���
		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		//simultaneousʹ����������һ֡��δ������Ⱦʱ���ύ��һ֡����Ⱦָ��
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		beginInfo.pInheritanceInfo = nullptr;
		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to begin recording command buffer!");
		}
		//��ʼ��Ⱦ����
		VkRenderPassBeginInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
		//ָ����Ⱦ����
		renderPassInfo.renderArea.offset = { 0,0 };
		renderPassInfo.renderArea.extent = swapChainExtent;

		std::array<VkClearValue, 2>clearValues{};
		clearValues[0].color = {{0.0f,0.0f,0.0f,1.0f}};
		clearValues[1].depthStencil = { 1.0,0 };

		renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassInfo.pClearValues = clearValues.data();
		//VK_SUBPASS_CONTENTS_INLINE��ʾ����ָ�����Ҫ�����У�û�и���ָ�����Ҫִ��
		//VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS��ʾ�����Ը���ָ��Ļ�����Ҫִ��
		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
			VkViewport viewport{};
			viewport.x = 0.0f;
			viewport.y = 0.0f;
			viewport.width = (float)swapChainExtent.width;
			viewport.height = (float)swapChainExtent.height;
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;
			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
				
			VkRect2D scissor{};
			scissor.offset = { 0, 0 };
			scissor.extent = swapChainExtent;
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
			VkBuffer vertexBuffers[] = { vertexBuffer };
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
			vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdBindDescriptorSets(commandBuffer,
				VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
				&descriptorSets[currentFrame], 0, nullptr);
			vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
		vkCmdEndRenderPass(commandBuffer);
		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to record command buffer!");
		}
	}
	//����ͬ������
	void createSyncObjects()
	{
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
		VkSemaphoreCreateInfo semaphoreCreateInfo = {};
		semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceCreateInfo = {};
		fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT;i++)
		{
			if (vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create imageAvailble semaphore!");
			}
			if (vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create renderFinished semaphore!");
			}
			if (vkCreateFence(device, &fenceCreateInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create fence");
			}
		}
	}
	VkShaderModule createShaderModule(const std::vector<char>& code)
	{
		VkShaderModuleCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		//��Ҫ�Ƚ��洢�ֽ��������ָ��ת��Ϊconst uint32_t*��ƥ��ṹ���е��ֽ���ָ��ı�������
		//���⣬ָ���ָ���ַӦ�÷���uint32_t�������͵��ڴ���뷽ʽ������ʹ�õ���vector������Ҫ��
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule)!=VK_SUCCESS)
		{
			throw std::runtime_error("failed to create shader module!");
		}
		return shaderModule;
	}
	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
	{
		SwapChainSupportDetails details;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
		if (formatCount != 0)
		{
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}
		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
		if (presentModeCount != 0)
		{
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}
		return details;
	}
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
	{
		if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED)
		{
			return { VK_FORMAT_B8G8R8A8_UNORM,VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
		}
		for (const auto& format : availableFormats)
		{
			if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			return format;
		}
		return availableFormats[0];
	}
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>availablePresentModes)
	{
		VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;
		for (const auto& presentMode : availablePresentModes)
		{
			if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				return presentMode;
			}
			else if (presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
			{
				bestMode=presentMode;
			}
		}
		return bestMode;
	}
	bool checkDeviceExtensionSupport(VkPhysicalDevice device)
	{
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties>availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
		std::set<std::string>requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
		for (const auto& extension : availableExtensions)
		{
			requiredExtensions.erase(extension.extensionName);
		}
		return requiredExtensions.empty();
	}

	bool isDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device);
		bool extensionSupported = checkDeviceExtensionSupport(device);
		bool swapChainAdequate = false;
		if (extensionSupported)
		{
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}
		VkPhysicalDeviceFeatures supportedFeatures;
		vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
		return indices.isComplete() && extensionSupported&&swapChainAdequate&&supportedFeatures.samplerAnisotropy;
	}
	//������Χ�ǽ������е�ͼ��ֱ���
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR&capabilities)
	{
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		{
			return capabilities.currentExtent;
		}
		else
		{
			VkExtent2D actualExtent = { WIDTH,HEIGHT };
			actualExtent.width = std::min(capabilities.minImageExtent.width, std::max(capabilities.maxImageExtent.width, actualExtent.width));
			actualExtent.height = std::min(capabilities.minImageExtent.height, std::max(capabilities.maxImageExtent.height, actualExtent.height));
			return actualExtent;
		}

	}
	void createSwapChain()
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
		//�������п������ɵ�ͼ�����
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
		{
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}
		VkSwapchainCreateInfoKHR createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;
		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		//ÿ��ͼ���������Ĳ��
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily,indices.presentFamily };
		if (indices.graphicsFamily != indices.presentFamily)
		{
			//ͼ������ڶ������ʹ�ã�����Ҫ��ʽ�ı�����Ȩ��������Ҫ��������
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else
		{
			//һ��ͼ��һ��ʱ��ֻ����һ�������������ʽ�ظı�����Ȩ���������
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}
		//�����й̶��ķ�ת����
		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		//����alphaͨ��
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		//��Ϊtrue��ʾ�����ı�����ϵͳ�����������ڵ������ص���ɫ��������vulkan��ȡһ�����Ż�
		createInfo.clipped = VK_TRUE;
		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
		{
			throw::std::runtime_error("failed to create swap chain!");
		}
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}
	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
	{
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties>queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,queueFamilies.data());
		int i = 0;
		//�ҵ�һ��֧��VK_QUEUE__GRAPHICS_BIT�Ķ�����
		for (const auto& queueFamily : queueFamilies)
		{

			if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				indices.graphicsFamily = i;
			}
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
			if (presentSupport)
			{
				indices.presentFamily = i;
			}
			if (indices.isComplete())
			{
				break;
			}
			i++;
		}
		return indices;
	}
	void createLogicalDevice()
	{
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily, indices.presentFamily };
		for (uint32_t queueFamily : uniqueQueueFamilies)
		{
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			//vulkan��Ҫ���Ǹ���һ��0��1�ĸ�������ָ�����ȼ������ȼ����Կ���ָ���ִ��˳��
			float queuePriority = 1.0;
			queueCreateInfo.pQueuePriorities = &queuePriority;	
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures = {};
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		deviceFeatures.sampleRateShading = VK_TRUE;
		VkDeviceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.queueCreateInfoCount = queueCreateInfos.size();
		createInfo.pEnabledFeatures = &deviceFeatures;

		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();
		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else
		{
			createInfo.enabledLayerCount = 0;
		}
		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create logical device!");
		}
		//�����Դ�Ϊ�߼��豸�������������������������صĶ��о��
		vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily,  0, &presentQueue);
	}
	void createinstance()
	{
		//���������У��㲢����ЩУ����б�û��֧�֣��ͻᱨ��
		if (enableValidationLayers && !checkValidationLayerSupport())
		{
			throw std::runtime_error("validation layers requested, but not available!");
		}
		//��дӦ�ó�����Ϣ����Ȼ���Ǳ��룬���ǿ��ܻ���Ϊ�����Ż������ݣ�����ʹ����ĳ�����棬
		//Ȼ�������������������������Ż�
		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello,Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion= VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		//Vulkan������ͨ���ṹ�崫����Ϣ
		//����Vulkan����������Ҫʹ�õ�ȫ����չ��У��㣬ȫ��ָ������Ӧ�ö���Ч�������ǵ�һ�豸
		VkInstanceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else
		{
			createInfo.enabledLayerCount = 0;
			createInfo.pNext = nullptr;
		}

		//��һ�������Ǵ�����Ϣ�Ľṹ��ָ�룬�ڶ������Զ���������ص�����������û���Զ��������
		//��������ָ���¶������洢λ�õ�ָ��
		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create instance!");
		}

	}
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
	}

	void setupDebugMessenger() {
		if (!enableValidationLayers) return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}
	void mainLoop()
	{
		//ֻҪ����û�д����Ҵ���û�رգ���ô�Ϳ���һֱ����
		while (!glfwWindowShouldClose(window))
		{
			//ִ���¼�����
			glfwPollEvents();
			drawFrame();
		}
		vkDeviceWaitIdle(device);
	}

	//1.�ӽ�������ȡһ��ͼ��
	//2.��֡���帽��ִ��ָ����е���Ⱦָ��
	//3.������Ⱦ���ͼ�񵽽��������г��ֲ���	
	//�������漸�������������첽ִ�еģ�����������Ҫ��դ�����ź�������ͬ����
	//դ���Ƕ�Ӧ�ó��������Ⱦ���������첽�����ź����Ƕ�һ��ָ������ڵĲ����������н���ͬ��
	void drawFrame()
	{
		//�ȴ���ǰ֡��ʹ�õ�ָ������ִ��
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());
		uint32_t imageIndex;
		VkResult result=vkAcquireNextImageKHR(device, swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			reCreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
		{
			throw std::runtime_error("failed to acquire swap chain image!");
		}
		updateUniformBuffer(currentFrame);

		vkResetFences(device, 1, &inFlightFences[currentFrame]);
		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		recordCommandBuffer(commandBuffers[currentFrame], imageIndex);
		//�ύ��Ϣ��ָ�����
		VkSubmitInfo submitInfo{};		
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame]};
		//�ȴ�ͼ����ߵ������д����ɫ���ŵĽ׶�
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		//ָ��ִ��ǰ��Ҫ�ȴ����ź���
		submitInfo.pWaitSemaphores = waitSemaphores;
		//ָ����Ҫ�ȴ��Ĺ��߽׶�
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame]};
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo,inFlightFences[currentFrame]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to submit draw command buffer!");
		}
		//����
		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;
		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains; 
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;
		result=vkQueuePresentKHR(presentQueue, &presentInfo);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR||framebufferResized)
		{
			framebufferResized = false;
			reCreateSwapChain();
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("failed to present swap chain image!");
		}
		//vkQueueWaitIdle(presentQueue);
		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void cleanupSwapChain()
	{
		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);

		vkDestroyImageView(device, colorImageView, nullptr);
		vkDestroyImage(device, colorImage, nullptr);
		vkFreeMemory(device, colorImageMemory, nullptr);
		for (auto frameBuffer:swapChainFramebuffers )
		{
			vkDestroyFramebuffer(device, frameBuffer, nullptr);
		}
		for (auto imageView : swapChainImageViews)
		{
			vkDestroyImageView(device, imageView, nullptr);
		}
		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	void cleanup()
	{
		cleanupSwapChain();
		vkDestroySampler(device, textureSampler, nullptr);
		vkDestroyImageView(device, textureImageView, nullptr);
		vkDestroyImage(device, textureImage, nullptr);
		vkFreeMemory(device, textureImageMemory, nullptr);
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
		}
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		vkDestroyBuffer(device, indexBuffer, nullptr);
		vkFreeMemory(device, indexBufferMemory, nullptr);
		vkDestroyBuffer(device, vertexBuffer, nullptr);
		vkFreeMemory(device, vertexBufferMemory, nullptr);
		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}
		vkDestroyCommandPool(device, commandPool, nullptr);

		vkDestroyDevice(device, nullptr);

		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}
		//���ʵ��
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
		//���ٴ��ڣ������Դ
		glfwDestroyWindow(window);
		glfwTerminate();
	}
	std::vector<const char*>getRequiredExtensions()
	{
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		std::vector<const char*>extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
		if (enableValidationLayers)
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}
		return extensions;
	}
	bool checkValidationLayerSupport()
	{
		uint32_t layerCount;
		//��ȡУ�����
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		std::vector<VkLayerProperties>availableLayers(layerCount);
		//��ø�У�������
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
		//������е�У����б�
		for (const char* layerName : validationLayers)
		{
			bool layerFound = false;
			//�鿴��availableLayers���Ƿ����ҵ�
			for (const auto& layerProperties : availableLayers)
			{
				if (strcmp(layerName, layerProperties.layerName) == 0)
				{
					layerFound = true;
					break;
				}
			}
			if (!layerFound)
			{
				return false;
			}
		}
		return true;
	}
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}

};
int main()
{
	HelloTriangleApplication app;

	try {
		app.run();
	}catch (const std::exception& e){
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}