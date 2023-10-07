#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>
#include<fstream>
#include<algorithm>
#include<iostream>
#include<set>
#include<stdexcept>
#include<functional>
#include<cstdlib>
#include<vector>
const int WIDTH = 800;
const int HEIGHT = 600;
//ָ��У�������
const std::vector<const char*>validationLayers = { "VK_LAYER_KHRONOS_validation" };
//�Ƿ���У���
bool enableValidationLayers = true;

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
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;
	VkCommandPool commandPool;
	std::vector<VkCommandBuffer>commandBuffers;
	VkSemaphore imageAvailableSemaphore;
	VkSemaphore renderFinishedSemaphore;

	VkDebugUtilsMessengerEXT debugMessenger;
	void initWindow()
	{
		glfwInit();
		//��ʾ������GLFW��ֹ���Զ�����openGL������
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		//��ֹ���ڴ�С�ı�
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		//ǰ���������ֱ�Ϊ��ȣ��߶ȣ����⣬
		//���ĸ�����ָ�����ĸ���ʾ������ʾ���������openGL��أ������vulkanû��
		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

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
		//������ǰ�������й��ߣ��������ܸ�������������Ż��ռ�
		createGraphicsPipeline();
		//�����������
		createFramebuffers();
		//ָ���
		createCommandPool();
		//ָ���
		createCommandBuffers();
		createSemaphores();
	}
	void createSurface()
	{
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create window surface!");
		}
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
		size_t swapChainImages_size=swapChainImages.size();
		for (size_t i = 0; i < swapChainImages_size; ++i)
		{
			VkImageViewCreateInfo createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = swapChainImages[i];
			//ָ��һά�����Ƕ�ά��������ά�����������
			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			//ָ��ͼ�����ݽ�����ʽ
			createInfo.format = swapChainImageFormat;
			//����ͼ����ɫͨ����ӳ��
			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			//����ָ��ͼ�����;����һ�����ܹ�����
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;
			if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i])!=VK_SUCCESS)
			{
				throw std::runtime_error("failed to create ImageViews!");
			}
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

	void createRenderPass()
	{
		//��������

		VkAttachmentDescription colorAttachment = {};
		//��ɫ���帽�Ÿ�ʽ
		colorAttachment.format = swapChainImageFormat;
		//������
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
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
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		//�����̺͸�������
		//һ����Ⱦ���̿��԰�����������̣���������������һ�����̴�����֡��������
		//ÿ�������̿�������һ����������
		VkAttachmentReference colorAttachmentRef = {};
		//��VkAttachmentDescription�е�����,�ᱻƬ����ɫ��ʹ���� layout(location=0)out vec4 outColor
		colorAttachmentRef.attachment = 0;
		//ָ��ͼ�񲼾֣�OPTIMAL�������
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		//ָ������һ����Ⱦ��������
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		
		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		//֮ǰ�����������̵�����
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		//ָ����Ҫ�ȴ��Ĺ��߽׶�
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		////�����̽����еĲ�������������Ϊ��ɫ���ŵĶ�д����
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			

		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colorAttachment;
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

		//��������
		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		//����󶨷�ʽ������֮��ļ��������ǰ��𶥵�ķ�ʽ���ǰ���ʵ���ķ�ʽ
		vertexInputInfo.vertexBindingDescriptionCount = 0;
		vertexInputInfo.pVertexBindingDescriptions = nullptr;
		//�������������ݸ�������ɫ�����������ͣ����ڽ����԰󶨵�������ɫ���ı���
		vertexInputInfo.vertexAttributeDescriptionCount = 0;
		vertexInputInfo.pVertexAttributeDescriptions = nullptr;
		
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
		rasterizationInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizationInfo.depthBiasEnable = VK_FALSE;
		rasterizationInfo.depthBiasConstantFactor = 0.0f;
		rasterizationInfo.depthBiasClamp = 0.0f;
		rasterizationInfo.depthBiasSlopeFactor = 0.0f;

		//���ز���
		VkPipelineMultisampleStateCreateInfo multisampleInfo = {};
		multisampleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		//���������Ƚ��ã������½ڻ���ϸ����
		multisampleInfo.sampleShadingEnable = VK_FALSE;
		multisampleInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampleInfo.minSampleShading = 1.0f;
		multisampleInfo.pSampleMask = nullptr;
		multisampleInfo.alphaToCoverageEnable = VK_FALSE;
		multisampleInfo.alphaToOneEnable = VK_FALSE;
		
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
		VkPipelineDynamicStateCreateInfo dynamicStateInfo;
		dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicStateInfo.dynamicStateCount = 2;
		//�������Ҫ�ڹ��ߴ�����̬�޸ģ����Խ�ָ����Ϊnullptr
		dynamicStateInfo.pDynamicStates = dynamicStates.data();

		//���߲���
		//���ǿ�������ɫ����ʹ��uniform�������������ڹ��߽�����̬�ر�Ӧ�ó����޸ģ�ʵ�ֶ���ɫ������һ���̶�
		//�Ķ�̬���ã�uniform���������������ݱ任�����������ɫ�����Լ�������������������Ƭ����ɫ��
		//uniform������Ҫ�ڹ��ߴ���ʱʹ��VkPipelineLayout���壬��Ȼ������ʱ������uniform
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 0;
		pipelineLayoutInfo.pSetLayouts = nullptr;
		pipelineLayoutInfo.pushConstantRangeCount = 0;
		pipelineLayoutInfo.pPushConstantRanges = nullptr;

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
		pipelineInfo.pDepthStencilState = nullptr;
		pipelineInfo.pColorBlendState = &colorBlendInfo;
		pipelineInfo.pDynamicState = nullptr;
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		//basePipelineHandle����ָ��һ���Ѿ������õĹ���Ϊ��������һ���µĹ��ߣ���������Ա������ֻ����
		//VkGraphicsPipelineCreateInfo�ṹ��ĳ�Ա����ʹ����VK_PIPELINE_CREATE_DERIVATIVE_BIT����²Ż���Ч
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;
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
			VkImageView attachments[] = { swapChainImageViews[i] };
			VkFramebufferCreateInfo framebufferInfo = {};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments = attachments;
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
		commandPoolInfo.flags = 0;
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
		VkClearValue clearColor = { 0.0f,0.0f,0.0f,1.0f };
		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = &clearColor;
		//VK_SUBPASS_CONTENTS_INLINE��ʾ����ָ�����Ҫ�����У�û�и���ָ�����Ҫִ��
		//VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS��ʾ�����Ը���ָ��Ļ�����Ҫִ��
		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
		vkCmdDraw(commandBuffer, 3, 1, 0, 0);
		vkCmdEndRenderPass(commandBuffer);
		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to record command buffer!");
		}
	}
	void createSemaphores()
	{
		VkSemaphoreCreateInfo semaphoreCreateInfo = {};
		semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		if (vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create imageAvailble semaphore!");
		}
		if (vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create renderFinished semaphore!");
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
		return indices.isComplete() && extensionSupported&&swapChainAdequate;
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
		VkSwapchainKHR swapChain;
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
		uint32_t imageIndex;
		vkAcquireNextImageKHR(device, swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
		//�ύ��Ϣ��ָ�����
		VkSubmitInfo submitInfo{};		
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		VkSemaphore waitSemaphores[] = { imageAvailableSemaphore };	
		//�ȴ�ͼ����ߵ������д����ɫ���ŵĽ׶�
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		//ָ��ִ��ǰ��Ҫ�ȴ����ź���
		submitInfo.pWaitSemaphores = waitSemaphores;
		//ָ����Ҫ�ȴ��Ĺ��߽׶�
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphore };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;
		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
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
		vkQueuePresentKHR(presentQueue, &presentInfo);
	}

	void cleanup()
	{
		vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
		vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
		vkDestroyCommandPool(device, commandPool, nullptr);
		for (auto framebuffer : swapChainFramebuffers)
		{
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}
		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);
		for (auto& imageView : swapChainImageViews)
		{
			vkDestroyImageView(device, imageView, nullptr);
		}
		vkDestroySwapchainKHR(device, swapChain, nullptr);
		vkDestroyDevice(device, nullptr);
		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}
		//���ʵ��
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