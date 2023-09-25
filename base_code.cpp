#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>
#include<iostream>
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
struct QueueFamilyIndices {
	int graphicsFamily=-1;

	bool isComplete() {
		return graphicsFamily>=0;
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
	GLFWwindow* window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	void initWindow()
	{
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
		pickPhysicalDevice();
	}
	void pickPhysicalDevice()
	{
		VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
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
	bool isDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device);

		return indices.isComplete();
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
			if (indices.isComplete())
			{
				break;
			}
			i++;
		}
		return indices;
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
		}
	}
	void cleanup()
	{
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