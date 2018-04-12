
import std.stdio;
import std.range;
import std.array;
import std.algorithm : all, any;
import std.exception;
import std.conv;
import std.string;
import std.traits;

import erupted;
import gl3n.math, gl3n.linalg;
import util;
import core.stdc.string : memcpy;

import derelict.glfw3.glfw3;


private void enforceVK(VkResult res) {
  enforce(res == VkResult.VK_SUCCESS, res.to!string);
}

extern(System) VkBool32 VkDebug_Report_Callback(
  VkDebugReportFlagsEXT      flags,
  VkDebugReportObjectTypeEXT object_type,
  uint64_t                   object,
  size_t                     location,
  int32_t                    message_code,
  const (char*)              pLayerPrefix,
  const (char*)              pMessage,
  void*                      pUserData ) nothrow @nogc
{
  printf("--- Encountered error or warning (VkDebug_Report_Callback) ---\n");
  printf("ObjectType: %i\n", object_type);
  printf("Message: ");
  printf(pMessage);
  printf("\n");
  return VK_FALSE;
}

template VkRBuffer(alias func) {
  alias Params = Parameters!func;
  static if(is(Params[$-1] == T*, T)) alias BufferType = T;
  alias RType = ReturnType!func;

  BufferType[] VkRBuffer(Params[0..$-2] args) {
    // -- grab lengths;
    uint len;
    static if ( is(RType == void) ) func(args, &len, null);
    else enforceVK(func(args, &len, null));
    // -- set buffer
    BufferType[] buf; buf.length = len;
    static if ( is(RType == void) ) func(args, &len, buf.ptr);
    else enforceVK(func(args, &len, buf.ptr));
    return buf;
  }
}

alias VkRDeviceExtensionProperties =
        VkRBuffer!vkEnumerateDeviceExtensionProperties;
alias vkRPhysicalDeviceFormatsKHR =
        VkRBuffer!vkGetPhysicalDeviceSurfaceFormatsKHR;
alias vkRPhysicalDeviceSurfacePresentModesKHR =
        VkRBuffer!vkGetPhysicalDeviceSurfacePresentModesKHR;
alias vkRSwapchainImagesKHR = VkRBuffer!vkGetSwapchainImagesKHR;

/**
  Params:
    layer_name: String naming the layer to retrieve extensions from
  Return: A buffer of all global extension properties, VkExtensionProperties
**/
VkExtensionProperties[] RInstance_Extension_Properties (string layer_name = ""){
  VkExtensionProperties[] props;
  uint plen; vkEnumerateInstanceExtensionProperties(null, &plen, null);
  props.length = plen;
  auto layer_ptr = layer_name == "" ? null : layer_name.ptr;
  vkEnumerateInstanceExtensionProperties(layer_ptr, &plen, props.ptr);
  return props;
}

/**
  Return: A buffer of all global layer properties
**/
VkLayerProperties[] RInstance_Layer_Properties ( ) {
  VkLayerProperties[] props;
  uint plen; vkEnumerateInstanceLayerProperties(&plen, null);
  props.length = plen;
  vkEnumerateInstanceLayerProperties(&plen, props.ptr);
  return props;
}

/**
  Return: A buffer of all physical devices accessible to vulkan instance
**/
VkPhysicalDevice[] RPhysical_Devices ( VkInstance inst ) {
  VkPhysicalDevice[] devices;
  uint plen; vkEnumeratePhysicalDevices(inst, &plen, null);
  devices.length = plen;
  vkEnumeratePhysicalDevices(inst, &plen, devices.ptr);
  return devices;
}

/**
  Parameter: A physical device
  Return: A buffer of all family properties of the specified physical device
**/
VkQueueFamilyProperties[] RPhysical_Device_Queue_Family(
                                    VkPhysicalDevice device ) {
  VkQueueFamilyProperties[] properties;
  uint plen; vkGetPhysicalDeviceQueueFamilyProperties(device, &plen, null);
  properties.length = plen;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &plen, properties.ptr);
  return properties;
}

/**
  Parameter: A physical device
  Return: device properties of the physical device
**/
VkPhysicalDeviceProperties RPhysical_Device_Properties ( VkPhysicalDevice d ) {
  VkPhysicalDeviceProperties device_properties;
  vkGetPhysicalDeviceProperties(d, &device_properties);
  return device_properties;
}

/**
  Parameter: A physical device
  Return: device features of the physical device
**/
VkPhysicalDeviceFeatures* RPhysical_Device_Features ( VkPhysicalDevice d ) {
  VkPhysicalDeviceFeatures* device_features = new VkPhysicalDeviceFeatures();
  vkGetPhysicalDeviceFeatures(d, device_features);
  return device_features;
}

enum QueueFamily { Gfx, Present, Transfer };
struct QueueFamilyIndices {
  private int[QueueFamily.max+1] families;

  void Init ( ) {
    foreach ( ref i; families ) i = -1;
  }

  void Set_Index ( QueueFamily family, uint idx ) {
    families[cast(uint)family] = cast(int)idx;
  }

  void Set_Index ( QueueFamily family, size_t idx ) {
    Set_Index(family, cast(uint)idx);
  }

  uint RIndex ( QueueFamily family ) { return families[cast(uint)family]; }

  size_t RLength ( ) { return families.length; }

  bool Is_Complete ( ) {
    return families[cast(uint)QueueFamily.Gfx] >= 0 &&
           families[cast(uint)QueueFamily.Present] >= 0;
  }
}

struct SwapchainDetails {
  VkSwapchainKHR swapchain;
  VkImage[] images;
  VkImageView[] image_views; // Just how to access images, aka view into image
  VkFramebuffer[] framebuffers;
  VkFormat image_format;
  VkExtent2D extent;
}

struct SwapchainDetailsPreCreation {
  VkSurfaceFormatKHR surface_format;
  VkPresentModeKHR present_mode;
  VkExtent2D extent;
}

struct SwapchainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  VkSurfaceFormatKHR[] formats;
  VkPresentModeKHR[] present_modes;

  @disable this();
  this ( VkPhysicalDevice device, VkSurfaceKHR surface ) {
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &capabilities);
    formats = vkRPhysicalDeviceFormatsKHR(device, surface);
    present_modes = vkRPhysicalDeviceSurfacePresentModesKHR(device, surface);
  }

  bool Sufficient() { return !formats.empty() && !present_modes.empty(); }

  /**
    Returns a SwapchainDetailsPreCreation struct, selecting from the current details
  **/
  SwapchainDetailsPreCreation RDetails (int width, int height) {
    SwapchainDetailsPreCreation swapchain_details;
    swapchain_details.surface_format = Select_Swap_Surface_Format(formats);
    swapchain_details.present_mode   = Select_Swap_Present_Mode(present_modes);
    swapchain_details.extent  = Select_Swap_Extent(capabilities, width, height);
    return swapchain_details;
  }

  /**
    Returns a partially filled VkSwapchainCreateInfoKHR, specifically with:
      `minImageCount`, `preTransform`
  **/
  VkSwapchainCreateInfoKHR RSwapchain_Create_Info ( ) {
    VkSwapchainCreateInfoKHR create_info;
    uint image_count = capabilities.minImageCount+1;
    // Get swap chain length, try to get one more above the minimum image count
    //   for triple buffering.
    uint max_image_count = capabilities.maxImageCount; // 0 = no limit
    if ( max_image_count > 0 && image_count > max_image_count )
      image_count = max_image_count;
    create_info.minImageCount = image_count;
    create_info.preTransform = capabilities.currentTransform;
    return create_info;
  }
}

/**
  Return a proper swap surface format
**/
VkSurfaceFormatKHR Select_Swap_Surface_Format(VkSurfaceFormatKHR[] formats) {
  // The best case scenario is unspecified, as that entails we can choose
  // any format, which in this case we'll use non-linear SRGB
  if ( formats.length == 1 && formats[0].format == VK_FORMAT_UNDEFINED )
    return VkSurfaceFormatKHR(VK_FORMAT_B8G8R8A8_UNORM,
                              VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);
  // check for nonlinear in the available combinations
  foreach ( f; formats ) {
    if ( f.format == VK_FORMAT_B8G8R8A8_UNORM &&
          f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR ) {
      return f;
    }
  }

  // Just go with the first format then
  return formats[0];
}

/**
  Return a proper swap present mode
**/
VkPresentModeKHR Select_Swap_Present_Mode(VkPresentModeKHR[] modes) {
  // Choose best modes (triple buffering -> immediate -> vsync).
  // Vsync is the only mode gaurunteed to be present
  VkPresentModeKHR best_mode = VK_PRESENT_MODE_FIFO_KHR;
  foreach ( m; modes ) {
    // triple-buffering (if queue is full, replace old surfaces with new)
    if ( m == VK_PRESENT_MODE_MAILBOX_KHR ) return m;
    // Immediate mode, aka no vsync
    if ( m == VK_PRESENT_MODE_IMMEDIATE_KHR ) best_mode = m;
  }
  return best_mode;
}

VkExtent2D Select_Swap_Extent(VkSurfaceCapabilitiesKHR capabilities,
                                              int width, int height) {
  // Check if window manager allows us to change current swap extent
  //   (they set the current extent to maximum uint value)
  if ( capabilities.currentExtent.width != uint.max )
    return capabilities.currentExtent;

  // Choose our own extent
  auto min = &capabilities.minImageExtent,
       max = &capabilities.maxImageExtent;
  VkExtent2D actual_extent = VkExtent2D(width, height);
  actual_extent.width = clamp(actual_extent.width, min.width, max.width);
  actual_extent.height = clamp(actual_extent.height, min.height, max.height);
  return actual_extent;
}


struct VkContext {
  VkInstance instance;
  VkDebugReportCallbackEXT debug_callback;
  VkSurfaceKHR surface;
  VkPhysicalDevice physical_device;
  VkDevice device;
  QueueFamilyIndices queue_family;
  VkQueue graphics_queue, present_queue;
  SwapchainDetails swapchain;
  VkCommandPool command_pool;
  VkCommandBuffer[] command_buffers;
  VkRenderPass render_pass;
  VkDescriptorSetLayout descriptor_set_layout;
  VkDescriptorPool descriptor_pool;
  VkDescriptorSet descriptor_set;
  VkPipelineLayout pipeline_layout;
  VkPipeline pipeline;
  VkSemaphore image_available_semaphore, render_finished_semaphore;

  VkBuffer vertex_buffer, index_buffer, uniform_buffer;
  VkImage texture_image;
  VkImageView texture_image_view;
  VkSampler texture_sampler;
  VkDeviceMemory vertex_buffer_memory, index_buffer_memory,
                 uniform_buffer_memory, texture_image_memory;

  VkImage depth_image;
  VkDeviceMemory depth_image_memory;
  VkImageView depth_image_view;
}

void Cleanup_Swapchain ( ref VkContext ctx ) {
  writeln("Destroying depth image");
  vkDestroyImageView(ctx.device, ctx.depth_image_view, null);
  vkDestroyImage(ctx.device, ctx.depth_image, null);
  vkFreeMemory(ctx.device, ctx.depth_image_memory, null);
  writeln("Destroying framebuffers");
  foreach ( framebuffer; ctx.swapchain.framebuffers )
    vkDestroyFramebuffer(ctx.device, framebuffer, null);
  vkFreeCommandBuffers(ctx.device, ctx.command_pool,
          cast(uint)ctx.command_buffers.length, ctx.command_buffers.ptr);
  writeln("Destroying pipeline");
  vkDestroyPipeline(ctx.device, ctx.pipeline, null);
  writeln("Destroying pipeline layout");
  vkDestroyPipelineLayout(ctx.device, ctx.pipeline_layout, null);
  writeln("Destroying render pass");
  vkDestroyRenderPass(ctx.device, ctx.render_pass, null);
  writeln("Destroying image view");
  foreach ( i; ctx.swapchain.image_views )
    vkDestroyImageView(ctx.device, i, null);
  writeln("Destroying swapchain");
  vkDestroySwapchainKHR(ctx.device, ctx.swapchain.swapchain, null);
}

void Cleanup ( ref VkContext ctx ) {
  writeln("Destroying debug report callback");
  vkDestroyDebugReportCallbackEXT(ctx.instance, ctx.debug_callback, null);
  writeln("Destroying semaphores");
  vkDestroySemaphore(ctx.device, ctx.image_available_semaphore, null);
  vkDestroySemaphore(ctx.device, ctx.render_finished_semaphore, null);
  ctx.Cleanup_Swapchain;
  writeln("Destroying sampler");
  vkDestroySampler(ctx.device, ctx.texture_sampler, null);
  writeln("Destroying texture image + memory");
  vkDestroyImage(ctx.device, ctx.texture_image, null);
  vkFreeMemory(ctx.device, ctx.texture_image_memory, null);
  writeln("Destroying image view");
  vkDestroyImageView(ctx.device, ctx.texture_image_view, null);
  writeln("Destroying descriptor pool");
  vkDestroyDescriptorPool(ctx.device, ctx.descriptor_pool, null);
  writeln("Destroying descriptor set layouts");
  vkDestroyDescriptorSetLayout(ctx.device, ctx.descriptor_set_layout, null);
  writeln("Destroying uniform buffer");
  vkDestroyBuffer(ctx.device, ctx.uniform_buffer, null);
  writeln("Destroying uniform buffer memory");
  vkFreeMemory(ctx.device, ctx.uniform_buffer_memory, null);
  writeln("Destroying index buffers");
  vkDestroyBuffer(ctx.device, ctx.index_buffer, null);
  writeln("Freeing index buffer memory");
  vkFreeMemory(ctx.device, ctx.index_buffer_memory, null);
  writeln("Destroying vertex buffers");
  vkDestroyBuffer(ctx.device, ctx.vertex_buffer, null);
  writeln("Freeing vertex buffer memory");
  vkFreeMemory(ctx.device, ctx.vertex_buffer_memory, null);
  writeln("Destroying command pool");
  vkDestroyCommandPool(ctx.device, ctx.command_pool, null);
  // VkPhysicalDevice is destroyed implicitly with VkInstance
  writeln("Destroying logical device");
  vkDestroyDevice(ctx.device, null);
  writeln("Destroying surface");
  vkDestroySurfaceKHR(ctx.instance, ctx.surface, null);
  writeln("Destroying instance");
  vkDestroyInstance(ctx.instance, null);
  writeln("Cleanup finished");
}

struct Vertex {
  float3 origin;
  float3 colour;
  float2 tex_coord;

  /**
    Returns a binding description for this vertex struct. Describing at which
      rate to load data from memory throughout the vertices
  **/
  static VkVertexInputBindingDescription RBinding_Description ( ) {
    VkVertexInputBindingDescription binding_description;
    binding_description.binding = 0;
    binding_description.stride = Vertex.sizeof;
    binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return binding_description;
  }

  /**
    Returns a attribute description for this vertex struct, describing how
      to extract attribute information from the struct
  **/
  static VkVertexInputAttributeDescription[3] RAttribution_Description () {
    VkVertexInputAttributeDescription[3] attribute_descriptions;
    // origin
    attribute_descriptions[0].binding = 0;
    attribute_descriptions[0].location = 0;
    attribute_descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attribute_descriptions[0].offset = Vertex.origin.offsetof;
    // colour
    attribute_descriptions[1].binding = 0;
    attribute_descriptions[1].location = 1;
    attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attribute_descriptions[1].offset = Vertex.colour.offsetof;
    // UV coordinates
    attribute_descriptions[2].binding = 0;
    attribute_descriptions[2].location = 2;
    attribute_descriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attribute_descriptions[2].offset = Vertex.tex_coord.offsetof;
    return attribute_descriptions;
  }
}

struct UniformBufferObject {
  float4x4 model;
  float4x4 view;
  float4x4 proj;
}

mixin DerelictGLFW3_VulkanBind;

struct GLFWContext {
  GLFWwindow* window;
}

class TriangleApplication {
private:
  GLFWContext glfw_ctx;
  VkContext vk_ctx;

  immutable Vertex[] vertices = [
    {float3(-0.5f, -0.5f, -0.5f),  float3(1f, 0f, 0f), float2(1.0f, 0.0f)},
    {float3( 0.5f, -0.5f, -0.5f),  float3(0f, 1f, 0f), float2(0.0f, 0.0f)},
    {float3( 0.5f, 0.5f,  -0.5f),  float3(0f, 0f, 1f), float2(0.0f, 1.0f)},
    {float3(-0.5f, 0.5f,  -0.5f),  float3(1f, 1f, 1f), float2(1.0f, 1.0f)},

    {float3(-0.3f, -0.3f, -0.40f), float3(1f, 0f, 0f), float2(1.0f, 0.0f)},
    {float3( 0.7f, -0.3f, -0.40f), float3(0f, 1f, 0f), float2(0.0f, 0.0f)},
    {float3( 0.7f, 0.7f, -0.40f), float3(0f, 0f, 1f), float2(0.0f, 1.0f)},
    {float3(-0.3f, 0.7f, -0.40f), float3(1f, 1f, 1f), float2(1.0f, 1.0f)},
  ];
  immutable uint[] indices = [
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4
  ];
  // Vertex[] vertices;
  // uint[] indices;

  void Load_Model ( ) {
    import wavefront;
    WavefrontObj obj = new WavefrontObj("african.obj");
  }

  bool Vk_Has_All(string prop, Range)(const(char*)[] elem, Range vk) {
    import core.stdc.string;
    mixin(q{
      return elem[].all!(name =>
        vk[].any!(ext =>
          strcmp(cast(const(char*))ext.%s, name) == 0));
    }.format(prop));
  }

  // Get device features
  static immutable(const(char*)[]) Device_extensions = [
    "VK_KHR_swapchain",
  ];

  const(char*)[] RVulkan_Extensions ( ) {
    uint glfw_extension_count = 0;
    const (char)** glfw_extensions;
    glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
    const(char*)[] extensions;
    foreach ( i; 0 .. glfw_extension_count )
      extensions ~= glfw_extensions[i];
    extensions ~= VK_EXT_DEBUG_REPORT_EXTENSION_NAME;
    return extensions;
  }

  bool Check_Validation_Layer_Support ( ) {
    auto layer_properties = RInstance_Layer_Properties();
    return Vk_Has_All!("layerName")(Validation_layers, layer_properties);
  }

  void Setup_Vk_Instance ( ) {
    // Check validation support
    enforce(Check_Validation_Layer_Support(),
            "Validation layers required");
    // -- create application info --
    VkApplicationInfo appinfo;
    appinfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appinfo.pNext = null;
    appinfo.pApplicationName = "Hello Triangle";
    appinfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appinfo.pEngineName = "No engine";
    appinfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appinfo.apiVersion = VK_API_VERSION_1_0;
    // -- get extensions
    const(char*)[] extension_buffer = RVulkan_Extensions();
    // -- create instance info --
    VkInstanceCreateInfo instinfo;
    instinfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instinfo.pNext = null; // reserved for possible future params
    instinfo.pApplicationInfo = &appinfo;
    instinfo.enabledExtensionCount = cast(uint)extension_buffer.length;
    instinfo.enabledLayerCount     = cast(uint)Validation_layers.length;
    instinfo.ppEnabledExtensionNames = extension_buffer.ptr;
    instinfo.ppEnabledLayerNames     = Validation_layers.ptr;

    enforceVK(vkCreateInstance(&instinfo, null, &vk_ctx.instance));
    loadInstanceLevelFunctions(vk_ctx.instance);
  }

  void Setup_Vk_Debug_Callback ( ) {
    // Setup debug callback info
    auto callback_info = new VkDebugReportCallbackCreateInfoEXT(
      VkStructureType.VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
      null,
      VkDebugReportFlagBitsEXT.VK_DEBUG_REPORT_ERROR_BIT_EXT |
      VkDebugReportFlagBitsEXT.VK_DEBUG_REPORT_WARNING_BIT_EXT |
      VkDebugReportFlagBitsEXT.VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT,
      &VkDebug_Report_Callback, null);
    // Create debug callback
    enforceVK(vkCreateDebugReportCallbackEXT(vk_ctx.instance, callback_info,
                                             null, &vk_ctx.debug_callback));
  }

  void Setup_Vk_Surface ( ) {
    enforceVK(glfwCreateWindowSurface(vk_ctx.instance, glfw_ctx.window, null,
                                     &vk_ctx.surface));
  }

  QueueFamilyIndices RQueue_Families ( VkPhysicalDevice device ) {
    QueueFamilyIndices indices;
    indices.Init();
    auto queue_family = RPhysical_Device_Queue_Family(device);
    // Find VK_QUEUE_GRAPHICS_BIT and check for support on device&surface
    foreach ( iter, queue; queue_family ) {
      if ( queue.queueCount == 0 ) continue;
      // check graphics bit
      if ( queue.queueFlags&VK_QUEUE_GRAPHICS_BIT ) {
        indices.Set_Index(QueueFamily.Gfx, iter);
      }
      // check transfer bit (on a different queue)
      if ( indices.RIndex(QueueFamily.Gfx) != iter ) {
        if ( queue.queueFlags&VK_QUEUE_TRANSFER_BIT ) {
          indices.Set_Index(QueueFamily.Transfer, iter);
        }
      }
      { // check present support
        VkBool32 present_support = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, cast(uint)iter,
                                   vk_ctx.surface, &present_support);
        if ( present_support ) { indices.Set_Index(QueueFamily.Present, iter); }
      }
      if ( indices.Is_Complete() ) break;
    }
    if ( indices.RIndex(QueueFamily.Transfer) == -1 ) { // only one queue family
      indices.Set_Index(QueueFamily.Transfer, indices.RIndex(QueueFamily.Gfx));
    }
    return indices;
  }

  /**
    Checks extension support for the given physical device
    Params:
      device = A physical device to check extension support for
      extensions = The required extensions that the device should support
    Return: If the device supports all required extensions
  **/
  bool Check_Device_Extension_Support(VkPhysicalDevice device,
                                      const(char*)[] extensions) {
    VkExtensionProperties[] device_extensions;
    device_extensions = VkRDeviceExtensionProperties(device, null);
    return Vk_Has_All!("extensionName")(extensions, device_extensions);
  }


  /**
    Return: A QueueFamilyIndices struct that returns true on Is_Complete
              if the device is suitable for this application.
  **/
  QueueFamilyIndices Is_Device_Suitable ( VkPhysicalDevice phydevice ) {
    QueueFamilyIndices failure;
    failure.Init();
    // Check extension support
    if ( !Check_Device_Extension_Support(phydevice, Device_extensions) ) {
      return failure;
    }
    // Check device features
    VkPhysicalDeviceFeatures supported_features;
    vkGetPhysicalDeviceFeatures(phydevice, &supported_features);
    if ( !supported_features.samplerAnisotropy ) return failure;
    // Check swap and chain
    auto swapchain_support = SwapchainSupportDetails(phydevice, vk_ctx.surface);
    if ( !swapchain_support.Sufficient() ) return failure;
    // Get family properties of the GPU
    auto device_properties = RPhysical_Device_Properties(phydevice);
    // Check for a discrete GPU
    return RQueue_Families(phydevice);
  }

  void Setup_Vk_Physical_Device ( ) {
    auto devices = RPhysical_Devices(vk_ctx.instance);
    foreach ( index, device; devices ) {
      auto indices = Is_Device_Suitable(device);
      if ( indices.Is_Complete() ) {
        vk_ctx.physical_device = device;
        vk_ctx.queue_family = indices;
        break;
      }
    }
    enforce(vk_ctx.physical_device != null, "Could not find device");
  }

  static immutable const(char*)[] Validation_layers = [
    "VK_LAYER_LUNARG_standard_validation\0"
  ];

  void Setup_Vk_Logical_Device ( ) {
    // get queue index
    auto gfx_idx = vk_ctx.queue_family.RIndex(QueueFamily.Gfx);
    // Create device queue info
    Set!uint unique_queue_families = [
      vk_ctx.queue_family.RIndex(QueueFamily.Gfx),
      vk_ctx.queue_family.RIndex(QueueFamily.Present)
    ];
    VkDeviceQueueCreateInfo[] queue_create_infos;
    float queue_priority = 1.0f;
    foreach ( queue_family; unique_queue_families ) {
      VkDeviceQueueCreateInfo queue_create_info;
      queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queue_create_info.queueFamilyIndex = queue_family;
      queue_create_info.queueCount = 1;
      queue_create_info.pQueuePriorities = &queue_priority;
      queue_create_infos ~= queue_create_info;
    }
    // Get device create info
    VkPhysicalDeviceFeatures features = VkPhysicalDeviceFeatures();
    features.samplerAnisotropy = VK_TRUE;
    features.shaderClipDistance = VK_TRUE;
    features.sampleRateShading = VK_TRUE;
    VkDeviceCreateInfo create_info;

    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = cast(uint)queue_create_infos.length;
    create_info.pQueueCreateInfos = queue_create_infos.ptr;
    create_info.pEnabledFeatures = &features;
    create_info.enabledExtensionCount = Device_extensions.length;
    create_info.ppEnabledExtensionNames = Device_extensions.ptr;

    // layers
    create_info.enabledLayerCount = Validation_layers.length;
    create_info.ppEnabledLayerNames = Validation_layers.ptr;
    // create device
    enforceVK(vkCreateDevice(vk_ctx.physical_device, &create_info, null,
                             &vk_ctx.device));
    // load vulkan functions for the device
    loadDeviceLevelFunctions(vk_ctx.device);
    // get relevant queues to submit command buffers
    vkGetDeviceQueue(vk_ctx.device, gfx_idx, 0, &vk_ctx.graphics_queue);
    vkGetDeviceQueue(vk_ctx.device, gfx_idx, 0, &vk_ctx.present_queue);
  }

  void Setup_Swapchain ( ) {
    // get window width/height
    int width, height;
    glfwGetWindowSize(glfw_ctx.window, &width, &height);
    // Get swapchain details
    SwapchainSupportDetails swap = SwapchainSupportDetails(
                    vk_ctx.physical_device, vk_ctx.surface);
    SwapchainDetailsPreCreation details = swap.RDetails(width, height);
    // -- setup swap chain create info --
    VkSwapchainCreateInfoKHR create_info = swap.RSwapchain_Create_Info();
    // -- fill in swapchain create info --
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.imageFormat = details.surface_format.format;
    create_info.imageColorSpace = details.surface_format.colorSpace;
    create_info.imageExtent = details.extent;
    create_info.presentMode = details.present_mode;
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = vk_ctx.surface;
    create_info.imageArrayLayers = 1;
    // Render directly to image
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    // Check which imageSharingMode we should use
    auto gfx_idx = vk_ctx.queue_family.RIndex(QueueFamily.Gfx),
         pre_idx = vk_ctx.queue_family.RIndex(QueueFamily.Present);
    uint[] queue_family_indices = [gfx_idx, pre_idx];
    if ( gfx_idx != pre_idx ) {
      // Images can be shared across multiple queue families without explicit
      // ownership transfer
      create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      create_info.queueFamilyIndexCount = cast(uint)queue_family_indices.length;
      create_info.pQueueFamilyIndices = queue_family_indices.ptr;
    } else {
      // Image is owned by one queue family and ownership must be transfered
      // explicitly before using it in another queue family. Best performance
      create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
      create_info.queueFamilyIndexCount = 0;
      create_info.pQueueFamilyIndices = null;
    }
    // Ignore blending with other windows in the window manager
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.clipped = VK_TRUE; // Don't care about obscured pixels
    create_info.oldSwapchain = VK_NULL_HANDLE;
    // create & store swap chain info
    enforceVK(vkCreateSwapchainKHR(vk_ctx.device, &create_info, null,
                                  &vk_ctx.swapchain.swapchain));
    vk_ctx.swapchain.images = vkRSwapchainImagesKHR(vk_ctx.device,
                                                    vk_ctx.swapchain.swapchain);
    vk_ctx.swapchain.image_format = details.surface_format.format;
    vk_ctx.swapchain.extent = details.extent;
  }

  void Setup_Image_Views ( ) {
    // allocate buffers
    vk_ctx.swapchain.image_views.length = vk_ctx.swapchain.images.length;
    // Create views into the image
    foreach ( iter, img; vk_ctx.swapchain.images ) {
      vk_ctx.swapchain.image_views[iter] = Create_Image_View(img,
                                   vk_ctx.swapchain.image_format);
    }
  }

  VkShaderModule Create_Shader_Module ( inout ubyte[] code ) {
    VkShaderModuleCreateInfo info;
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = code.length;
    info.pCode = cast(uint*)code;
    VkShaderModule shader_module;
    enforceVK(vkCreateShaderModule(vk_ctx.device, &info, null, &shader_module));
    return shader_module;
  }

  VkShaderModule Create_Shader_Module ( string filename ) {
    return Create_Shader_Module(Read_Spirv(filename));
  }

  void Setup_Render_Pass ( ) {
    // -- create render attachments --

    // -- create colour attachment
    // Handles how the image/renders are handled
    VkAttachmentDescription colour_attachment;
    colour_attachment.format = vk_ctx.swapchain.image_format;
    colour_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colour_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colour_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colour_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colour_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colour_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colour_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    // -- create colour attachment reference --
    VkAttachmentReference colour_attachment_ref;
    colour_attachment_ref.attachment = 0; // (location = 0) out_colour
    colour_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // -- create depth attachment --
    VkAttachmentDescription depth_attachment;
    depth_attachment.format = Find_Depth_Format();
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    // -- create depth attachment reference --
    VkAttachmentReference depth_attachment_ref;
    depth_attachment_ref.attachment = 1; // (location = 1) out_depth
    depth_attachment_ref.layout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // -- create subpass description --
    // ei for multiple subpasses into a single render
    VkSubpassDescription subpass;
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colour_attachment_ref;
    subpass.pDepthStencilAttachment = &depth_attachment_ref;
    // -- subpass dependencies --
    VkSubpassDependency dependency;
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0; // value must be higher than src to avoid cycle
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    // -- construct render pass --
    VkAttachmentDescription[] attachments =
      [colour_attachment, depth_attachment];
    VkRenderPassCreateInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = cast(uint)attachments.length;
    render_pass_info.pAttachments = attachments.ptr;
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 1;
    render_pass_info.pDependencies = &dependency;
    enforceVK(vkCreateRenderPass(vk_ctx.device, &render_pass_info, null,
                                &vk_ctx.render_pass));
  }

  void Setup_Graphics_Pipeline ( ) {
    // -- wrap shaders in modules --
    VkShaderModule vert_shader_module, frag_shader_module;
    vert_shader_module = Create_Shader_Module("spirv-shaders/vert.spv");
    frag_shader_module = Create_Shader_Module("spirv-shaders/frag.spv");
    // -- create shader stage --
    VkPipelineShaderStageCreateInfo vert_stage_info;
    vert_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_stage_info._module = vert_shader_module;
    vert_stage_info.pName = "main";
    VkPipelineShaderStageCreateInfo frag_stage_info;
    frag_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_stage_info._module = frag_shader_module;
    frag_stage_info.pName = "main";
    VkPipelineShaderStageCreateInfo[] shader_stages;
    shader_stages = [ vert_stage_info, frag_stage_info ];

    // -- setup vertex pipeline --
    auto binding_description = Vertex.RBinding_Description();
    auto attribute_descriptions = Vertex.RAttribution_Description();
    VkPipelineVertexInputStateCreateInfo vertex_input_info;
    vertex_input_info.sType = // v JESUS CHRIST v
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_info.vertexBindingDescriptionCount = 1;
    vertex_input_info.pVertexBindingDescriptions = &binding_description;
    vertex_input_info.vertexAttributeDescriptionCount =
                                 cast(uint)attribute_descriptions.length;
    vertex_input_info.pVertexAttributeDescriptions = attribute_descriptions.ptr;
    // -- setup input assembly --
    VkPipelineInputAssemblyStateCreateInfo input_assembly;
    input_assembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly.primitiveRestartEnable = VK_FALSE;
    // -- setup viewport --
    VkViewport viewport;
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = cast(float)vk_ctx.swapchain.extent.width;
    viewport.height = cast(float)vk_ctx.swapchain.extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    // -- setup scissor --
    VkRect2D scissor;
    scissor.offset = VkOffset2D(0, 0);
    scissor.extent = vk_ctx.swapchain.extent;
    // -- combine into viewport --
    VkPipelineViewportStateCreateInfo viewport_state;
    viewport_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = &scissor;
    // -- rasterization create --
    VkPipelineRasterizationStateCreateInfo rasterizer;
    rasterizer.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;
    // -- multisampling create --
    VkPipelineMultisampleStateCreateInfo multisampling;
    multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = null;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;
    // -- depth and stencil buffers --
    VkPipelineDepthStencilStateCreateInfo depth_stencil_info;
    depth_stencil_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil_info.depthTestEnable = VK_TRUE;
    depth_stencil_info.depthWriteEnable = VK_TRUE;
    depth_stencil_info.depthCompareOp = VK_COMPARE_OP_LESS;
    depth_stencil_info.depthBoundsTestEnable = VK_FALSE; // no depth bounds
    depth_stencil_info.minDepthBounds = 0.0f;
    depth_stencil_info.maxDepthBounds = 1.0f;
    depth_stencil_info.stencilTestEnable = VK_FALSE; // no stencil test
    // -- colour blending --
    VkPipelineColorBlendAttachmentState colour_blend_attachment;
    colour_blend_attachment.colorWriteMask =
                  VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colour_blend_attachment.blendEnable = VK_FALSE; // disable blend i guess
    colour_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colour_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colour_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    colour_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colour_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colour_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    VkPipelineColorBlendStateCreateInfo colour_blending;
    colour_blending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colour_blending.logicOpEnable = VK_FALSE;
    colour_blending.logicOp = VK_LOGIC_OP_COPY;
    colour_blending.attachmentCount = 1;
    colour_blending.pAttachments = &colour_blend_attachment;
    colour_blending.blendConstants[0..3] = 0.0f;
    // -- dynamic state (skip) --
    // -- pipeline layout (uniforms) --
    VkPipelineLayoutCreateInfo pipeline_layout_info;
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &vk_ctx.descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 0;
    pipeline_layout_info.pPushConstantRanges = null;
    enforceVK(vkCreatePipelineLayout(vk_ctx.device, &pipeline_layout_info, null,
                                    &vk_ctx.pipeline_layout));
    // -- create graphics pipeline --
    VkGraphicsPipelineCreateInfo pipeline_info;
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = cast(uint)shader_stages.length;
    pipeline_info.pStages = shader_stages.ptr;
    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pDepthStencilState = &depth_stencil_info;
    pipeline_info.pColorBlendState = &colour_blending;
    pipeline_info.pDynamicState = null; // no dynamic state
    pipeline_info.layout = vk_ctx.pipeline_layout;
    pipeline_info.renderPass = vk_ctx.render_pass;
    pipeline_info.subpass = 0; // < index to subpass
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE; // no existing pipeline
    pipeline_info.basePipelineIndex = -1;
    enforceVK(vkCreateGraphicsPipelines(vk_ctx.device, VK_NULL_HANDLE, 1,
                                &pipeline_info, null, &vk_ctx.pipeline));
    // -- free shader modules --
    vkDestroyShaderModule(vk_ctx.device, frag_shader_module, null);
    vkDestroyShaderModule(vk_ctx.device, vert_shader_module, null);
  }

  void Setup_Framebuffers ( ) {
    vk_ctx.swapchain.framebuffers.length = vk_ctx.swapchain.image_views.length;
    // for each view create a framebuffer
    foreach ( iter, ref view; vk_ctx.swapchain.image_views ) {
      // make an attachment for the swapchain image view and depth image view
      // this is allowed since we use same depth image view for each swapchain
      VkImageView[] attachments = [ view, vk_ctx.depth_image_view ];

      VkFramebufferCreateInfo framebuffer_info;
      framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebuffer_info.renderPass = vk_ctx.render_pass;
      framebuffer_info.attachmentCount = cast(uint)attachments.length;
      framebuffer_info.pAttachments = attachments.ptr;
      framebuffer_info.width = vk_ctx.swapchain.extent.width;
      framebuffer_info.height = vk_ctx.swapchain.extent.height;
      framebuffer_info.layers = 1;

      enforceVK(vkCreateFramebuffer(vk_ctx.device, &framebuffer_info, null,
                                   &vk_ctx.swapchain.framebuffers[iter]));
    }
  }

  void Setup_Command_Pool ( ) {
    // Create command pool for graphics
    VkCommandPoolCreateInfo pool_info;
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = vk_ctx.queue_family.RIndex(QueueFamily.Gfx);
    // pool_info.flags = 0;
    enforceVK(vkCreateCommandPool(vk_ctx.device, &pool_info, null,
                                 &vk_ctx.command_pool));
  }

  void Create_Image(uint w, uint h, VkFormat format, VkImageTiling tiling,
                  VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
                  ref VkImage image, ref VkDeviceMemory image_memory) {
    // -- create image on device --
    VkImageCreateInfo image_info;
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent.width = w;
    image_info.extent.height = h;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = format;
    image_info.tiling = tiling;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage = usage;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.flags = 0;
    enforceVK(vkCreateImage(vk_ctx.device, &image_info, null, &image));
    // -- get memory requirements for image --
    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(vk_ctx.device, image, &mem_req);
    // -- allocate memory on device for texture image --
    VkMemoryAllocateInfo alloc_info;
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_req.size;
    alloc_info.memoryTypeIndex = Find_Memory_Type(mem_req.memoryTypeBits,
                                          properties);
    enforceVK(vkAllocateMemory(vk_ctx.device, &alloc_info, null,
                              &image_memory));
    // bind memory to buffer
    vkBindImageMemory(vk_ctx.device, image, image_memory, 0);
  }

  void Copy_Buffer_To_Image ( VkBuffer buffer, VkImage image, uint width,
                              uint height ) {
    VkCommandBuffer command_buffer = Begin_Single_Time_Commands();

    VkBufferImageCopy region;
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = VkOffset3D(0, 0, 0);
    region.imageExtent = VkExtent3D( width, height, 1);
    vkCmdCopyBufferToImage(command_buffer, buffer, image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    End_Single_Time_Commands(command_buffer);
  }

  void Setup_Texture_Image ( ) {
    // load image to host
    import imageformats;
    auto img = read_image("textures/Texture.png");
    // create staging buffer
    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;
    auto len = img.pixels.length;
    Create_Buffer(len, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging_buffer, staging_buffer_memory);
    // copy image to staging buffer
    void* data;
    vkMapMemory(vk_ctx.device, staging_buffer_memory, 0, len, 0, &data);
      memcpy(data, img.pixels.ptr, len);
    vkUnmapMemory(vk_ctx.device, staging_buffer_memory);
    // -- move staging buffer to image --
    Create_Image(img.w, img.h, VK_FORMAT_R8G8B8A8_UNORM,
                 VK_IMAGE_TILING_OPTIMAL,
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vk_ctx.texture_image, vk_ctx.texture_image_memory);
    // -- copy staging buffer to texture image --
    Transition_Image_Layout(vk_ctx.texture_image, VK_FORMAT_R8G8B8A8_UNORM,
           VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    Copy_Buffer_To_Image(staging_buffer, vk_ctx.texture_image, img.w, img.h);
    Transition_Image_Layout(vk_ctx.texture_image, VK_FORMAT_R8G8B8A8_UNORM,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    // -- cleanup
    vkDestroyBuffer(vk_ctx.device, staging_buffer, null);
    vkFreeMemory(vk_ctx.device, staging_buffer_memory, null);
  }

  VkImageView Create_Image_View(VkImage image,
                VkFormat format,
                VkImageAspectFlags aspect_flags = VK_IMAGE_ASPECT_COLOR_BIT) {
    VkImageViewCreateInfo view_info;
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.subresourceRange.aspectMask = aspect_flags;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    VkImageView image_view;
    enforceVK(vkCreateImageView(vk_ctx.device, &view_info, null, &image_view));
    return image_view;
  }

  void Setup_Texture_Image_View ( ) {
    // -- setup image view create --
    vk_ctx.texture_image_view = Create_Image_View(vk_ctx.texture_image,
                                              VK_FORMAT_R8G8B8A8_UNORM);
  }

  void Setup_Texture_Sampler ( ) {
    // -- setup sampler --
    VkSamplerCreateInfo sampler_info;
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.anisotropyEnable = VK_TRUE;
    sampler_info.maxAnisotropy = 16;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.mipLodBias = 0.0f;
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = 0.0f;
    enforceVK(vkCreateSampler(vk_ctx.device, &sampler_info, null,
                                       &vk_ctx.texture_sampler));
  }

  VkCommandBuffer Begin_Single_Time_Commands ( ) {
    // -- allocate command buffer --
    VkCommandBufferAllocateInfo alloc_info;
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = vk_ctx.command_pool;
    alloc_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer;
    enforceVK(vkAllocateCommandBuffers(vk_ctx.device, &alloc_info,
                                      &command_buffer));
    // -- begin command buffer --
    VkCommandBufferBeginInfo begin_info;
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command_buffer, &begin_info);
    return command_buffer;
  }

  void End_Single_Time_Commands(VkCommandBuffer command_buffer) {
    vkEndCommandBuffer(command_buffer);
    // -- submit info --
    VkSubmitInfo submit_info;
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    vkQueueSubmit(vk_ctx.graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(vk_ctx.graphics_queue);
    // free command
    vkFreeCommandBuffers(vk_ctx.device, vk_ctx.command_pool, 1,
                         &command_buffer);
  }

  void Setup_Command_Buffers() {
    // Allocate command buffers so we can record drawing commands to them
    // Since one of the commands is to bind to a frame buffer, we have to
    // record a command buffer for every image in the swap chain
    vk_ctx.command_buffers.length = vk_ctx.swapchain.framebuffers.length;
    // -- create command buffer alloc info --
    VkCommandBufferAllocateInfo alloc_info;
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = vk_ctx.command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = cast(uint)vk_ctx.command_buffers.length;
    enforceVK(vkAllocateCommandBuffers(vk_ctx.device, &alloc_info,
                                       vk_ctx.command_buffers.ptr));
    foreach ( iter, ref cmd_buffer; vk_ctx.command_buffers ) {
      // -- start command buffer recording --
      VkCommandBufferBeginInfo begin_info;
      begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
      begin_info.pInheritanceInfo = null;
      vkBeginCommandBuffer(cmd_buffer, &begin_info);
      // -- start a render pass --
      VkRenderPassBeginInfo render_pass_info;
      render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      render_pass_info.renderPass = vk_ctx.render_pass;
      render_pass_info.framebuffer = vk_ctx.swapchain.framebuffers[iter];
      render_pass_info.renderArea.offset = VkOffset2D(0, 0);
      render_pass_info.renderArea.extent = vk_ctx.swapchain.extent;
      // -- set clear for colour and depth --
      VkClearValue[] clear_values; clear_values.length = 2;
      clear_values[0].color.float32 = [0.0f, 0.0f, 0.0f, 1.0f];
      clear_values[1].depthStencil = VkClearDepthStencilValue(1.0f, 0);
      render_pass_info.clearValueCount = cast(uint)clear_values.length;
      render_pass_info.pClearValues = clear_values.ptr;
      // -- set clear depth 
      // -- apply render pass and draw --
      vkCmdBeginRenderPass(cmd_buffer, &render_pass_info,
                           VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          vk_ctx.pipeline);
        // -- bind vertex buffer --
        VkBuffer[] vertex_buffers = [ vk_ctx.vertex_buffer ];
        VkDeviceSize[] offsets = [0];
        vkCmdBindVertexBuffers(cmd_buffer, 0, 1, vertex_buffers.ptr,
                              offsets.ptr);
        // -- bind index buffer --
        vkCmdBindIndexBuffer(cmd_buffer, vk_ctx.index_buffer, 0,
                             VK_INDEX_TYPE_UINT32);
        // -- bind descriptor set --
        vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                vk_ctx.pipeline_layout, 0, 1,
                                &vk_ctx.descriptor_set, 0, null);
        // -- draw --
        vkCmdDrawIndexed(cmd_buffer, cast(uint)indices.length, 1, 0, 0, 0);
      vkCmdEndRenderPass(cmd_buffer);
      enforceVK(vkEndCommandBuffer(cmd_buffer));
    }
  }

  void Setup_Semaphores ( ) {
    VkSemaphoreCreateInfo semaphore_info;
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    enforceVK(vkCreateSemaphore(vk_ctx.device, &semaphore_info, null,
                               &vk_ctx.image_available_semaphore));
    enforceVK(vkCreateSemaphore(vk_ctx.device, &semaphore_info, null,
                               &vk_ctx.render_finished_semaphore));
  }

  /**
    Finds right type of memory to use
  **/
  uint Find_Memory_Type ( uint type_filter, VkMemoryPropertyFlags properties ) {
    // -- get info about available types of memory --
    VkPhysicalDeviceMemoryProperties mem_prop;
    vkGetPhysicalDeviceMemoryProperties(vk_ctx.physical_device, &mem_prop);
    // iterate bitfield and find valid memory index (typeField bit = 1, correct
    // properties)
    foreach ( i; 0 .. mem_prop.memoryTypeCount ) {
      auto property_flag = (mem_prop.memoryTypes[i].propertyFlags & properties);
      if ( type_filter&(1<<i) && property_flag ) return i;
    }
    enforce(false, "Failed to find suitable memory type");
    assert(false);
  }

  /** Takes a list of formats and returns first one that is supported **/
  VkFormat Find_Supported_Format(inout VkFormat[] candidates,
                        VkImageTiling tiling, VkFormatFeatureFlags features) {
    foreach ( format; candidates ) {
      VkFormatProperties props;
      vkGetPhysicalDeviceFormatProperties(vk_ctx.physical_device, format,
                                          &props);
      switch ( tiling ) {
        default: break;
        case VK_IMAGE_TILING_LINEAR:
          if ( (props.linearTilingFeatures&features) == features )
            return format;
        break;
        case VK_IMAGE_TILING_OPTIMAL:
          if ( (props.optimalTilingFeatures&features) == features )
            return format;
        break;
      }
    }
    throw new Exception("Could not find a supported VkFormat");
  }

  /** Finds a depth format **/
  VkFormat Find_Depth_Format() {
    return Find_Supported_Format([VK_FORMAT_D32_SFLOAT,
             VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT],
             VK_IMAGE_TILING_OPTIMAL,
             VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
  }

  bool Has_Stencil_Component(VkFormat format) {
    // check for 8 bit stencil component S8
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
           format == VK_FORMAT_D24_UNORM_S8_UINT;
  }

  void Create_Buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, ref VkBuffer buffer,
                      ref VkDeviceMemory buffer_memory) {
    // -- create buffer --
    VkBufferCreateInfo buffer_info;
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    enforceVK(vkCreateBuffer(vk_ctx.device, &buffer_info, null, &buffer));
    // -- get memory requirements --
    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(vk_ctx.device, buffer, &mem_req);
    // -- allocate memory --
    VkMemoryAllocateInfo alocinfo;
    alocinfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alocinfo.allocationSize = mem_req.size;
    alocinfo.memoryTypeIndex = Find_Memory_Type(mem_req.memoryTypeBits,
                                                  properties);
    enforceVK(vkAllocateMemory(vk_ctx.device, &alocinfo, null, &buffer_memory));
    // -- bind memory to buffer --
    vkBindBufferMemory(vk_ctx.device, buffer, buffer_memory, 0);
  }

  void Setup_Vertex_Buffers ( ) {
    // -- create staging buffer --
    VkDeviceSize buffer_size = vertices[0].sizeof * vertices.length;
    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;
    Create_Buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer,
          staging_buffer_memory);
    // -- copy memory to staging buffer --
    void* data;
    vkMapMemory(vk_ctx.device, staging_buffer_memory, 0, buffer_size, 0, &data);
      memcpy(data, vertices.ptr, buffer_size);
    vkUnmapMemory(vk_ctx.device, staging_buffer_memory);
    // -- create vertex buffer & copy memory over --
    Create_Buffer(buffer_size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vk_ctx.vertex_buffer,
        vk_ctx.vertex_buffer_memory);
    Copy_Buffer(staging_buffer, vk_ctx.vertex_buffer, buffer_size);
    // -- free memory --
    vkDestroyBuffer(vk_ctx.device, staging_buffer, null);
    vkFreeMemory(vk_ctx.device, staging_buffer_memory, null);
  }

  void Setup_Index_Buffers ( ) {
    // -- create staging buffer --
    VkDeviceSize buffer_size = indices[0].sizeof * indices.length;
    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;
    Create_Buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer,
                staging_buffer_memory);
    // -- copy memory to staging buffer --
    void* data;
    vkMapMemory(vk_ctx.device, staging_buffer_memory, 0, buffer_size, 0, &data);
      memcpy(data, indices.ptr, buffer_size);
    vkUnmapMemory(vk_ctx.device, staging_buffer_memory);
    // -- create index buffer & copy memory over --
    Create_Buffer(buffer_size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vk_ctx.index_buffer,
        vk_ctx.index_buffer_memory);
    Copy_Buffer(staging_buffer, vk_ctx.index_buffer, buffer_size);
    // -- free memory --
    vkDestroyBuffer(vk_ctx.device, staging_buffer, null);
    vkFreeMemory(vk_ctx.device, staging_buffer_memory, null);
  }

  void Copy_Buffer ( VkBuffer src_buf, VkBuffer dst_buf, VkDeviceSize size ) {
    VkCommandBuffer command_buffer = Begin_Single_Time_Commands();
      VkBufferCopy copy_region;
      copy_region.srcOffset = copy_region.dstOffset = 0;
      copy_region.size = size;
      vkCmdCopyBuffer(command_buffer, src_buf, dst_buf, 1, &copy_region);
    End_Single_Time_Commands(command_buffer);
  }

  void Transition_Image_Layout(VkImage image, VkFormat format,
                        VkImageLayout old_layout, VkImageLayout new_layout) {
    VkCommandBuffer command_buffer = Begin_Single_Time_Commands();

    // -- initialize memory barrier for pipeline
    VkImageMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    // -- pipeline src/dst mask
    VkPipelineStageFlags source_stage;
    VkPipelineStageFlags destination_stage;
    // -- apply src -> dst layout --
    if ( old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
         new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ) {
      // set destination mask :: top of pipe -> transfer write stage
      // transfer writes that don't need to wait
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if ( old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ) {
      // :: transfer write stage -> fragment read stage
      // shader reads should wait on transfer writes
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if ( old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
                new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL){
      // :: top of pipe -> early fragment stage
      // shader should be read from to check for visible fragment
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      destination_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    } else {
      assert(false, "Unsupported layout transition");
    }
    vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage,
                         0, 0, null, 0, null, 1, &barrier);

    End_Single_Time_Commands(command_buffer);
  }

  void Setup_Descriptor_Set_Layout ( ) {
    // -- setup descriptor layout binding --
    VkDescriptorSetLayoutBinding ubo_binding;
    ubo_binding.binding = 0;
    ubo_binding.descriptorCount = 1;
    ubo_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ubo_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    ubo_binding.pImmutableSamplers = null;
    // -- setup sampler layout binding --
    VkDescriptorSetLayoutBinding sampler_binding;
    sampler_binding.binding = 1;
    sampler_binding.descriptorCount = 1;
    sampler_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sampler_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    sampler_binding.pImmutableSamplers = null;
    // -- combine bindings into array --
    VkDescriptorSetLayoutBinding[] bindings = [ubo_binding, sampler_binding];
    // -- create descriptor layout --
    VkDescriptorSetLayoutCreateInfo layout_info;
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = cast(int)bindings.length;
    layout_info.pBindings = bindings.ptr;
    enforceVK(vkCreateDescriptorSetLayout(vk_ctx.device, &layout_info, null,
                                         &vk_ctx.descriptor_set_layout));
  }

  void Setup_Uniform_Buffer ( ) {
    // -- create data --
    VkDeviceSize buffer_size = UniformBufferObject.sizeof;
    Create_Buffer(buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vk_ctx.uniform_buffer, vk_ctx.uniform_buffer_memory);
    // buffer is updated every frame, don't copy anything over yet
  }

  void Setup_Descriptor_Pool ( ) {
    // -- setup the size of descriptors for descriptor pool --
    VkDescriptorPoolSize[] pool_sizes; pool_sizes.length = 2;
    // -- create uniform buffer pool --
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[0].descriptorCount = 1;
    // -- create combined image sampler pool --
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[1].descriptorCount = 1;
    // -- create descriptor pool --
    VkDescriptorPoolCreateInfo pool_info;
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = cast(int)pool_sizes.length;
    pool_info.pPoolSizes = pool_sizes.ptr;
    pool_info.maxSets = 1;
    pool_info.flags = 0; // optionally set it to be freed
    enforceVK(vkCreateDescriptorPool(vk_ctx.device, &pool_info, null,
                                    &vk_ctx.descriptor_pool));
  }

  void Setup_Descriptor_Set ( ) {
    // -- allocate descriptor set --
    VkDescriptorSetLayout[] layouts = [ vk_ctx.descriptor_set_layout ];
    VkDescriptorSetAllocateInfo alloc_info;
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = vk_ctx.descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = layouts.ptr;
    enforceVK(vkAllocateDescriptorSets(vk_ctx.device, &alloc_info,
                                      &vk_ctx.descriptor_set));
    // -- setup a descriptor buffer --
    VkDescriptorBufferInfo buffer_info;
    buffer_info.buffer = vk_ctx.uniform_buffer;
    buffer_info.offset = 0;
    buffer_info.range = UniformBufferObject.sizeof;
    // -- setup descriptor image to bind image and sampler --
    VkDescriptorImageInfo image_info;
    image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_info.imageView = vk_ctx.texture_image_view;
    image_info.sampler = vk_ctx.texture_sampler;
    // -- setup descriptor buffer to be written to --
    VkWriteDescriptorSet[] descriptor_writes; descriptor_writes.length = 2;
    // -- setup uniform buffer descriptor write --
    descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[0].dstSet = vk_ctx.descriptor_set;
    descriptor_writes[0].dstBinding = 0;
    descriptor_writes[0].dstArrayElement = 0;
    descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptor_writes[0].descriptorCount = 1;
    descriptor_writes[0].pBufferInfo = &buffer_info;
    // -- setup combined image sampler to be written to --
    descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[1].dstSet = vk_ctx.descriptor_set;
    descriptor_writes[1].dstBinding = 1;
    descriptor_writes[1].dstArrayElement = 0;
    descriptor_writes[1].descriptorType =
                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptor_writes[1].descriptorCount = 1;
    descriptor_writes[1].pImageInfo = &image_info;
    // update descriptor write to the set
    vkUpdateDescriptorSets(vk_ctx.device, cast(int)descriptor_writes.length,
                           descriptor_writes.ptr, 0, null);
  }

  void Setup_Depth_Resources ( ) {
    // -- find proper depth format & create image from it --
    VkFormat depth_format = Find_Depth_Format();
    Create_Image(vk_ctx.swapchain.extent.width,
          vk_ctx.swapchain.extent.height,
          depth_format,
          VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          vk_ctx.depth_image, vk_ctx.depth_image_memory);
    // -- create image view with depth bit --
    vk_ctx.depth_image_view = Create_Image_View(vk_ctx.depth_image,
                                         depth_format,
                                         VK_IMAGE_ASPECT_DEPTH_BIT);
    // -- setup transition pipeline barrier --
    // undefined (we don't care about initial image contents) ->
    // depth/stencil attachment
    Transition_Image_Layout(vk_ctx.depth_image, depth_format,
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  }

  void Initialize_Vulkan ( ){
    writeln("Loading derelict-erupted Vulkan bindings");
    DerelictErupted.load();
    // Create vulkan instance
    writeln("Setup_Vk_Instance");           Setup_Vk_Instance();
    writeln("Setup_Vk_Debug_Callback");     Setup_Vk_Debug_Callback();
    writeln("Setup_Vk_Surface");            Setup_Vk_Surface();
    writeln("Setup_Vk_Physical_Device");    Setup_Vk_Physical_Device();
    writeln("Setup_Vk_Logical_Device");     Setup_Vk_Logical_Device();
    writeln("Setup_Swapchain");             Setup_Swapchain();
    writeln("Setup_Image_Views");           Setup_Image_Views();
    writeln("Setup_Render_Pass");           Setup_Render_Pass();
    writeln("Setup_Descriptor_Set_Layout"); Setup_Descriptor_Set_Layout();
    writeln("Setup_Graphics_Pipeline");     Setup_Graphics_Pipeline();
    writeln("Setup_Command_Pool");          Setup_Command_Pool();
    writeln("Setup Depth Resources");       Setup_Depth_Resources();
    writeln("Setup_Framebuffers");          Setup_Framebuffers();
    writeln("Setup Texture Image");         Setup_Texture_Image();
    writeln("Setup Texture Image View");    Setup_Texture_Image_View();
    writeln("Setup Texture Sampler");       Setup_Texture_Sampler();
    writeln("Loading model");               Load_Model();
    writeln("Setup_Vertex_Buffers");        Setup_Vertex_Buffers();
    writeln("Setup_Index_Buffers");         Setup_Index_Buffers();
    writeln("Setup_Uniform_Buffer");        Setup_Uniform_Buffer();
    writeln("Setup_Descriptor_Pool");       Setup_Descriptor_Pool();
    writeln("Setup_Descriptor_Set");        Setup_Descriptor_Set();
    writeln("Setup_Command_Buffers");       Setup_Command_Buffers();
    writeln("Setup_Semaphores");            Setup_Semaphores();

    writeln("Vulkan setup finished");
  }

  void Initialize_GLFW ( ) {
    // -- load derelict --
    DerelictGLFW3.load();
    DerelictGLFW3_loadVulkan();
    glfwInit();
    assert(glfwVulkanSupported(), "Vulkan not supported on this device");
    // -- create window --
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);
    glfw_ctx.window = glfwCreateWindow(800, 600, "Vulkan", null, null);
    // -- set callbacks --
    glfwSetWindowSizeCallback(glfw_ctx.window, &On_Window_Resize);
    // -- load funcs --
    loadGlobalLevelFunctions(cast(typeof(vkGetInstanceProcAddr))
      glfwGetInstanceProcAddress(null, "vkGetInstanceProcAddr"));
  }

  float4x4 Perspective(float fovy, float aspect, float znear, float zfar) {
    float tan_half_fovy = tan(fovy/2.0f);
    float4x4 result = float4x4.identity;
    result[0][0] = 1.0f/(aspect*tan_half_fovy);
    result[1][1] = 1.0f/(tan_half_fovy);
    result[2][2] = (zfar+znear)/(zfar - znear);
    result[2][3] = 1.0f;
    result[3][2] = -(2.0f*zfar*znear)/(zfar-znear);
    return float4x4(
      1.0f, 0.0f, 0.0f, 0.0f,
      0.0f, -1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.5f, 0.5f,
      0.0f, 0.0f, 0.0f, 1.0f
    )*result;
  }

  void Update_Uniform_Buffer ( ) {
    float time = cast(float)glfwGetTime();
    // -- setup model/view/projection matrix --
    UniformBufferObject ubo;
    auto aspect = cast(float)vk_ctx.swapchain.extent.width/
                  cast(float)vk_ctx.swapchain.extent.height;
    ubo.model = float4x4.identity.rotate(time*0.2f, float3(0.0f, 0.0f, 1.0f));
    ubo.view = float4x4.look_at(float3(2.0f, 0.0f, 2.0f), float3(0.0f), float3(0.0f, 1f, 0f));
    ubo.proj = Perspective(radians(90.0f), aspect, 0.1f, 10.0f);
    // -- copy memory over --
    void* data;
    vkMapMemory(vk_ctx.device, vk_ctx.uniform_buffer_memory, 0, ubo.sizeof,
                0, &data);
      memcpy(data, &ubo, ubo.sizeof);
    vkUnmapMemory(vk_ctx.device, vk_ctx.uniform_buffer_memory);
  }

  void Loop ( ) {
    auto st = glfwGetTime();
    while ( !glfwWindowShouldClose(glfw_ctx.window) &&
             glfwGetKey(glfw_ctx.window, GLFW_KEY_ESCAPE) != GLFW_PRESS ) {
      glfwPollEvents();
      Update_Uniform_Buffer();
      Draw_Frame();
      // writeln((glfwGetTime()-st)*1000, " MS");
      st=glfwGetTime();
      import core.thread;
      import core.time;
      Thread.sleep(dur!("msecs")(2));
    }

    vkDeviceWaitIdle(vk_ctx.device);
  }

  void Draw_Frame ( ) {
    // explicit sync
    // vkQueueWaitIdle(vk_ctx.present_queue);
    uint image_index;
    // Get next image in the swap chain
    VkResult result = vkAcquireNextImageKHR(vk_ctx.device,
              vk_ctx.swapchain.swapchain, ulong.max,
              vk_ctx.image_available_semaphore, VK_NULL_HANDLE, &image_index);
    if ( result == VK_ERROR_OUT_OF_DATE_KHR ) {
      Recreate_Swapchain();
      return;
    }
    enforce(result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR);
    // -- Submit command buffer --
    VkSemaphore[] wait_semaphores = [ vk_ctx.image_available_semaphore ];
    VkSemaphore[] signal_semaphores = [ vk_ctx.image_available_semaphore ];
    VkPipelineStageFlags[] wait_stages = [
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
    ];
    VkSubmitInfo submit_info;
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.waitSemaphoreCount = cast(uint)wait_semaphores.length;
    submit_info.pWaitSemaphores = wait_semaphores.ptr;
    submit_info.pWaitDstStageMask = wait_stages.ptr;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &vk_ctx.command_buffers[image_index];
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores.ptr;
    enforceVK(vkQueueSubmit(vk_ctx.graphics_queue, 1, &submit_info,
                            VK_NULL_HANDLE));
    // -- Present to screen --
    VkSwapchainKHR[] swapchains = [ vk_ctx.swapchain.swapchain ];
    VkPresentInfoKHR present_info;
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = signal_semaphores.ptr;
    present_info.swapchainCount = cast(uint)swapchains.length;
    present_info.pSwapchains = swapchains.ptr;
    present_info.pImageIndices = &image_index;
    present_info.pResults = null;
    enforceVK(vkQueuePresentKHR(vk_ctx.present_queue, &present_info));
    // explict sync
    vkQueueWaitIdle(vk_ctx.present_queue);
  }

  void Cleanup ( ) {
    vk_ctx.Cleanup();
    glfwDestroyWindow(glfw_ctx.window);
    glfwTerminate();
  }
public:
  void Run ( ) {
    Initialize_GLFW();
    Initialize_Vulkan();
    Loop();
    Cleanup();
  }

  void Recreate_Swapchain ( ) {
    // clear memory
    vk_ctx.Cleanup_Swapchain;
    // Check if window width/height is 0, ei if it's minimized
    int width, height;
    glfwGetWindowSize(glfw_ctx.window, &width, &height);
    if ( width == 0 || height == 0 ) return;
    // Wait for device to idle so we can swap the swapchain
    vkDeviceWaitIdle(vk_ctx.device);
    // Replace swapchain along with everything that is affected by it.
    Setup_Swapchain();
    Setup_Image_Views();
    Setup_Render_Pass();
    Setup_Graphics_Pipeline(); // TODO use dynamic state for viewport/scissors
    Setup_Depth_Resources();
    Setup_Framebuffers();
    Setup_Command_Buffers();
  }
}

TriangleApplication application;

void main ( ) {
  application = new TriangleApplication();

  try {
    application.Run();
  } catch ( Exception e ) {
    writeln("Caught exception: ", e);
  }
}

extern(C) void On_Window_Resize(GLFWwindow* window, int W, int H) nothrow {
  try {application.Recreate_Swapchain();} catch (Exception E) {assert(0);}
}
