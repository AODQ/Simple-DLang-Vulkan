
import std.stdio;
import std.range;
import std.array;
import std.algorithm;
import std.exception;
import std.conv;
import std.string;

import erupted;
import vector;
import gfm.math;
import derelict.sdl2.sdl;

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
  printf("ObjectType: %i\n", object_type);
  printf(pMessage);
  return VK_FALSE;
}

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

enum QueueFamily { Graphics };
struct QueueFamilyIndices {
  private uint[QueueFamily.max+1] families;

  bool Set_Family ( QueueFamily family, uint index, VkPhysicalDevice device,
                    VkSurfaceKHR surface ) {
    VkBool32 present;
    writeln("device: ", device);
    writeln(index, " ", surface);
    writeln("func: ", vkGetPhysicalDeviceSurfaceSupportKHR);
    vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface, &present);
    writeln("hm");
    if ( present )
      families[cast(uint)family] = index;
    return cast(bool)present;
  }

  bool Set_Family ( QueueFamily family, size_t index, VkPhysicalDevice device,
                    VkSurfaceKHR surface ) {
    return Set_Family(family, cast(uint)index, device, surface);
  }

  auto RFamily ( QueueFamily family ) { return families[cast(uint)family]; }

  bool Is_Complete ( ) {
    return families[cast(uint)QueueFamily.Graphics] >= 0;
  }
}

struct VkContext {
  VkInstance instance;
  VkDebugReportCallbackEXT debug_callback;
  VkSurfaceKHR surface;
  VkPhysicalDevice physical_device;
  VkDevice logical_device;

  QueueFamilyIndices queue_family_indices;

  VkQueue graphics_queue;
  uint width = -1, height = -1;
  VkSwapchainKHR swapchain;
  VkCommandBuffer setup_cmd_buffer, draw_cmd_buffer;
  VkImage[] present_images;
  VkImage depth_image;
  VkPhysicalDeviceMemoryProperties memory_properties;
  VkImageView depth_image_view;
  VkRenderPass render_pass;
  VkFramebuffer[] frame_buffers;
  VkBuffer vertex_input_buffer;
  VkPipelineLayout pipeline_layout;
  VkPipeline pipeline;
  VkSemaphore present_complete_semaphore, rendering_complete_semaphore;
}

struct SDLContext {
  SDL_Window* window;
  SDL_SysWMinfo window_info;
}

class TriangleApplication {
private:
  SDLContext sdl_context;

  VkContext vk_ctx;

  bool Vk_Has_All(string prop, Range)(const(char*)[] elem, Range vk) {
    import core.stdc.string;
    mixin(q{
      return elem[].all!(name =>
        vk[].any!(ext =>
          strcmp(cast(const(char*))ext.%s, name) == 0));
    }.format(prop));
  }

  const(char*)[] RVulkan_Extensions ( ) {
    const(char*)[] extension_names = [
      "VK_KHR_surface",
      "VK_KHR_xlib_surface",
      "VK_EXT_debug_report"
    ];
    // validate that they exist
    auto extension_properties = RInstance_Extension_Properties();
    enforce(Vk_Has_All!("extensionName")(extension_names, extension_properties),
           "Missing extensions");
    return extension_names;
  }

  const(char*)[] RVulkan_Layers ( ) {
    const(char*)[] layer_names = [
      "VK_LAYER_LUNARG_standard_validation"
    ];
    // validate that they exist
    auto layer_properties = RInstance_Layer_Properties();
    enforce(Vk_Has_All!("layerName")(layer_names, layer_properties),
           "Missing layers");
    return layer_names;
  }

  void Setup_Vk_Instance ( VkApplicationInfo* appinfo,
                            VkInstanceCreateInfo* instinfo ) {
    // -- create application info --
    appinfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appinfo.pNext = null;
    appinfo.pApplicationName = "Hello Triangle";
    appinfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appinfo.pEngineName = "No engine";
    appinfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appinfo.apiVersion = VK_API_VERSION_1_0;
    // -- get extension and layer information --
    const(char*)[] extension_buffer = RVulkan_Extensions(),
                   layer_buffer     = RVulkan_Layers();
    // -- create instance info --
    instinfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instinfo.pNext = null; // reserved for possible future params
    instinfo.pApplicationInfo = appinfo;
    instinfo.enabledExtensionCount = cast(uint)extension_buffer.length;
    instinfo.enabledLayerCount     = cast(uint)layer_buffer.length;
    instinfo.ppEnabledExtensionNames = extension_buffer.ptr;
    instinfo.ppEnabledLayerNames     = layer_buffer.ptr;

    enforceVK(vkCreateInstance(instinfo, null, &vk_ctx.instance));
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
    // Create X11 window
    auto xlib_info = new VkXlibSurfaceCreateInfoKHR(
            VkStructureType.VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
            null,
            0,
            sdl_context.window_info.info.x11.display,
            sdl_context.window_info.info.x11.window);
    enforceVK(vkCreateXlibSurfaceKHR(vk_ctx.instance, xlib_info, null,
                                     &vk_ctx.surface));
    writeln("CREATED SURFACE: ", vk_ctx.surface);
  }

  QueueFamilyIndices RQueue_Families ( VkPhysicalDevice device ) {
    QueueFamilyIndices indices;
    auto queue_family = RPhysical_Device_Queue_Family(device);
    writeln(queue_family);
    // Find VK_QUEUE_GRAPHICS_BIT and check for support on device&surface
    foreach ( iter, queue; queue_family ) {
      iter.writeln;
      queue.queueCount.writeln;
      queue.queueFlags.writeln;
      if ( queue.queueCount > 0 && queue.queueFlags&VK_QUEUE_GRAPHICS_BIT ) {
        writeln("set family..");
        bool success = indices.Set_Family(QueueFamily.Graphics, iter, device,
                                          vk_ctx.surface);
        success.writeln;
        if ( success ) break;
      }
    }
    return indices;
  }

  /**
    Return: A QueueFamilyIndices struct that returns true on Is_Complete
              if the device is suitable for this application.
  **/
  QueueFamilyIndices Is_Device_Suitable ( VkPhysicalDevice device ) {
    // Get family properties of the GPU
    auto device_properties = RPhysical_Device_Properties(device);
    // Check for a discrete GPU
    writeln(device_properties.deviceType);
    return RQueue_Families(device);
    // if ( device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
         // device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
      // return RQueue_Families(device);
    // return QueueFamilyIndices();// default values -1
  }

  void Setup_Vk_Physical_Device ( ) {
    auto devices = RPhysical_Devices(vk_ctx.instance);
    foreach ( index, device; devices ) {
      auto indices = Is_Device_Suitable(device);
      writeln("is complete?");
      if ( indices.Is_Complete() ) {
        writeln("yes");
        vk_ctx.physical_device = device;
        vk_ctx.queue_family_indices = indices;
        writeln("IDX: ", indices.RFamily(QueueFamily.Graphics));
        break;
      }
    }
    enforce(vk_ctx.physical_device != null, "Could not find device");
  }

  void Setup_Vk_Logical_Device ( ) {
    // get queue index
    auto gfx_idx = vk_ctx.queue_family_indices.RFamily(QueueFamily.Graphics);
    // Create device queue info
    VkDeviceQueueCreateInfo* queue_create_info = new VkDeviceQueueCreateInfo();
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    writeln("IDX: ", gfx_idx);
    queue_create_info.queueFamilyIndex = gfx_idx;
    queue_create_info.queueCount = 1;
    float* priority = new float(1.0f);
    queue_create_info.pQueuePriorities = priority; // max priority
    // Get device features
    static const(char*)[] device_extensions = ["VK_KHR_swapchain"];
    // Get device create info
    VkPhysicalDeviceFeatures* features = new VkPhysicalDeviceFeatures();
    features.shaderClipDistance = VK_TRUE;
    VkDeviceCreateInfo* create_info = new VkDeviceCreateInfo;
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.pQueueCreateInfos = queue_create_info;
    create_info.queueCreateInfoCount = 1;
    create_info.pEnabledFeatures = features;
    // layers
    create_info.enabledLayerCount = 1;
    create_info.ppEnabledLayerNames = device_extensions.ptr;
    // create device
    enforceVK(vkCreateDevice(vk_ctx.physical_device, create_info, null,
                             &vk_ctx.logical_device));
    // vkGetDeviceQueue(vk_ctx.logical_device, gfx_idx, 0, &vk_ctx.graphics_queue);
  }

  void Initialize_Vulkan ( ){
    DerelictErupted.load();
    // Create vulkan instance
    Setup_Vk_Instance(new VkApplicationInfo(), new VkInstanceCreateInfo());
    Setup_Vk_Debug_Callback();
    Setup_Vk_Surface();
    Setup_Vk_Physical_Device();
    Setup_Vk_Logical_Device();
    writeln("Vulkan setup finished");
  }

  void Initialize_SDL ( ) {
    DerelictSDL2.load();
    sdl_context.window = SDL_CreateWindow("Vulkan", 0, 0, 800, 600, 0);
    SDL_VERSION(&sdl_context.window_info.version_);
    enforce(SDL_GetWindowWMInfo(sdl_context.window, &sdl_context.window_info),
            "Could not get SDL Window information");
  }

  void Loop ( ) {
    SDL_Event event;
    while ( true ) {
      while ( SDL_PollEvent(&event) ) {
        switch ( event.type ) {
          default: break;
          case SDL_QUIT: return;
        }
      }
    }
  }

  void Cleanup ( ) {
    writeln("Cleaning up...");
    vkDestroyDebugReportCallbackEXT(vk_ctx.instance,
                                    vk_ctx.debug_callback, null);
    // VkPhysicalDevice is destroyed implicitly with VkInstance
    vkDestroyDevice(vk_ctx.logical_device, null);
    vkDestroyInstance(vk_ctx.instance, null);
    writeln("Cleanup finished");
  }
public:
  void Run ( ) {
    Initialize_Vulkan();
    Initialize_SDL();
    // Loop();
    Cleanup();
  }
}

void main ( ) {
  auto application = new TriangleApplication();

  try {
    application.Run();
  } catch ( Exception e ) {
    writeln("Caught exception: ", e);
  }
}

