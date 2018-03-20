
import std.stdio;
import std.range;
import std.array;
import std.algorithm;
import std.exception;
import std.conv;
import std.string;

import erupted;
import derelict.glfw3.glfw3;
import vector;
import gfm.math;



private void enforceVK(VkResult res) {
  enforce(res == VkResult.VK_SUCCESS, res.to!string);
}
void main ( ) {
  DerelictErupted.load();
  DerelictGLFW3.load();
  glfwInit();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_FLOATING, VK_TRUE);
  glfwWindowHint(GLFW_RESIZABLE, VK_TRUE);
  glfwWindowHint(GLFW_REFRESH_RATE, VK_FALSE);
  glfwSwapInterval(1);
  auto window = glfwCreateWindow(800, 600, "Vulkan window", null, null);
  uint extension_count;
  vkEnumerateInstanceExtensionProperties(null, &extension_count, null);
  writeln("extensions: ", extension_count);
  float4x4 mat;
  float4 vec;
  auto test = mat*vec;
  while ( !glfwWindowShouldClose(window) ) {
    glfwPollEvents();
  }

  glfwDestroyWindow(window);
  glfwTerminate();
}

