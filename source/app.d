
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

void main ( ) {
  DerelictErupted.load();
  DerelictSDL2.load();
  auto sdl_window = SDL_CreateWindow("Vulkan", 0, 0, 800, 600, 0);

  uint extension_count;
  vkEnumerateInstanceExtensionProperties(null, &extension_count, null);
  writeln("extensions: ", extension_count);
  float4x4 mat;
  float4 vec;
  auto test = mat*vec;

  SDL_Event event;
  while ( true ) {
    while ( SDL_PollEvent(&event) ) {
      if ( event.type is SDL_QUIT ) goto END;
    }
  }
END:
}

