#version 450
#extension GL_ARB_separate_shader_objects : enable

//--- uniforms ---
layout(binding = 0) uniform UniformBufferObject {
  mat4 model;
  mat4 view;
  mat4 proj;
} ubo;

//--- in ---
layout(location = 0) in vec3 in_vertex;
layout(location = 1) in vec3 in_colour;
layout(location = 2) in vec2 in_tex_coord;

//--- out ---
layout(location = 0) out vec3 frag_colour;
layout(location = 1) out vec2 frag_tex_coord;

out gl_PerVertex {
    vec4 gl_Position;
};

// -- entry point --
void main() {
  gl_Position = ubo.proj*(ubo.view)*ubo.model*vec4(in_vertex, 1.0f);
  gl_Position.z = (gl_Position.z + gl_Position.w)/2.0f;
  // gl_Position.y = -gl_Position.y;
  frag_colour = in_colour;
  frag_tex_coord = in_tex_coord;
}