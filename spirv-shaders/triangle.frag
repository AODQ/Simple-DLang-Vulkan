#version 450
#extension GL_ARB_separate_shader_objects : enable

// -- uniforms --
layout(binding = 1) uniform sampler2D tex_sampler;

// -- in --
layout(location = 0) in vec3 frag_colour;
layout(location = 1) in vec2 frag_tex_coord;

// -- out --
layout(location = 0) out vec4 out_colour;

// -- entry point --
void main() {
  out_colour = mix(vec4(frag_colour, 1.0f), texture(tex_sampler, frag_tex_coord), vec4(0.5f));
}