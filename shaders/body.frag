#version 450

layout(location = 0) in vec4 in_color;
layout(location = 1) in vec2 in_local_position;
layout(location = 0) out vec4 out_color;

void main() {
    if (dot(in_local_position, in_local_position) > 1.0) discard;
    out_color = in_color;
}
