#version 450

layout(location = 0) in vec2 in_center;
layout(location = 1) in vec2 in_radius;
layout(location = 2) in vec4 in_color;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_local_position;

void main() {
    const vec2 corners[6] = vec2[6](
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
    gl_Position = vec4(in_center + corners[gl_VertexIndex] * in_radius, 0.0, 1.0);
    out_color = in_color;
    out_local_position = corners[gl_VertexIndex];
}
