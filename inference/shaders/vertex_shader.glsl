#version 330 core

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

out vec3 fragColor;
uniform mat4 mvpMatrix;

void main()
{
    gl_Position = mvpMatrix * vec4(inPosition, 1.0);
    fragColor = inColor;
}
