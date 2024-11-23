#include "app/viewer.h"
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

void CheckGLError(const char *stmt, const char *fname, int line)
{
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
    {
        std::cerr << "OpenGL error " << err << " at " << fname << ":" << line << " - for " << stmt << std::endl;
    }
}

#define GL_CHECK(stmt)                           \
    do                                           \
    {                                            \
        stmt;                                    \
        CheckGLError(#stmt, __FILE__, __LINE__); \
    } while (0)

viewer::viewer(int width, int height)
    : width(width), height(height), fbo(0), fboTexture(0), VAO(0), VBO(0), zoom(5.0f), panX(0.0f), panY(0.0f),
      leftMouseDown(false), rightMouseDown(false), lastMouseX(0), lastMouseY(0),
      rotation(glm::quat(-0.341659, 0.492046, -0.654409, -0.46142)) {}

viewer::~viewer()
{
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &fboTexture);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
}

void viewer::Initialize()
{
    SetupFrameBuffer();
    SetupVertexArray();

    shaderProgram = LoadShaders("vertex_shader.glsl", "fragment_shader.glsl");

    glPointSize(2.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    UpdateView();
}

void viewer::SetupFrameBuffer()
{
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(1, &fboTexture);
    glBindTexture(GL_TEXTURE_2D, fboTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTexture, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cerr << "Failed to create framebuffer!" << std::endl;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void viewer::SetupVertexArray()
{
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBindVertexArray(0);
}

void viewer::RenderOriginArrows()
{
    float arrowLength = 0.5f;
    float arrowVertices[] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, arrowLength, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, arrowLength, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, arrowLength, 0.0f, 0.0f, 1.0f};

    GLuint arrowVBO;
    glGenBuffers(1, &arrowVBO);
    glBindBuffer(GL_ARRAY_BUFFER, arrowVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(arrowVertices), arrowVertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));

    glLineWidth(3.0f);
    glDrawArrays(GL_LINES, 0, 6);
    glLineWidth(1.0f);

    glDeleteBuffers(1, &arrowVBO);
}

void viewer::Render(const std::vector<frame> &frames)
{
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glViewport(0, 0, width, height);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaderProgram);

    for (const auto &frame : frames)
    {
        RenderPointCloud(frame._pointcloud);
    }

    RenderOriginArrows();

    glUseProgram(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDisable(GL_DEPTH_TEST);
}

void viewer::RenderPointCloud(const pointcloud &cloud)
{
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, cloud.data.size() * sizeof(point), &cloud.data[0], GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(point), (void *)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(point), (void *)(3 * sizeof(float)));

    glDrawArrays(GL_POINTS, 0, cloud.data.size());

    glBindVertexArray(0);
}

GLuint viewer::GetTextureID() const { return fboTexture; }

void viewer::MouseButton(int button, int action, int x, int y)
{
    if (button == 0)
    {
        leftMouseDown = (action == 1);
    }
    else if (button == 1)
    {
        rightMouseDown = (action == 1);
    }
}

void viewer::MouseMotion(int x, int y)
{
    if (leftMouseDown)
    {
        float deltaX = static_cast<float>(x - lastMouseX);
        float deltaY = static_cast<float>(y - lastMouseY);
        RotateCamera(deltaX, deltaY);
    }
    else if (rightMouseDown)
    {
        float deltaX = static_cast<float>(x - lastMouseX);
        float deltaY = static_cast<float>(y - lastMouseY);
        PanObject(deltaX, deltaY);
    }

    lastMouseX = x;
    lastMouseY = y;
}

void viewer::Zoom(float delta)
{
    float zoomSensitivity = 0.1f;
    zoom *= (1.0f - delta * zoomSensitivity);
    zoom = std::clamp(zoom, 0.1f, 100.0f);

    UpdateView();
}

void viewer::FitViewToPoints(const pointcloud &pointCloud)
{
    if (pointCloud.data.empty())
        return;

    float minX = pointCloud.data[0].x, maxX = pointCloud.data[0].x;
    float minY = pointCloud.data[0].y, maxY = pointCloud.data[0].y;
    float minZ = pointCloud.data[0].z, maxZ = pointCloud.data[0].z;

    for (const auto &p : pointCloud.data)
    {
        minX = std::min(minX, p.x);
        maxX = std::max(maxX, p.x);
        minY = std::min(minY, p.y);
        maxY = std::max(maxY, p.y);
        minZ = std::min(minZ, p.z);
        maxZ = std::max(maxZ, p.z);
    }

    float centerX = (minX + maxX) / 2.0f;
    float centerY = (minY + maxY) / 2.0f;
    float centerZ = (minZ + maxZ) / 2.0f;

    float sizeX = maxX - minX;
    float sizeY = maxY - minY;
    float sizeZ = maxZ - minZ;
    float maxSize = std::max({sizeX, sizeY, sizeZ});

    panX = -centerX;
    panY = -centerY;

    rotation = glm::quat(-0.341659, 0.492046, -0.654409, -0.46142);

    zoom = std::max(zoom, 0.1f);

    UpdateView();
}
void viewer::UpdateView()
{
    glm::mat4 projectionMatrix = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);

    glm::mat4 rotationMatrix = glm::mat4_cast(rotation);
    glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(panX, panY, -zoom));
    glm::mat4 viewMatrix = translationMatrix * rotationMatrix;
    glm::mat4 mvpMatrix = projectionMatrix * viewMatrix;

    GLint mvpMatrixLoc = glGetUniformLocation(shaderProgram, "mvpMatrix");
    glUseProgram(shaderProgram);
    glUniformMatrix4fv(mvpMatrixLoc, 1, GL_FALSE, glm::value_ptr(mvpMatrix));
    glUseProgram(0);
}

std::string viewer::LoadShaderSource(const std::string &filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Error opening shader file: " << filepath << std::endl;
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

GLuint viewer::CompileShader(GLenum shaderType, const std::string &source)
{
    GLuint shader = glCreateShader(shaderType);
    const char *sourceCStr = source.c_str();
    glShaderSource(shader, 1, &sourceCStr, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        GLchar infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader Compilation Failed:\n"
                  << infoLog << std::endl;
    }

    return shader;
}

GLuint viewer::LoadShaders(const std::string &vertexShaderPath, const std::string &fragmentShaderPath)
{
    std::string vertexShaderSource = LoadShaderSource(vertexShaderPath);
    std::string fragmentShaderSource = LoadShaderSource(fragmentShaderPath);

    GLuint vertexShader = CompileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = CompileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        GLchar infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Shader Program Linking Failed:\n"
                  << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

void viewer::RotateCamera(float deltaX, float deltaY)
{
    const float rotationSpeed = 0.0075f;

    glm::quat rotX = glm::angleAxis(rotationSpeed * -deltaY, glm::vec3(1.0f, 0.0f, 0.0f));
    glm::quat rotY = glm::angleAxis(rotationSpeed * deltaX, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::quat newRotation = rotX * rotY * rotation;

    rotation = glm::normalize(newRotation);

    UpdateView();
}

void viewer::PanObject(float deltaX, float deltaY)
{
    constexpr float panSensitivity = 0.005f;

    panX += deltaX * panSensitivity;
    panY -= deltaY * panSensitivity;

    UpdateView();
}
