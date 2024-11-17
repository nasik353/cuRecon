#ifndef VIEWER_H
#define VIEWER_H

#include <vector>
#include <fmt/core.h>
#include <string>
#include <algorithm>
#include <GL/glew.h>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "cuRecon/types.h"

class viewer
{
public:
    viewer(int width, int height);
    ~viewer();

    void Initialize();
    void Render(const std::vector<frame> &pointCloud);
    void MouseButton(int button, int state, int x, int y);
    void MouseMotion(int x, int y);
    void Zoom(float delta);
    void FitViewToPoints(const pointcloud &pointCloud);
    GLuint GetTextureID() const;
    void RotateCamera(float deltaX, float deltaY);
    void PanObject(float deltaX, float deltaY);

private:
    void SetupFrameBuffer();
    void SetupVertexArray();
    void UpdateView();
    void RenderOriginArrows();
    void RenderPointCloud(const pointcloud &cloud);

    GLuint LoadShaders(const std::string &vertexShaderPath, const std::string &fragmentShaderPath);
    std::string LoadShaderSource(const std::string &filepath);
    GLuint CompileShader(GLenum shaderType, const std::string &source);

    int width, height;
    GLuint fbo, fboTexture;
    GLuint VAO, VBO;

    float zoom;
    float panX, panY;
    bool leftMouseDown, rightMouseDown;
    int lastMouseX, lastMouseY;

    glm::quat rotation;
    glm::vec3 cameraPosition;
    glm::vec3 cameraTarget;
    glm::vec3 upVector;

    GLuint shaderProgram;
};

#endif // VIEWER_H
