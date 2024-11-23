#include <GL/glew.h>
#include <fmt/core.h>
#define GLFW_INCLUDE_NONE
#include "app/hdf5_handler.h"
#include "app/imgui_impl_glfw.h"
#include "app/imgui_impl_opengl3.h"
#include "app/input.h"
#include "app/viewer.h"
#include "cuRecon/pointcloud_processing.h"
#include "imgui.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdio.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <filesystem>

static const int window_width = 1200;
static const int window_height = 1000;
static const int settings_width = 300;
static const int viewer_width = 900;
static const int viewer_height = 1000;

void saveToPLY(const pointcloud &pc, const std::string &filename)
{

    std::ofstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << pc.data.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";

    for (const auto &p : pc.data)
    {
        file << p.x << " " << p.y << " " << p.z << " " << static_cast<int>(p.r * 255) << " "
             << static_cast<int>(p.g * 255) << " " << static_cast<int>(p.b * 255) << "\n";
    }

    file.close();
    fmt::println("Point cloud saved to {} successfully.", filename);
}

void removePointCloudNoise(pointcloud &cloud, const std::pair<std::vector<int64_t>, std::vector<int64_t>> &neighbors,
                           int min_neighbors)
{
    std::unordered_map<int64_t, int64_t> neighbor_count;

    for (const int64_t idx : neighbors.second)
    {
        neighbor_count[idx]++;
    }

    std::unordered_set<int64_t> valid_indices;
    for (const auto &entry : neighbor_count)
    {
        if (entry.second >= min_neighbors)
        {
            valid_indices.insert(entry.first);
        }
    }

    std::vector<point> filtered_points;
    filtered_points.reserve(valid_indices.size());

    for (const int64_t idx : valid_indices)
    {
        filtered_points.push_back(cloud.data[idx]);
    }

    cloud.data = std::move(filtered_points);
}

static void glfw_error_callback(int error, const char *description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

GLFWwindow *init_glfw()
{

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
    {
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWwindow *window = glfwCreateWindow(window_width, window_height, "cuRecon", nullptr, nullptr);
    if (window == nullptr)
    {
        return nullptr;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return nullptr;
    }

    return window;
}

void init_imgui(GLFWwindow *window)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();

    ImGuiStyle &style = ImGui::GetStyle();
    style.Colors[ImGuiCol_TitleBg] = ImVec4(41.0f / 255.0f, 74.0f / 255.0f, 122.0f / 255.0f, 1.0f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(41.0f / 255.0f, 74.0f / 255.0f, 122.0f / 255.0f, 1.0f);
    style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(41.0f / 255.0f, 74.0f / 255.0f, 122.0f / 255.0f, 1.0f);
    style.FrameRounding = 0.0f;
    style.WindowRounding = 0.0f;
    style.FramePadding = ImVec2(4, 4);

    const char *glsl_version = "#version 130";
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

int main(int, char **)
{
    GLFWwindow *window = init_glfw();
    init_imgui(window);

    // init viewer
    viewer viewer(viewer_width, viewer_height);
    viewer.Initialize();
    glfwSetWindowUserPointer(window, &viewer);

    // init others
    input inputHandler(&viewer);
    std::vector<frame> selected_frames;
    std::mutex frames_mutex;

    std::unique_ptr<hdf5::HDF5ImageLoader> loader;

    if (std::filesystem::exists("data/egg_scan.h5"))
    {
        loader = std::make_unique<hdf5::HDF5ImageLoader>("data/egg_scan.h5");
    }

    std::vector<std::string> frame_names = loader->get_frame_names();
    std::vector<bool> selected(frame_names.size(), false);

    float duration = 0.0f;
    int original_points_count = 0.0f;
    int downsampled_points_count = 0.0f;

    float radius_search = 0.02f;
    int max_num_neighbors = 50;
    int min_num_neighbors = 3;

    bool frame_selection_changed = false;

    while (!glfwWindowShouldClose(window))
    {
        // clean screen
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // process input
        inputHandler.handle_input();

        if (ImGui::IsKeyPressed(ImGuiKey_D))
        {
            for (size_t i = 0; i < selected_frames.size(); ++i)
            {
                auto &frame = selected_frames[i];

                float downsample_factor = 0.5f;
                bool random_start = true;
                auto start = std::chrono::high_resolution_clock::now();
                original_points_count = frame._pointcloud.data.size();
                frame._pointcloud = farthestPointSampling(frame._pointcloud, downsample_factor, random_start);
                downsampled_points_count = frame._pointcloud.data.size();
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> _duration = end - start;
                duration = _duration.count();
            }
        }

        if (ImGui::IsKeyPressed(ImGuiKey_S))
        {
            for (size_t i = 0; i < selected_frames.size(); ++i)
            {
                auto &frame = selected_frames[i];

                saveToPLY(frame._pointcloud, fmt::format("frame_{}.ply", frame._id));

                // float downsample_factor = 0.5f;
                // bool random_start = true;
                // auto start = std::chrono::high_resolution_clock::now();
                // original_points_count = frame._pointcloud.data.size();

                // auto radius_results = radius(frame._pointcloud, frame._pointcloud, radius_search, max_num_neighbors);
                // removePointCloudNoise(frame._pointcloud, radius_results, min_num_neighbors);

                // downsampled_points_count = frame._pointcloud.data.size();
                // auto end = std::chrono::high_resolution_clock::now();
                // std::chrono::duration<double, std::milli> _duration = end - start;
                // duration = _duration.count();
            }
        }

        if (ImGui::IsKeyPressed(ImGuiKey_E))
        {
            frame_selection_changed = true;
        }

        // render
        viewer.Render(selected_frames);

        ImGui::SetNextWindowPos(ImVec2(window_width - settings_width, 0));
        ImGui::SetNextWindowSize(ImVec2(settings_width, window_height));

        ImGui::Begin("Settings", nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        ImGui::InputFloat("search radius", &radius_search);
        ImGui::InputInt("max_num_neighbors", &max_num_neighbors);
        ImGui::InputInt("min_num_neighbors", &min_num_neighbors);

        ImGui::Text("fps %.3f ms", duration);
        ImGui::Text("original %.3i", original_points_count);
        ImGui::Text("downsampled %.3i", downsampled_points_count);

        if (ImGui::TreeNode("Frames"))
        {
            for (size_t i = 0; i < frame_names.size(); ++i)
            {
                bool isSelected = selected[i];
                if (ImGui::Checkbox(frame_names[i].c_str(), &isSelected))
                {
                    selected[i] = isSelected;
                    frame_selection_changed = true;
                }
            }
            ImGui::TreePop();
        }

        if (frame_selection_changed)
        {
            selected_frames.clear();
            for (size_t i = 0; i < frame_names.size(); ++i)
            {
                if (selected[i])
                {
                    selected_frames.push_back(loader->load_frame(frame_names[i]));
                }
            }
        }
        frame_selection_changed = false;

        ImGui::End();

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(viewer_width, viewer_height));
        ImGui::Begin("Viewer", nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
        ImVec2 available_size = ImGui::GetContentRegionAvail();
        ImGui::Image((void *)(intptr_t)viewer.GetTextureID(), available_size);
        ImGui::End();

        ImGui::Render();

        // clean up
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
