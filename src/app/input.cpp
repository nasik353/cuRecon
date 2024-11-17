#include "app/input.h"
#include <fmt/core.h>

input::input(viewer *v) : m_viewer(v), last_mouse_pos(0, 0) {}

void input::handle_input()
{
    ImGuiIO &io = ImGui::GetIO();

    ImVec2 mouse_pos = io.MousePos;
    ImVec2 mouse_delta = ImVec2(mouse_pos.x - last_mouse_pos.x, mouse_pos.y - last_mouse_pos.y);

    if (mouse_pos.x < 0 || mouse_pos.x > 600 || mouse_pos.y < 0 || mouse_pos.y > 600)
    {
        last_mouse_pos = mouse_pos;
        return;
    }

    if (io.MouseDown[0])
    {
        m_viewer->RotateCamera(mouse_delta.x, mouse_delta.y);
    }
    else if (io.MouseDown[1])
    {
        m_viewer->PanObject(mouse_delta.x, -mouse_delta.y);
    }

    if (io.MouseWheel != 0.0f)
    {
        m_viewer->Zoom(io.MouseWheel);
    }

    last_mouse_pos = mouse_pos;
}