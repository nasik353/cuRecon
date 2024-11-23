// input.h

#ifndef INPUT_H
#define INPUT_H

#include "viewer.h"
#include "imgui.h"

class input
{
public:
    input(viewer *v);
    void handle_input();

private:
    viewer *m_viewer;
    ImVec2 last_mouse_pos;
};

#endif // INPUT_H
