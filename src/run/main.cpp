#include "cppgl/context.h"
#include "cppgl/camera.h"

int main(int argc, char** argv) {
    ContextParameters params;
    params.title = "VolGL";
    Context::init(params);

    while (Context::running()) {
        // handle input
        Camera::default_input_handler(Context::frame_time());

        // update
        Camera::current()->update();

        // draw
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // finish frame
        Context::swap_buffers();
    }
}
