#include "util.h"
#include "color.h"
#include "ray.h"
#include "vec3.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"

#include <fstream>

int main() {
    // output setup
    std::ofstream output("image.ppm");
    std::streambuf* standard_out = std::cout.rdbuf();
    std::cout.rdbuf(output.rdbuf());

    hittable_list world;

    // Scene
    world.add(std::make_shared<sphere>(point3(0, 0 , -1), 0.5));
    //world.add(std::make_shared<sphere>(point3(1.25, 0 , -1), 0.4));
    world.add(std::make_shared<sphere>(point3(0, -100.5, -1), 100));

    // Camera
    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_depth         = 50;

    cam.render(world);

    // restore stdout
    std::cout.rdbuf(standard_out);
    output.close();
    return 0;
}