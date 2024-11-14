// Project
#include <util.h>
#include <color.h>
#include <hittable.h>
#include <hittable_list.h>
#include <sphere.h>
#include <camera.h>
#include <material.h>

// std
#include <fstream>

void fill_world(hittable_list& world);

int main() {
    // output setup
    std::ofstream output("image.ppm");
    std::streambuf* standard_out = std::cout.rdbuf();
    std::cout.rdbuf(output.rdbuf());

    hittable_list world;

    // Scene
    fill_world(world);

    // Camera
    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 10;
    cam.max_depth         = 50;

    cam.vfov     = 20;
    cam.lookfrom = point3(13, 2, 3);
    cam.lookat   = point3(0, 0, 0);
    cam.vup      = vec3(0, 1, 0);

    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;

    cam.render(world);

    // restore stdout
    std::cout.rdbuf(standard_out);
    output.close();
    return 0;
}

void fill_world(hittable_list& world) {
    auto ground_material = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(std::make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_real_number();
            point3 center(a + 0.9 * random_real_number(), 0.2, b + 0.9 * random_real_number());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                material_ptr sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = std::make_shared<lambertian>(albedo);
                    world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
                }
                else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_real_number(0, 0.5);
                    sphere_material = std::make_shared<metal>(albedo, fuzz);
                    world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
                }
                else {
                    // glass
                    sphere_material = std::make_shared<dielectric>(1.5);
                    world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = std::make_shared<dielectric>(1.5);
    world.add(std::make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = std::make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(std::make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = std::make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(std::make_shared<sphere>(point3(4, 1, 0), 1.0, material3));
}