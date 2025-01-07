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
#include <chrono>
#include <unistd.h>

// SFML
#ifdef SFML
#include <SFML/Graphics.hpp>
#endif

using namespace std::chrono;
using timer = high_resolution_clock;
using get_time = duration<double>;

/**
 * @brief Fills the world with objects (spheres) of different materials.
 *
 * @param world A reference to a `hittable_list` object that will hold the generated
 *              spheres and materials.
 */
void fill_world(hittable_list& world);

#ifdef SFML
/**
 * @brief Displays a ppm image given a frame buffer of pixels.
 * @param frame_buffer The frame buffer containing all pixels.
 * @param width The width of the image.
 * @param height The height of the image.
 */
void display_ppm(const std::vector<color>& frame_buffer, int width, int height);
#endif

int main(int argc, char** argv) {
    // output setup
    bool redirect = isatty(fileno(stdout));
    std::ofstream output;
    std::streambuf* standard_out;
    std::string filename = "image_cpu.ppm";
    if (redirect) {
        output.open(filename);
        standard_out = std::cout.rdbuf();
        std::cout.rdbuf(output.rdbuf());
    }

    int width = 1200;
    if (argc == 2) {
        try {
            width = std::stoi(argv[1]);
        } catch (std::invalid_argument& e) {
            std::cerr << "Invalid argument! Expected 'int', but got '" << argv[1] << "'.\n";
            return 0;
        }
        if (width < 100) {
            std::cerr << "Invalid width! width=" << width << " is to small, minimum required is width=100.\n";
            return 0;
        }
    }

    hittable_list world;

    // Scene
    fill_world(world);

    // Camera
    camera cam;

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = width;
    cam.samples_per_pixel = 10;
    cam.max_depth         = 50;

    cam.vfov              = 20;
    cam.lookfrom          = point3(13, 2, 3);
    cam.lookat            = point3(0, 0, 0);
    cam.vup               = vec3(0, 1, 0);

    cam.defocus_angle     = 0.6;
    cam.focus_dist        = 10.0;

    auto start = timer::now();
    cam.render(world);
    auto stop = timer::now();
    std::clog << "Rendering time: " << get_time(stop - start).count() << " seconds.\n";

    // restore stdout
    if (redirect) {
        std::cout.rdbuf(standard_out);
        output.close();
    }

#ifdef SFML
    display_ppm(cam.frame_buffer(), cam.width(), cam.height());
#endif

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

#ifdef SFML
void display_ppm(const std::vector<color>& frame_buffer, int width, int height) {
    sf::Image sfImage;
    sfImage.create(width, height);

    const interval color_interval(0.0, 0.999);

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j * width + i;
            const color& pixel = frame_buffer[pixel_index];

            int r = static_cast<int>(256 * color_interval.clamp(pixel.x()));
            int g = static_cast<int>(256 * color_interval.clamp(pixel.y()));
            int b = static_cast<int>(256 * color_interval.clamp(pixel.z()));

            sfImage.setPixel(i, j, sf::Color(r, g, b));
        }
    }

    sf::Texture texture;
    texture.loadFromImage(sfImage);

    sf::Sprite sprite(texture);

    sf::RenderWindow window(sf::VideoMode(width, height), "CPU Ray Tracer");

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)
                window.close();
        }

        window.clear();
        window.draw(sprite);
        window.display();
    }
}
#endif