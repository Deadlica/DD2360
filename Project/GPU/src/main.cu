// Project
#include <util.cuh>
#include <vec3.cuh>
#include <ray.cuh>
#include <color.cuh>
#include <hittable.cuh>
#include <hittable_list.cuh>
#include <sphere.cuh>
#include <camera.cuh>
#include <material.cuh>

// std
#include <fstream>
#include <unistd.h>
#include <curand_kernel.h>

constexpr int num_hittables = 22 * 22 + 1 + 3; ///< Total number of hittable objects in the scene
constexpr int seed          = 1984;            ///< Random seed used for generating random numbers

/**
 * @brief Initializes the random state for curand.
 *
 * This kernel initializes the curand state used for generating random numbers on the GPU.
 * It sets up the random number generator using the provided seed.
 *
 * @param rand_state Pointer to the random state structure on the device.
 */
__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(seed, 0, 0, rand_state);
    }
}

/**
 * @brief Renders a scene by tracing rays from the camera.
 *
 * This kernel performs the rendering of the scene by shooting rays from the camera's position
 * for each pixel and computes the color based on interactions with the world.
 *
 * @param frame_buffer Pointer to the frame buffer where the resulting pixel color will be stored.
 * @param max_x Maximum x-coordinate (width) of the image.
 * @param max_y Maximum y-coordinate (height) of the image.
 * @param cam Pointer to the camera object used for ray generation.
 * @param world Pointer to the world (list of hittable objects).
 * @param rand_state Pointer to the random state structure for generating random numbers.
 */
__global__ void render(vec3* frame_buffer, int max_x, int max_y,
                       camera** cam, hittable** world, curandState* rand_state) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= max_x || y >= max_y) {
        return;
    }
    int pixel_index = y * max_x + x;
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
    curandState local_rand_state = rand_state[pixel_index];
    int samples_per_pixel = (*cam)->samples_per_pixel;
    color pixel_color(0, 0, 0);
    for (int sample = 0; sample < samples_per_pixel; sample++) {
        datatype dx = datatype(x + curand_uniform(&local_rand_state));
        datatype dy = datatype(y + curand_uniform(&local_rand_state));
        ray r = (*cam)->get_ray(dx, dy, &local_rand_state);
        pixel_color += (*cam)->ray_color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    frame_buffer[pixel_index] = pixel_color / datatype(samples_per_pixel);
}

#ifndef RND
#define RND (curand_uniform(&local_rand_state)) ///< Macro to generate random float numbers between 0 and 1
#endif

/**
 * @brief Creates the scene by populating the world with spheres and setting up the camera.
 *
 * @param d_list Pointer to the list of hittable objects.
 * @param d_world Pointer to the world object containing all hittable objects.
 * @param d_camera Pointer to the camera object.
 * @param rec Camera configuration containing scene parameters.
 * @param rand_state Pointer to the random state structure.
 */
__global__ void create_world(hittable** d_list, hittable** d_world,
                             camera** d_camera, cam_record rec, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));

        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                datatype choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < datatype(0.8)) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < datatype(0.95)) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(datatype(0.5) * (datatype(1.0) + RND),
                                                            datatype(0.5) * (datatype(1.0) + RND),
                                                            datatype(0.5) * (datatype(1.0) + RND)),
                                                       datatype(0.5) * RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3( 0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3( 4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;

        *d_world      = new hittable_list(d_list, num_hittables);
        *d_camera     = new camera();
        (*d_camera)->aspect_ratio      = rec.aspect_ratio;
        (*d_camera)->image_width       = rec.image_width;
        (*d_camera)->samples_per_pixel = rec.samples_per_pixel;
        (*d_camera)->vfov              = rec.vfov;
        (*d_camera)->lookfrom          = rec.lookfrom;
        (*d_camera)->lookat            = rec.lookat;
        (*d_camera)->vup               = rec.vup;
        (*d_camera)->defocus_angle     = rec.defocus_angle;
        (*d_camera)->focus_dist        = rec.focus_dist;
        (*d_camera)->initialize();
    }
}

/**
 * @brief Cleans up and deallocates resources used by the world and camera.
 *
 * This kernel deletes the objects in the world and the camera to free up memory.
 *
 * @param d_list Pointer to the list of hittable objects.
 * @param d_world Pointer to the world object containing all hittable objects.
 * @param d_camera Pointer to the camera object.
 */
__global__ void delete_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    for (int i = 0; i < num_hittables; i++) {
        delete ((sphere*) d_list[i])->mat;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main() {
    // output setup
    bool redirect = isatty(fileno(stdout));
    std::ofstream output;
    std::streambuf* standard_out;
    if (redirect) {
        output.open("image_gpu.ppm");
        standard_out = std::cout.rdbuf();
        std::cout.rdbuf(output.rdbuf());
    }

    // Camera variables
    cam_record rec;
    rec.aspect_ratio      = datatype(16.0) / datatype(9.0);
    rec.image_width       = 1200;
    rec.samples_per_pixel = 10;

    rec.vfov              = 20;
    rec.lookfrom          = point3(13, 2, 3);
    rec.lookat            = point3( 0, 0, 0);
    rec.vup               = vec3  ( 0, 1, 0);

    rec.defocus_angle     = 0.6;
    rec.focus_dist        = 10.0;


    int image_height = int(rec.image_width / rec.aspect_ratio);
    image_height     = image_height < 1 ? 1 : image_height;
    int num_pixels   = rec.image_width * image_height;

    // Frame buffer
    size_t frame_buffer_size = num_pixels * sizeof(vec3);
    vec3* frame_buffer;
    checkCudaErrors(cudaMallocManaged((void**) &frame_buffer, frame_buffer_size));

    // Curand
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void**) &d_rand_state, num_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void**) &d_rand_state2, sizeof(curandState)));

    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // World
    hittable** d_list;
    checkCudaErrors(cudaMalloc((void**) &d_list, num_hittables * sizeof(hittable*)));
    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**) &d_world, sizeof(hittable*)));
    camera**   d_camera;
    checkCudaErrors(cudaMalloc((void**) &d_camera, sizeof(camera*)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera, rec, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    timeval start;
    timer_start(&start);

    // Rendering
    dim3 db(TPB.x, TPB.y);
    dim3 dg((rec.image_width + db.x - 1) / db.x, (image_height + db.y - 1) / db.y);

    render<<<dg, db>>>(frame_buffer, rec.image_width, image_height,
                       d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    double time_elapsed;
    timer_stop(&start, &time_elapsed);
    std::clog << "Rendering time: " << time_elapsed << " seconds.\n";

    // Output frame_buffer as Image
    std::cout << "P3\n" << rec.image_width << " " << image_height << "\n255\n";
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < rec.image_width; i++) {
            size_t pixel_index = j * rec.image_width + i;
            write_color(std::cout, frame_buffer[pixel_index]);
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    delete_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(frame_buffer));
    cudaDeviceReset();

    // restore stdout
    if (redirect) {
        std::cout.rdbuf(standard_out);
        output.close();
    }

    return 0;
}