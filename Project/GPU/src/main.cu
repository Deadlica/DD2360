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
#include <curand_kernel.h>

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

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera,
                             datatype aspect_ratio, int image_width, int samples_per_pixel) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        material* material_ground = new lambertian(color(0.8, 0.8, 0.0));
        material* material_center = new lambertian(color(0.1, 0.2, 0.5));
        material* material_left   = new dielectric(1.50);
        material* material_bubble = new dielectric(1.00 / 1.50);
        material* material_right  = new metal(color(0.8, 0.6, 0.2), 0.0);

        d_list[0] = new sphere(point3( 0.0, -100.5, -1.0), 100.0, material_ground);
        d_list[1] = new sphere(point3( 0.0,    0.0, -1.2),   0.5, material_center);
        d_list[2] = new sphere(point3(-1.0,    0.0, -1.0),   0.5, material_left);
        d_list[3] = new sphere(point3(-1.0,    0.0, -1.0),   0.4, material_bubble);
        d_list[4] = new sphere(point3( 1.0,    0.0, -1.0),   0.5, material_right);

        *d_world      = new hittable_list(d_list, 5);
        *d_camera     = new camera();
        (*d_camera)->aspect_ratio      = aspect_ratio;
        (*d_camera)->image_width       = image_width;
        (*d_camera)->samples_per_pixel = samples_per_pixel;
        (*d_camera)->initialize();
    }
}

__global__ void delete_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    for (int i = 0; i < 5; i++) {
        delete ((sphere*) d_list[i])->mat;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main() {
    // output setup
    std::ofstream output("image.ppm");
    std::streambuf* standard_out = std::cout.rdbuf();
    std::cout.rdbuf(output.rdbuf());

    // Camera variables
    datatype aspect_ratio      = datatype(16.0) / datatype(9.0);
    int      image_width       = 800;
    int      image_height      = int(image_width / aspect_ratio);
    int      samples_per_pixel = 100;
    image_height = image_height < 1 ? 1 : image_height;
    int      num_pixels        = image_width * image_height;

    // Frame buffer
    size_t frame_buffer_size = num_pixels * sizeof(vec3);
    vec3* frame_buffer;
    checkCudaErrors(cudaMallocManaged((void**) &frame_buffer, frame_buffer_size));

    // Curand
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void**) &d_rand_state, num_pixels * sizeof(curandState)));

    // World
    hittable** d_list;
    checkCudaErrors(cudaMalloc((void**) &d_list, 5 * sizeof(hittable*)));
    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**) &d_world, sizeof(hittable*)));
    camera**   d_camera;
    checkCudaErrors(cudaMalloc((void**) &d_camera, sizeof(camera*)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera,
                           aspect_ratio, image_width, samples_per_pixel);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    timeval start;
    timer_start(&start);

    // Rendering
    dim3 db(TPB.x, TPB.y);
    dim3 dg((image_width + db.x - 1) / db.x, (image_height + db.y - 1) / db.y);

    render<<<dg, db>>>(frame_buffer, image_width, image_height,
                       d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    double time_elapsed;
    timer_stop(&start, &time_elapsed);
    std::clog << "Rendering time: " << time_elapsed << " seconds.\n";

    // Output frame_buffer as Image
    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width + i;
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
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(frame_buffer));
    cudaDeviceReset();

    // restore stdout
    std::cout.rdbuf(standard_out);
    output.close();
    return 0;
}