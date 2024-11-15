// Project
#include <util.cuh>
#include <vec3.cuh>
#include <ray.cuh>
#include <color.cuh>

// std
#include <fstream>

__device__ bool hit_sphere(const vec3& center, datatype radius, const ray& r) {
    vec3 oc = r.origin() - center;
    datatype a = dot(r.direction(), r.direction());
    datatype b = datatype(2.0) * dot(oc, r.direction());
    datatype c = dot(oc, oc) - radius * radius;
    datatype discriminant = b * b - datatype(4.0) * a * c;
    return discriminant > datatype(0.0);
}

__device__ vec3 ray_color(const ray& r) {
    if (hit_sphere(vec3(0, 0, -1), 0.5, r)) {
        return vec3(1, 0, 0);
    }
    vec3 unit_direction = unit_vector(r.direction());
    datatype a = datatype(0.5) * (unit_direction.y() + datatype(1.0));
    return (datatype(1.0) - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}

__global__ void render(vec3* frame_buffer, int max_x, int max_y,
                       vec3 pixel00_loc, vec3 horizontal, vec3 vertical, vec3 origin) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= max_x || y >= max_y) {
        return;
    }
    int pixel_index = y * max_x + x;
    ray r(origin, pixel00_loc + datatype(x) * horizontal + datatype(y) * vertical);
    frame_buffer[pixel_index] = ray_color(r);
}

int main() {
    // output setup
    std::ofstream output("image.ppm");
    std::streambuf* standard_out = std::cout.rdbuf();
    std::cout.rdbuf(output.rdbuf());

    datatype aspect_ratio    = datatype(16.0) / datatype(9.0);
    int      image_width     = 800;
    int      image_height    = int(image_width / aspect_ratio);

    image_height = image_height < 1 ? 1 : image_height;
    size_t frame_buffer_size = image_width * image_height * sizeof(vec3);

    // Camera
    datatype focal_length = 1.0;
    datatype viewport_height = 2.0;
    datatype viewport_width = viewport_height * (datatype(image_width)/image_height);
    auto camera_center = point3(0, 0, 0);

    auto viewport_x = vec3(viewport_width, 0, 0);
    auto viewport_y = vec3(0, -viewport_height, 0);

    auto pixel_delta_x = viewport_x / image_width;
    auto pixel_delta_y = viewport_y / image_height;

    auto viewport_upper_left = camera_center
                               - vec3(0, 0, focal_length) - viewport_x / 2 - viewport_y / 2;
    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_x + pixel_delta_y);

    vec3* frame_buffer;
    checkCudaErrors(cudaMallocManaged((void**) &frame_buffer, frame_buffer_size));

    dim3 db(TPB.x, TPB.y);
    dim3 dg((image_width + db.x - 1) / db.x, (image_height + db.y - 1) / db.y);
    render<<<dg, db>>>(frame_buffer, image_width, image_height,
                       pixel00_loc,
                       pixel_delta_x,
                       pixel_delta_y,
                       camera_center);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Output frame_buffer as Image
    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width + i;
            write_color(std::cout, frame_buffer[pixel_index]);
        }
    }
    checkCudaErrors(cudaFree(frame_buffer));

    // restore stdout
    std::cout.rdbuf(standard_out);
    output.close();
    return 0;
}