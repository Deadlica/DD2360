// Project
#include <util.cuh>

// std
#include <fstream>

__global__ void render(datatype *fb, int max_x, int max_y) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= max_x || y >= max_y) {
        return;
    }
    int pixel_index = y * max_x * 3 + x * 3;
    fb[pixel_index + 0] = datatype(x) / max_x;
    fb[pixel_index + 1] = datatype(y) / max_y;
    fb[pixel_index + 2] = 0.2;
}

int main() {
    // output setup
    std::ofstream output("image.ppm");
    std::streambuf* standard_out = std::cout.rdbuf();
    std::cout.rdbuf(output.rdbuf());

    int image_width          = 1920;
    int image_height         = 1080;
    int num_pixels           = image_width * image_height;
    size_t frame_buffer_size = 3 * num_pixels * sizeof(datatype);

    datatype* frame_buffer;
    checkCudaErrors(cudaMallocManaged((void**) &frame_buffer, frame_buffer_size));

    dim3 db(TPB.x, TPB.y);
    dim3 dg((image_width + db.x - 1) / db.x, (image_height + db.y - 1) / db.y);
    render<<<dg, db>>>(frame_buffer, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Output FB as Image
    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * 3 * image_width + i * 3;
            datatype r = frame_buffer[pixel_index + 0];
            datatype g = frame_buffer[pixel_index + 1];
            datatype b = frame_buffer[pixel_index + 2];
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    checkCudaErrors(cudaFree(frame_buffer));

    // restore stdout
    std::cout.rdbuf(standard_out);
    output.close();
    return 0;
}