import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    executable = "matrix_mult"
    source_file = "matrix_mult.cu"
    if not os.path.isfile("./" + executable):
        print(f"{executable} not found. Compiling {source_file}...")
        compile_result = subprocess.run([
            "nvcc", "-arch=sm_50", "-Wno-deprecated-gpu-targets", "-O3",
            "-o", executable, source_file
        ], capture_output=True, text=True)

        if compile_result.returncode != 0:
            print(f"Compilation failed:\n{compile_result.stderr}")
            exit(1)
        else:
            print("Compilation successful.")

    matrix_sizes = [(1024, 1024), (2048, 2048), (3072, 3072), (4096, 4096), (5120, 5120), (6144, 6144), (7168, 7168), (8192, 8192), (9216, 9216), (10240, 10240)]
    host_to_device_times = []
    kernel_times = []
    device_to_host_times = []

    for rows, cols in matrix_sizes:
        result = subprocess.run(["./" + executable, str(rows), str(cols), str(rows)], capture_output=True, text=True)
        output = result.stdout

        try:
            for line in output.splitlines():
                if "Host to Device transfer time" in line:
                    host_to_device_times.append(float(line.split(": ")[1].split(" ")[0]))
                elif "Kernel execution time" in line:
                    kernel_times.append(float(line.split(": ")[1].split(" ")[0]))
                elif "Device to Host transfer time" in line:
                    device_to_host_times.append(float(line.split(": ")[1].split(" ")[0]))
        except Exception as e:
            print(f"Error parsing output for matrix size {rows}x{cols}: {e}")
            continue

    host_to_device_times = np.array(host_to_device_times)
    kernel_times = np.array(kernel_times)
    device_to_host_times = np.array(device_to_host_times)

    matrix_size_labels = [f"{rows}x{cols}" for rows, cols in matrix_sizes]

    plt.bar(matrix_size_labels, host_to_device_times, label="Host to Device Copy", color="blue")
    plt.bar(matrix_size_labels, device_to_host_times, bottom=host_to_device_times, label="Device to Host Copy", color="orange")
    plt.bar(matrix_size_labels, kernel_times,
            bottom=host_to_device_times + device_to_host_times,
            label="Kernel Execution", color="green")

    plt.xlabel("Matrix Size")
    plt.ylabel("Time (s)")
    plt.xticks(matrix_size_labels, rotation=45)
    plt.legend()
    plt.title("Breakdown of Execution Time for Different Matrix Sizes")
    plt.tight_layout()
    plt.savefig(executable + ".png")