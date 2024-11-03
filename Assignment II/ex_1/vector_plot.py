import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    executable = "vector_add"
    source_file = "vector_add.cu"
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

    input_lengths = [1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000]
    host_to_device_times = []
    kernel_times = []
    device_to_host_times = []
    for length in input_lengths:
        result = subprocess.run(["./" + executable, str(length)], capture_output=True, text=True)
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
            print(f"Error parsing output for input length {length}: {e}")
            continue

    host_to_device_times = np.array(host_to_device_times)
    kernel_times = np.array(kernel_times)
    device_to_host_times = np.array(device_to_host_times)

    input_lengths = [str(length) for length in input_lengths]

    plt.bar(input_lengths, host_to_device_times, label="Host to Device Copy", color="blue")
    plt.bar(input_lengths, device_to_host_times, bottom=host_to_device_times, label="Device to Host Copy", color="orange")
    plt.bar(input_lengths, kernel_times,
            bottom=host_to_device_times + device_to_host_times,
            label="Kernel Execution", color="green")

    plt.xlabel("Vector Length")
    plt.ylabel("Time (s)")
    plt.xticks(input_lengths, rotation=45)
    plt.legend()
    plt.title("Breakdown of Execution Time for Different Vector Lengths")
    plt.tight_layout()
    plt.savefig(executable + ".png")