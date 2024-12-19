import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

def compile_cuda_code(source_file, output_file):
    print(f"Compiling {source_file}...")
    compile_command = f"nvcc {source_file} -O3 -o {output_file}"
    result = subprocess.run(compile_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr.decode()}")
        exit(1)
    print("Compilation successful.")

def run_cuda_program(executable, input_length):
    print(f"Running {executable} {input_length}")
    run_command = f"{executable} {input_length}"
    result = subprocess.run(run_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Execution failed:\n{result.stderr.decode()}")
        exit(1)
    print(f"{executable} finished with return code {result.returncode}")
    return result.stdout.decode()

def parse_data(output):
    lines = output.splitlines()
    segment_length = int(lines[0].split()[-1])
    exec_time = float(lines[1].split()[-2])
    return segment_length, exec_time


def plot_bar(inputs, exec_times, filename):
    print(f"Plotting the output to {filename}")
    exec_times = np.array(exec_times)
    inputs = [str(length) for length in inputs]
    plt.bar(inputs, exec_times, label="Execution time", color="blue")

    plt.xlabel("Vector Length")
    plt.ylabel("Time (s)")
    plt.xticks(inputs, rotation=45)
    plt.ylim(0, 0.1)
    plt.legend()
    plt.title("Breakdown of Execution Time for Different Vector Lengths")
    plt.tight_layout()
    plt.savefig(filename)

if __name__ == "__main__":
    name        = "stream_add"

    if len(sys.argv) > 2:
        print("Too many arguments provided.")
        exit(0)
    elif len(sys.argv) == 2:
        if sys.argv[1] == "stream_add":
            name = sys.argv[1]
        elif sys.argv[1] == "vector_add":
            name = sys.argv[1]
        else:
            print("Invalid argument provided.")
            exit(0)

    source_file = name + ".cu"
    output_file = name
    plot_file   = name + ".png"
    inputs = [1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000]
    segments     = []
    exec_times = []

    compile_cuda_code(source_file, output_file)

    for input in inputs:
        output = run_cuda_program(f"./{output_file}", input)
        segment_length, exec_time = parse_data(output)
        segments.append(segment_length)
        exec_times.append(exec_time)

    plot_bar(inputs, exec_times, plot_file)

    print("Cleaning up...")
    os.remove(output_file)
    print("Finished!")