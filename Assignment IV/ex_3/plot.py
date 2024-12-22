import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

def compile_cuda_code(source_file, output_file):
    print(f"Compiling {source_file}...")
    compile_command = f"nvcc {source_file} -O3 -o {output_file} -lcublas -lcusparse"
    result = subprocess.run(compile_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr.decode()}")
        exit(1)
    print("Compilation successful.")

def run_cuda_program(executable, dim, nstep):
    print(f"Running {executable} {dim} {nstep}")
    run_command = f"{executable} {dim} {nstep}"
    result = subprocess.run(run_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Execution failed:\n{result.stderr.decode()}")
        exit(1)
    print(f"{executable} finished with return code {result.returncode}")
    return result.stdout.decode()

def parse_data(output):
    lines = output.splitlines()
    time_row = lines[5].split()
    time_in_microseconds = int(time_row[time_row.index("Elasped") + 1])
    return time_in_microseconds


def plot_bar(dims, nsteps, flops, filename):
    print(f"Plotting the output to {filename}")
    plt.figure(figsize=(10, 6))
    plt.plot(dims, flops, marker="o", linewidth=2, markersize=8)

    plt.xlabel("Matrix Dimension, Time Steps (dimX, nsteps)")
    plt.ylabel("FLOPS")
    plt.title("SpMV Performance")

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    name        = "heat_equation"
    source_file = name + ".cu"
    output_file = name
    plot_file   = name + ".png"

    # Plot values for question 1
    #dims = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    #nsteps = [1000 for i in range(11)]

    # Plot values for question 2
    dims = [1024 for i in range(11)]
    nsteps = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    flops = []

    compile_cuda_code(source_file, output_file)

    for dim, nstep in zip(dims, nsteps):
        output = run_cuda_program(f"./{output_file}", dim, nstep)
        exec_time = parse_data(output)
        avg_time = float(exec_time) / nstep
        ops_per_spmv = 2.0 * (3 * dim - 6)
        flops.append((ops_per_spmv * nstep) / (exec_time * 1e-6))


    plot_bar(dims, nsteps, flops, plot_file)

    print("Cleaning up...")
    os.remove(output_file)
    print("Finished!")