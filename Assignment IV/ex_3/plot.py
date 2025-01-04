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

def parse_data(output, metric="flops"):
    lines = output.splitlines()
    if metric == "flops":
        time_row = lines[5].split()
        time_in_microseconds = int(time_row[time_row.index("Elasped") + 1])
        return time_in_microseconds
    else: # metric == "error"
        error_row = lines[6].split()
        error_val = float(error_row[-1])
        return error_val

def plot_results(x_values, y_values, filename, metric="flops", dim=None, nstep=None):
    print(f"Plotting the output to {filename}")
    plt.figure(figsize=(10, 6))

    if metric == "flops":
        plt.plot(x_values, y_values, marker="o", linewidth=2, markersize=8)
        plt.xlabel("Matrix Dimension (dimX)")
        plt.ylabel("FLOPS")
        plt.title("SpMV Performance")
    else:  # metric == "error"
        plt.semilogx(x_values, y_values, marker="o", linewidth=2, markersize=8)
        plt.xlabel("Number of Time Steps")
        plt.ylabel("Relative Error")
        plt.title(f"Relative Error Convergence (dimX={dim})")

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")

def calculate_flops(dim, nstep, exec_time):
    ops_per_spmv = 2.0 * (3 * dim - 6)
    total_time_seconds = exec_time * 1e-6
    return (ops_per_spmv * nstep) / total_time_seconds

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["flops", "error"]:
        print("Usage: python plot.py [flops|error]")
        sys.exit(1)

    metric      = sys.argv[1]
    name        = "heat_equation"
    source_file = name + ".cu"
    output_file = name
    plot_file   = f"{name}_{metric}.png"

    if metric == "flops":
        dims = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        nsteps = [1000] * len(dims)
        x_values = dims
    else:
        dim_x = 100
        dims = [dim_x for i in range(20)]
        nsteps = np.logspace(2, 4, 20, dtype=int)
        x_values = nsteps

    y_values = []
    compile_cuda_code(source_file, output_file)

    for dim, nstep in zip(dims, nsteps):
        output = run_cuda_program(f"./{output_file}", dim, nstep)
        if metric == "flops":
            exec_time = parse_data(output, "flops")
            y_values.append(calculate_flops(dim, nstep, exec_time))
        else:
            error = parse_data(output, "error")
            if error is not None:
                y_values.append(error)

    plot_results(x_values, y_values, plot_file, metric, dim_x if metric == "error" else None)

    print("Cleaning up...")
    os.remove(output_file)
    print("Finished!")