import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os

def compile_cuda_code(source_file, output_file, extra_flags="-DPLOT"):
    print(f"Compiling {source_file}...")
    compile_command = f"nvcc {source_file} -O3 -o {output_file} {extra_flags}"
    result = subprocess.run(compile_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr.decode()}")
        exit(1)
    print("Compilation successful.")

def run_cuda_program(executable, input_length):
    print(f"Running {executable}...")
    run_command = f"{executable} {input_length}"
    result = subprocess.run(run_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Execution failed:\n{result.stderr.decode()}")
        exit(1)
    print(f"{executable} finished with return code {result.returncode}")
    return result.stdout.decode()

def parse_histogram_data(output):
    lines = output.splitlines()
    input_length = int(lines[0].split()[-1])
    bins = [int(value) for value in lines[1:] if value.isdigit()]
    legend = lines[1] + "\n" + lines[2]
    return input_length, bins, legend

def plot_histogram(output_file, bins, input_length, legend):
    print(f"Plotting the output to {output_file}")
    x = np.arange(len(bins))
    plt.bar(x, bins, color="blue", label=legend)
    plt.title(f"Histogram for Input Length: {input_length}")
    plt.xlabel("Bins")
    plt.ylabel("Counts")
    plt.legend(loc="upper right")
    plt.savefig(output_file)

if __name__ == "__main__":
    name        = "histogram"
    source_file = name + ".cu"
    output_file = name
    plot_file   = name + ".png"
    input_length = 10000

    compile_cuda_code(source_file, output_file)

    output = run_cuda_program(f"./{output_file}", input_length)

    input_length, bins, legend = parse_histogram_data(output)

    plot_histogram(plot_file, bins, input_length, legend)

    print("Cleaning up...")
    os.remove(output_file)
    print("Finished!")
