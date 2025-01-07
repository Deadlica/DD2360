import argparse
import os
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import statistics

BIN_DIR = "build/"

def set_working_directory():
    """Set the working directory to the parent directory of the script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, ".."))


def build_binaries(args):
    """Build the raytracer binaries."""
    print("Building binaries...")
    subprocess.run(
        ["make", "raytracer_gpu"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    if args.compare_cpu_gpu:
        subprocess.run(
            ["make", "raytracer_cpu"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    if args.compare_gpu_mem:
        subprocess.run(
            ["make", "raytracer_gpu_unified"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


def run_binary(binary, width, runs):
    """Run a binary multiple times using 'make run_*' and return the rendering times."""
    times = []
    print(f"\t\t{binary.split(BIN_DIR)[-1]}:")
    for i in range(runs):
        print(f"\r\t\t\t Run {i+1}/{runs}", end='', flush=True)
        result = subprocess.run(
            [binary, str(width)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )
        # Rendering time is printed to std::clog
        stdclog = result.stderr
        match = re.search(r"Rendering time: ([\d.]+) seconds\.", stdclog)
        if match:
            times.append(float(match.group(1)))
    print()
    return times


def cleanup(args):
    """Clean up build artifacts and generated files."""
    print("Cleaning up...")
    if args.clean:
        subprocess.run(["make", "clean"], check=True)

    for file in os.listdir("."):
        if file.endswith(".ppm"):
            os.remove(file)
    print("Cleanup complete.")


def benchmark(args):
    """Benchmark the raytracer based on the provided flags."""
    runs = args.runs
    widths = args.widths
    results = {}

    if args.compare_cpu_gpu:
        print("Comparing CPU and GPU versions...")
        cpu_times = []
        gpu_times = []
        for width in widths:
            print(f"\tWidth: {width}")
            cpu_times.append(run_binary(BIN_DIR + "raytracer_cpu", width, runs))
            gpu_times.append(run_binary(BIN_DIR + "raytracer_gpu", width, runs))
        results["CPU vs GPU"] = {"CPU": cpu_times, "GPU": gpu_times}

    if args.compare_gpu_mem:
        print("Comparing GPU with and without unified memory...")
        gpu_times = []
        gpu_unified_times = []
        for width in widths:
            print(f"\tWidth: {width}")
            gpu_times.append(run_binary(BIN_DIR + "raytracer_gpu", width, runs))
            gpu_unified_times.append(run_binary(BIN_DIR + "raytracer_gpu_unified", width, runs))
        results["GPU Memory Comparison"] = {
            "Without Unified Memory": gpu_times,
            "With Unified Memory": gpu_unified_times,
        }

    return results


def plot_results(results, widths):
    """Plot benchmark results using lines instead of boxplots."""
    for test, data in results.items():
        plt.figure(figsize=(10, 6))
        labels = list(data.keys())

        is_cpu_gpu_comparison = any("cpu" in label.lower() or "gpu" in label.lower() for label in labels)

        for i, label in enumerate(labels):
            times = data[label]
            means = [np.mean(times_per_width) for times_per_width in times]
            std_errs = [statistics.stdev(times_per_width) / np.sqrt(len(times_per_width)) for times_per_width in times]

            plt.plot(widths, means, marker="o", label=label)
            plt.errorbar(widths, means, yerr=std_errs, fmt="o")

        plt.title(test)
        plt.ylabel("Rendering Time (seconds)")
        plt.xlabel("Image Width")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="Configuration")

        if is_cpu_gpu_comparison:
            plt.yscale("log")

        plt.tight_layout()

        plt.savefig(f"{test.replace(' ', '_').lower()}.png")
        print(f"Plot saved as {test.replace(' ', '_').lower()}.png")


def main():
    set_working_directory()

    parser = argparse.ArgumentParser(description="Benchmark the raytracer.")
    parser.add_argument(
        "-ccg",
        "--compare-cpu-gpu",
        action="store_true",
        help="Compare the rendering execution time between the CPU and GPU versions.",
    )
    parser.add_argument(
        "-cgm",
        "--compare-gpu-mem",
        action="store_true",
        help="Compare the rendering execution time between the GPU version with and without unified memory.",
    )
    parser.add_argument(
        "-r", "--runs", type=int, default=5, help="Number of runs for each render size (default: 5, minimum: 2)."
    )
    parser.add_argument(
        "-w", "--widths", type=int, nargs='+', default=[400, 800, 1200],
        help="List of image widths to benchmark (default: [400, 800, 1200])."
    )
    parser.add_argument(
        "-c", "--clean", action="store_true", default=False, help="Cleanup build directory when finished."
    )
    args = parser.parse_args()

    if not args.compare_cpu_gpu and not args.compare_gpu_mem:
        print("Error: At least one comparison flag (--compare-cpu-gpu or --compare-gpu-mem) must be provided.")
        parser.print_help()
        exit(0)

    if args.runs < 2:
        print("Error: --runs needs to be at least 2.")
        parser.print_help()
        exit(0)

    build_binaries(args)
    results = benchmark(args)
    plot_results(results, args.widths)
    cleanup(args)


if __name__ == "__main__":
    main()
