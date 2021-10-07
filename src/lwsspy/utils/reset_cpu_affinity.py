import psutil


def reset_cpu_affinity(verbose: bool = False):

    # Get main process
    p = psutil.Process()

    # Get current affinity
    if verbose:
        print("Current Affinity", p.cpu_affinity())

    # Get all CPUs
    all_cpus = list(range(0, psutil.cpu_count(logical=True), 1))

    # Set new affinity
    p.cpu_affinity(all_cpus)

    # Print new affinity
    if verbose:
        print("New Affinity", p.cpu_affinity())


if __name__ == "__main__":

    reset_cpu_affinity(True)
