import argparse
import subprocess


def get_command_arguments():
    """ Read input variables and parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Run CPU benchmarks for any slurm job that outputs a time',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-s', '--script', type=str, help='name of script to run')

    args = parser.parse_args()
    return args

def main():
    args = get_command_arguments()
    process = subprocess.Popen(["sbatch", args.script])

if __name__ == "__main__":
    main()