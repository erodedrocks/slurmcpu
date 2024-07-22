# CPUOpt SLURM Implementation
Instructions for using your own SLURM task within this system:

1. Create a template file similar to the one present in test/templates/torch-model-training.sh
2. Append to the benchmarks.log file with a string in this format within either your .sh file or your python/other script your run
    * [cores],[time in seconds]
3. Place the template SLURM .sh file into the templates folder
4. Run the cpu-benchmark.py file with the desired arguments and read the output.log file for collated data or benchmarks.log for raw data.

If any bugs are noticed, please create a pull request or submit a log. Thank you!