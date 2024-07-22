import re
import subprocess
import time
import os
import argparse
import sys
import numpy
import pandas


def get_command_arguments():
    """ Read input variables and parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Run CPU benchmarks for any slurm job that outputs a time',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-n', '--job-name-prefix', type=str, required=True, help='common name prefix for jobs')
    parser.add_argument('-c', '--initial-cpu-runs', type=str, default="1,2,4,8,16,32", help='initial cpu runs to determine optimal CPU use | NOTE: optimal CPU for criteria will only be calculated within the lowest and highest integers in this list (default: "1,2,4,8,16,32")')
    parser.add_argument('-s', '--slurm-partition', type=str, default="shared", help='slurm partition to run on')
    parser.add_argument('-o', '--optimization-runs', type=int, default="2", help='maximum optimization runs')
    parser.add_argument('-j', '--jobs-per-cpu-run', type=int, default="10", help='jobs per cpu run, more jobs means more accurate numbers')
    parser.add_argument('-f', '--filter-criteria', type=str, default="runtime", choices=["runtime", "efficiency"], help='filtering criteria')
    parser.add_argument('-m', '--margin-of-improvement', type=float, default=0.05, help='how much does the runtime speed / efficiency (in decimal) need to decreased by in order to be seen as an improvement? | bounded between [0, inf)')

    args = parser.parse_args()
    return args


def bprint(output):
    subprocess.call(["echo", str(output)])


def create_benchmark(args, cpus: int, partition: str):
    templatefile = open("test/templates/{args.job_name_prefix}.sh", "r")

    # templating filling
    filecontents = templatefile.read()
    filecontents = filecontents.replace("[|{CPUS}|]", str(cpus))
    filecontents = filecontents.replace("[|{PARTITION}|]", partition)
    filecontents = filecontents.replace("[|{JOBS}|]", args.jobs_per_cpu_run)

    open(f"test/templates/{args.job_name_prefix}-{str(cpus)}.sh", "w").write(filecontents)


def run_benchmark(args, cpus, partition):
    create_benchmark(args, cpus, partition)
    script = os.environ["SLURM_SUBMIT_DIR"] + f"/templates/{args.job_name_prefix}-" + str(cpus) + ".sh"
    process = subprocess.Popen(["sbatch", script])
    while process.poll() is None:
        pass
    bprint(cpus)


def processnames():
    # %x.o%A.%a.%N
    return str(subprocess.check_output(["squeue", "-u", os.environ["USER"], "-o", "%j.o%A"]))


def countbmsrunning(args):
    return len([m.start() for m in re.finditer(args.job_name_prefix, processnames())])


def wait_for_benchmark_completion(args):
    # Get names of processes running
    bprint(countbmsrunning(args))
    while countbmsrunning(args) != 0:
        time.sleep(15)
        bprint(str(countbmsrunning(args)) + " benchmarks still running")


def harvestbenchmarklogs():
    rawdata = {}
    benchmarklogging = open("benchmarks.log", "r")
    for line in benchmarklogging:
        line = line.strip("\n")
        splitline = line.split(",")
        cores: int = int(splitline[0])
        times: float = float(splitline[1])
        if rawdata[cores] is None:
            rawdata[cores] = [times]
        else:
            rawdata[cores] = rawdata[cores].append(times)
    averageddata = {}
    for key in rawdata.keys():
        averageddata[key] = sum(rawdata[key]) / len(rawdata[key])
    return rawdata, averageddata


def getbestval(rawdata):
    corelist = sorted(rawdata.keys())
    corecount = -1
    minseconds = sys.maxsize
    for i in corelist:
        if rawdata[i] < minseconds:
            minseconds = rawdata[i]
            corecount = i
    return minseconds, corecount

def getbestrange(rawdata):
    corelist = sorted(rawdata.keys())
    avgrange = []
    lowestavg = sys.maxsize
    for i in range(0, len(corelist)):
        avg = (corelist[i] + corelist[i + 1]) / 2
        if avg < lowestavg:
            lowestavg = avg
            avgrange = [i, i + 1]
    return lowestavg, avgrange

def main():
    inittime = time.time()
    if os.path.isfile("benchmarks.log"):
        os.rename("benchmarks.log", f"{str(inittime)}.benchmarks.log")
    if os.path.isfile("output.log"):
        os.rename("output.log", f"{str(inittime)}.output.log")
    args = get_command_arguments()

    benchmarkcpuruns = []
    for i in args.initial_cpu_runs.split(sep=","):
        benchmarkcpuruns.append(int(i))

    for i in benchmarkcpuruns:
        run_benchmark(args, i, args.slurm_partition)

    bprint("Attempted task creation")

    bprint(processnames())

    while countbmsrunning(args) != len(benchmarkcpuruns):
        time.sleep(1)

    bprint("Initial benchmarks started.")

    # behcmark procesing
    bprint(processnames())
    scriptlist = processnames().split("\\n")
    for i in range(0, len(scriptlist)):
        scriptlist[i] = scriptlist[i].strip("\'").strip("b")
    p = re.compile(args.job_name_prefix)
    scriptlist = [x for x in scriptlist if p.match(x)]
    bprint(scriptlist)

    wait_for_benchmark_completion(args)

    bprint("Initial benchmarks completed.")

    # NEW STRAT IDEA: IDENTIFY LOWEST NUMBER AND CREATE TWO JOBS ADJACENT TO THAT CPU COUNT | CONTINUE UNTIL END CRITERIA REACHED
    # TODO: ADD MARGIN OF IMPROVEMENT METRIC

    for _ in range(0, args.optimization_runs):
        rawdata, averageddata = harvestbenchmarklogs()
        seconds, corecount = getbestval(rawdata)
        if corecount == sorted(list(rawdata.keys()))[0]:
            idx = sorted(list(rawdata.keys())).index(corecount)
            highboundcpus = sorted(rawdata.keys())[idx + 1]
            if highboundcpus - corecount == 1:
                bprint(corecount)
                # corecount is the best value
                return 0
            highcpu = int((corecount + highboundcpus) / 2)
            if highcpu in rawdata.keys():
                # corecount is the best value
                bprint(corecount)
                return 0
            else:
                run_benchmark(args, highcpu, "shared")
        elif corecount == sorted(list(rawdata.keys()))[-1]:
            idx = sorted(list(rawdata.keys())).index(corecount)
            lowboundcpus = sorted(rawdata.keys())[idx - 1]
            if corecount - lowboundcpus == 1:
                bprint(corecount)
                # corecount is the best value
                return 0
            lowcpu = int((corecount + lowboundcpus) / 2)
            if lowcpu in rawdata.keys():
                # corecount is the best value
                bprint(corecount)
                return 0
            else:
                run_benchmark(args, lowcpu, "shared")
        else:
            idx = sorted(list(rawdata.keys())).index(corecount)
            lowboundcpus = sorted(list(rawdata.keys()))[idx - 1]
            highboundcpus = sorted(rawdata.keys())[idx + 1]
            if highboundcpus - corecount == 1 or corecount - lowboundcpus == 1:
                bprint(corecount)
                # corecount is the best value
                return 0

            lowcpu = int((corecount + lowboundcpus) / 2)
            highcpu = int((corecount + highboundcpus) / 2)
            flag = False
            if not (lowcpu in rawdata.keys()):
                run_benchmark(args, lowcpu, "shared")
                flag = True
            if not (highcpu in rawdata.keys()):
                run_benchmark(args, highcpu, "shared")
                flag = True
            if not flag:
                # corecount is the best value
                bprint(corecount)
                return 0



    # shrunkRange = [0, sys.maxsize]
    #
    # rawdata, averageddata = harvestbenchmarklogs()
    # lowestavg, avgrange = getbestrange(rawdata)
    # shrunkRange = avgrange
    #
    # while True:
    #     run_benchmark(args, int((avgrange[0] + avgrange[1]) / 2), args.slurm_partition)
    #     rawdata, averageddata = harvestbenchmarklogs()
    #     lowestavg, avgrange = getbestrange(rawdata)
    #     if not (shrunkRange[0] <= avgrange[0] and shrunkRange[1] >= avgrange[1]):
    #         pass
    #         # WE'VE REACHED OPTIMUM? ADD LOGIC LATER AKA OUTPUT FILE AND RETURN
    #     if avgrange[1] - avgrange[0] == 0:
    #         pass
    #         # avgrange[0] is optimal CPUs
    #     elif avgrange[1] - avgrange[0] == 1:
    #         pass
    #         # min(avgrange[0], avgrange[1]) is optimal CPUs

    return 0


if __name__ == '__main__':
    sys.exit(main())
