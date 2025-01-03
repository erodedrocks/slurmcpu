import math
import re
import subprocess
import time
import os
import argparse
import sys

import numpy as np
import scipy


def get_command_arguments():
    """ Read input variables and parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Run CPU benchmarks for any slurm job that outputs a time',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-n', '--job-name-prefix', type=str, required=True, help='common name prefix for jobs')
    parser.add_argument('-c', '--initial-cpu-runs', type=str, default="1,4,16,32",
                        help='initial cpu runs to determine optimal CPU use | NOTE: optimal CPU for criteria will only be calculated within the lowest and highest integers in this list (default: "1,2,4,8,16,32")')
    parser.add_argument('-s', '--slurm-partition', type=str, default="shared", help='slurm partition to run on')
    parser.add_argument('-o', '--optimization-runs', type=int, default="2", help='maximum optimization runs')
    parser.add_argument('-j', '--jobs-per-cpu-run', type=int, default="10",
                        help='jobs per cpu run, more jobs means more accurate numbers')
    parser.add_argument('-f', '--filter-criteria', type=str, default="goal_speedup",
                        choices=["runtime", "efficiency_bound", "goal_speedup"], help='filtering criteria')
    parser.add_argument('-i', '--filter-information', type=float, default=0,
                        help='filtering criteria arguments | runtime_bound: speed (in seconds) that it should be below | efficiency_bound: percentage that goal must be above compared to most efficient sample | goal_speedup: the goal speedup')
    parser.add_argument('-b', '--bound-margin', type=float, required=True,
                        help='margin for the bound (example: a bound-margin = 0.1, filter-criteria = goal_speedup, filter-information = 3 means goal should be between 2.7x and 3.3x faster | runtime_bound: bound margin for speed | efficiency_bound: bound margin for efficiency | goal_speedup: margin for speedup')

    args = parser.parse_args()
    return args


def bprint(output):
    subprocess.call(["echo", str(output)])


def create_benchmark(args, cpus: int, partition: str):
    templatefile = open(f"test/templates/{args.job_name_prefix}.sh", "r")

    # templating filling
    filecontents = templatefile.read()
    filecontents = filecontents.replace("[|{CPUS}|]", str(cpus))
    filecontents = filecontents.replace("[|{PARTITION}|]", partition)
    filecontents = filecontents.replace("[|{JOBS}|]", str(args.jobs_per_cpu_run))

    open(f"test/templates/{args.job_name_prefix}-{str(cpus)}.sh", "w").write(filecontents)


def run_benchmark(args, cpus, partition):
    create_benchmark(args, cpus, partition)
    script = os.environ["SLURM_SUBMIT_DIR"] + f"/templates/{args.job_name_prefix}-" + str(cpus) + ".sh"
    process = subprocess.Popen(["sbatch", script])
    while process.poll() is None:
        pass
    bprint(f"CPU Count Run: {str(cpus)}")


def processnames():
    # %x.o%A.%a.%N
    return str(subprocess.check_output(["squeue", "-u", os.environ["USER"], "-o", "%j.o%A"]))


def countbmsrunning(args):
    return len([m.start() for m in re.finditer(args.job_name_prefix, processnames())])


def wait_for_benchmark_completion(args):
    # Get names of processes running
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
        if rawdata.get(cores) is None:
            rawdata[cores] = [times]
        else:
            temp = rawdata[cores]
            temp.append(times)
            rawdata[cores] = temp
    averageddata = {}
    for key in rawdata.keys():
        averageddata[key] = sum(rawdata[key]) / len(rawdata[key])
    return rawdata, averageddata


def getbestval(avgdata):
    corelist = sorted(avgdata.keys())
    corecount = -1
    minseconds = sys.maxsize
    for i in corelist:
        if avgdata[i] < minseconds:
            minseconds = avgdata[i]
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


def calculateefficiency(averageddata, seconds, corecount):
    loweff = sys.maxsize
    for key in averageddata.keys():
        val = averageddata.get(key)
        if val * key < loweff:
            loweff = val * key
    return calculateefficiencywithbaseline(loweff, seconds, corecount)


def calculateefficiencywithbaseline(loweff, seconds, corecount):
    return loweff / (seconds * corecount)


def bestefficiency(averageddata, threshold):
    loweff = sys.maxsize
    for key in averageddata.keys():
        val = averageddata.get(key)
        if val * key < loweff:
            loweff = val * key
    newlowesteff = sys.maxsize
    lastcore = -1
    for key in sorted(averageddata.keys()):
        eff = calculateefficiencywithbaseline(loweff, averageddata.get(key), key)
        if eff <= newlowesteff:
            newlowesteff = eff
            if newlowesteff < threshold:
                # output cores
                return lastcore
        lastcore = key
    return -1

def bestruntime(averageddata, threshold):
    min(averageddata.keys(), key=lambda n: averageddata[n] - threshold)
    for key in sorted(averageddata.keys()):
        if averageddata.get(key) <= threshold:
            return key, averageddata.get(key)
    return False, -1

def getKey(data, value):
    for i in data.keys():
        if data[i] == value:
            return i
    return -1

def func(x, a):
    return x / (1 + (a * (x - 1)))

def invfunc(y, a):
    return (y * (1 - a)) / (1 - (a * y))

def closestspeedup(speedupdict, threshold):
    sortedvallist = sorted(list(speedupdict.values()))
    pastkey = "n"
    for val in sortedvallist:
        if val >= threshold:
            return pastkey
        for key in speedupdict.keys():
            if speedupdict[key] == val:
                pastkey = key
    for key in speedupdict.keys():
        if speedupdict[key] == sortedvallist[-1]:
            return key
    return -1

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
    time.sleep(5)

    bprint("Initial benchmarks started.")

    wait_for_benchmark_completion(args)

    bprint("Initial benchmarks completed.")

    # NEW STRAT IDEA: IDENTIFY LOWEST NUMBER AND CREATE TWO JOBS ADJACENT TO THAT CPU COUNT | CONTINUE UNTIL END CRITERIA REACHED
    # TODO: ADD MARGIN OF IMPROVEMENT METRIC

    # if args.filter_criteria == "efficiency_bound" and calculateefficiency(averageddata, seconds, corecount) < args.filter_information:
    #     # out reaches the efficiency value the quickest
    #     out = bestefficiency(averageddata, args.filter_information)
    #     bprint(out)
    #     return 0

    if args.filter_criteria == "runtime":
        bestCoreCount = -1
        for _ in range(0, args.optimization_runs):
            rawData, averagedData = harvestbenchmarklogs()
            seconds, bestCoreCount = getbestval(averagedData)
            if bestCoreCount == sorted(list(rawData.keys()))[0]:
                idx = sorted(list(rawData.keys())).index(bestCoreCount)
                highboundcpus = sorted(rawData.keys())[idx + 1]
                if highboundcpus - bestCoreCount == 1:
                    bprint(bestCoreCount)
                    # corecount is the best value
                    return 0
                highcpu = int((bestCoreCount + highboundcpus) / 2)
                if highcpu in rawData.keys():
                    # corecount is the best value
                    bprint(bestCoreCount)
                    return 0
                else:
                    run_benchmark(args, highcpu, "shared")
                    wait_for_benchmark_completion(args)
            elif bestCoreCount == sorted(list(rawData.keys()))[-1]:
                idx = sorted(list(rawData.keys())).index(bestCoreCount)
                lowboundcpus = sorted(rawData.keys())[idx - 1]
                if bestCoreCount - lowboundcpus == 1:
                    bprint(bestCoreCount)
                    # corecount is the best value
                    return 0
                lowcpu = int((bestCoreCount + lowboundcpus) / 2)
                if lowcpu in rawData.keys():
                    # corecount is the best value
                    bprint(bestCoreCount)
                    return 0
                else:
                    run_benchmark(args, lowcpu, "shared")
                    wait_for_benchmark_completion(args)
            else:
                idx = sorted(list(rawData.keys())).index(bestCoreCount)
                lowboundcpus = sorted(list(rawData.keys()))[idx - 1]
                highboundcpus = sorted(rawData.keys())[idx + 1]
                if highboundcpus - bestCoreCount == 1 or bestCoreCount - lowboundcpus == 1:
                    bprint(bestCoreCount)
                    # corecount is the best value
                    return 0

                lowcpu = int((bestCoreCount + lowboundcpus) / 2)
                highcpu = int((bestCoreCount + highboundcpus) / 2)

                # sanity checks (last part should never occur)
                flag = False
                if not (lowcpu in rawData.keys()):
                    run_benchmark(args, lowcpu, "shared")
                    flag = True
                if not (highcpu in rawData.keys()):
                    run_benchmark(args, highcpu, "shared")
                    flag = True
                if not flag:
                    # corecount is the best value
                    bprint(bestCoreCount)
                    return 0
                wait_for_benchmark_completion(args)
        bprint(bestCoreCount)
    elif args.filter_criteria == "efficiency_bound":
        for _ in range(0, args.optimization_runs):
            rawData, averagedData = harvestbenchmarklogs()
            bestCoreCount = bestefficiency(averagedData, args.filter_information)
            if bestCoreCount == -1:
                # last val best val!
                bprint(sorted(list(rawData.keys()))[-1])
                return 0
            else:
                idx = sorted(list(rawData.keys())).index(bestCoreCount)
                lowboundcpus = sorted(list(rawData.keys()))[idx + 1]
                if lowboundcpus - bestCoreCount == 1:
                    # bestCoreCount is the best value
                    bprint(bestCoreCount)
                    return 0
                lowcpu = int((bestCoreCount + lowboundcpus) / 2)
                if not (lowcpu in rawData.keys()):
                    run_benchmark(args, lowcpu, "shared")
                else:
                    # bestCoreCount is the best value
                    bprint(bestCoreCount)
                    return
            wait_for_benchmark_completion(args)
        rawData, averagedData = harvestbenchmarklogs()
        bprint(bestefficiency(averagedData, args.filter_information))
    elif args.filter_criteria == "goal_speedup":
        goalspeedup = args.filter_information
        speedupleeway = args.bound_margin

        for _ in range(0, args.optimization_runs):
            rawData, averagedData = harvestbenchmarklogs()
            speedupTable = {}
            maxElapsed = max(averagedData.values())
            for key in averagedData.keys():
                speedupTable[key] = maxElapsed / averagedData[key]
            optimalValues, pcov = scipy.optimize.curve_fit(func, np.array(list(speedupTable.keys())), np.array(list(speedupTable.values())), bounds=[0, np.inf])
            optimalA = optimalValues[0]
            predictedCores = math.ceil(invfunc(goalspeedup, optimalA))

            bprint(f"Optimal A: {str(optimalA)} | InvFunc output: {str(predictedCores)}")
            bprint(f"speedupTable: {str(speedupTable)}")

            # return upper bound if predicted value is unable to be reached
            if func(predictedCores, optimalA) > max(speedupTable.values()):
                maxElapsed = max(speedupTable.values())
                for key in speedupTable.keys():
                    if speedupTable[key] == maxElapsed:
                        bprint(key)
                        return 0

            bestCoreCount = closestspeedup(speedupTable, goalspeedup)  # closest current value
            if predictedCores in averagedData.keys():
                if math.fabs(averagedData[predictedCores] - goalspeedup) <= speedupleeway * goalspeedup:  # if within acceptable margin
                    bprint(predictedCores)
                    return 0
                # check if the floor is also in the keys
                predictedCores = math.floor(invfunc(goalspeedup, optimalA))
                if predictedCores in averagedData.keys():
                    if math.fabs(averagedData[predictedCores] - goalspeedup) <= speedupleeway * goalspeedup:  # if within acceptable margin
                        bprint(predictedCores)
                        return 0

                    if averagedData[predictedCores] < goalspeedup:
                        while predictedCores in averagedData.keys() and predictedCores <= max(averagedData.keys()):
                            predictedCores += 1
                    else:
                        while predictedCores in averagedData.keys() and predictedCores >= min(averagedData.keys()):  # min(averageddata.keys()) should always be 1
                            predictedCores -= 1

                    if predictedCores > max(averagedData.keys()) or predictedCores < min(averagedData.keys()):
                        bprint(bestCoreCount)
                        return 0

            run_benchmark(args, predictedCores, "shared")
            wait_for_benchmark_completion(args)
        rawData, averagedData = harvestbenchmarklogs()
        bprint(bestefficiency(averagedData, goalspeedup))
    return 0


if __name__ == '__main__':
    sys.exit(main())
