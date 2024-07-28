import math
import os
import re
import subprocess
import sys
import time

import numpy as np
import scipy


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
    rawdata, averageddata = harvestbenchmarklogs()

    speedupdict = {}
    maxsec = max(averageddata.values())
    for key in averageddata.keys():
        speedupdict[key] = maxsec / averageddata[key]

    popt, pcov = scipy.optimize.curve_fit(func, np.array(list(speedupdict.keys())), np.array(list(speedupdict.values())), bounds=[0, np.inf])
    nvalue = popt[0]
    newcorecheck = math.ceil(invfunc(3, nvalue))

    # return upper bound if upper bound unable to reach
    if func(newcorecheck, nvalue) > max(speedupdict.values()):
        maxval = max(speedupdict.values())
        for key in speedupdict.keys():
            if speedupdict[key] == maxval:
                bprint(key)
                return 0

    bestvalue = closestspeedup(speedupdict, 3)
    if newcorecheck in averageddata.keys():
        newcorecheck = math.floor(invfunc(3, nvalue))
        if newcorecheck in averageddata.keys():
            bprint(bestvalue)
            return 0

    run_benchmark(args, newcorecheck, "shared")



if __name__ == "__main__":
    main()