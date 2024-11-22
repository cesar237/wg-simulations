#! /usr/bin/python3

import os
import numpy as np
import random
import scipy as sp
from matplotlib import pyplot as plt
import seaborn as sns
import itertools
import copy
import pandas as pd

def new_stage(id=None, completion_time=100, finishing_time=None):
    # completion = completion_time + random.randint(completion_time)
    
    return {
        'id': id,
        'completion_time': completion_time,
        'arrival_time': finishing_time - completion_time,
        'finishing_time': finishing_time
    }

def generate_sched(stages, transition_proba, length=15):
    """_summary_ To generate a trace
    """
    res = []
    prev = random.choice(stages)
    res.append(prev)
    
    for _ in range(length-1):
        prev = random.choices(stages, transition_proba[prev], k=1)[0]
        res.append(prev)
    return res

def gen_job(stages):
    return {i: None for i in stages}

def run_simulation_(stages, stages_completion, transitions, total_func_calls, n_cores):
    trace = [generate_sched(stages=stages, transition_proba=transitions, length=total_func_calls) for _ in range(n_cores)]
    lasts = [0 for _ in stages]
    todo = [gen_job(stages) for _ in range(total_func_calls * n_cores // 2)]
    for job in todo:
        job['last_update'] = -1

    last_stage_cpu_timestamp = [0 for _ in range(n_cores)]

    for workers in zip(*trace):
        workers = list(workers)
        workers.sort()

        for cpu, worker in enumerate(workers):
            timestamp = last_stage_cpu_timestamp[cpu] + stages_completion[worker]
            stage = new_stage(id=worker, completion_time=stages_completion[worker], finishing_time=timestamp)
            last_stage_cpu_timestamp[cpu] = timestamp

            if worker == stages[0]:
            # First function marks start of working on next job
                if todo[lasts[worker]][worker] is None:
                    todo[lasts[worker]][worker] = stage
                    lasts[worker] += 1
                else:
                    lasts[worker] += 1
                    todo[lasts[worker]][worker] = stage
            else:
                # If last job is not finished yet, then skip.
                last_stage = todo[lasts[worker]][worker-1]
                if last_stage is None or last_stage['finishing_time'] < stage['arrival_time']:
                    continue
                else:
                    if todo[lasts[worker]][worker] is None:
                        todo[lasts[worker]][worker] = stage
                        lasts[worker] += 1
                    else:
                        lasts[worker] += 1
                        todo[lasts[worker]][worker] = stage

    return todo, last_stage_cpu_timestamp

def count_done(works, stages):
    res = 0
    latencies = []
    first_stage = stages[0]
    last_stage = stages[-1]
    while works[res][last_stage] is not None:
        w = works[res]
        latencies.append(w[last_stage] - w[first_stage] + 1)
        res += 1
    return res, latencies


def compute_metrics(todo, last_timestamps, n_stages):
    job = todo[0]
    done_jobs = 0
    latencies = []

    def is_completed(job):
        i = 0
        while i < n_stages:
            if job[i] is None:
                return False
        return True
    
    def compute_latency(job):
        return job[-1]['finishing_time'] - job[0]['arrival_time']
    
    if is_completed(job):
        done_jobs += 1
        latencies.append(compute_latency(job))

    total_time = max(last_timestamps)

    throughput = done_jobs / total_time
    tail_latency = np.percentile(latencies, 99)
    mean_latency = np.mean(latencies)
    median_latency = np.median(latencies)

    return throughput, median_latency, mean_latency, tail_latency

def run_simulation(stages, stages_completion, transitions, total_func_calls, n_cores, runs=1):
    names = ['throughput', 'median_latency', 'mean_latency', 'tail_latency']
    data = [run_simulation_(stages, stages_completion, transitions, total_func_calls, n_cores) for _ in range(runs)]
    return pd.DataFrame(data=data, columns=names)

def rand(x, y, n):
    if x == y:
        return 0
    else:
        return 1/(n-1)

def ordered(x, y, n, weight=1):
    if x == y:
        return 0
    m = y - x
    if m < 0:
        m += n
    if m == 1:
        return weight
    else:
        return (1-weight)/(n-2)

def generate_order_transition(weight, stages):
    n_stages = len(stages)
    res = [ [0 for _ in stages ] for _ in stages ]
    
    for x, y in itertools.product(stages, stages):
        res[x][y] = ordered(x, y, n_stages, weight=weight)
    return res

n_stages = 4
stages_completion = [100, 300, 200, 400]
stages = [i for i in range(n_stages)]
n_cores = 32
total_func_calls = 100
proba = [ [0 for _ in stages] for _ in stages ]
random_transitions = [
    [0, .5, .5],
    [.5, 0, .5],
    [.5, .5, 0]
]
ordered_transitions = [
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
]
semi_ordered_transitions = [
    [0, .99, .01],
    [.01, 0, .99],
    [.99, .01, 0]
]

rand_weights = [.50, .75, .90, .99, 1]

random_transitions = copy.deepcopy(proba)
order_transition = {i: generate_order_transition(i, stages) for i in rand_weights}

for x, y in itertools.product(stages, stages):
    random_transitions[x][y] = rand(x, y, n_stages)

def order_transition(i):
    return generate_order_transition(i, stages)

# print("Optimal throughput:", n_cores/n_stages)
# print("Optimal latency:", n_stages)

print("Optimal:")
df_random = run_simulation(stages, stages_completion, order_transition(1), total_func_calls, n_cores, runs=15)
df_random.median()

print("Semi-ordered:")
df_semi_ordered = run_simulation(stages, stages_completion, random_transitions, total_func_calls, n_cores, runs=15)
df_semi_ordered.median()



# df_random = run_simulation(stages, order_transition(.5), total_time, n_cores, runs=1000)
# df_random.median()

# df_random = run_simulation(stages, order_transition(.9), total_time, n_cores, runs=1000)
# df_random.median()
