#! /usr/bin/python3

import random
import itertools
import numpy as np

def new_function(id=None, start_time=None, end_time=None, duration=None, hit=True):
    return {
        'id': id,
        'starttime': start_time,
        'endtime': end_time,
        'duration': duration,
        'hit': hit
    }

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

def generate_order_transition(weight, functions):
    n_functions = len(functions)
    res = [ [0 for _ in functions ] for _ in functions ]
    
    for x, y in itertools.product(range(n_functions), range(n_functions)):
        res[x][y] = ordered(x, y, n_functions, weight=weight)
    return res

def generate_sched(functions, transition_proba, length=10):
    res = []
    prev = random.choice(functions)['id']
    # prev = 0
    res.append(functions[prev].copy())

    for _ in range(length-1):
        prev = random.choices(functions, transition_proba[prev], k=1)[0]['id']
        res.append(functions[prev].copy())
    return res

# random_transitions = generate_order_transition(.5, functions)
# inversed_transitions = generate_order_transition(.9, functions)

N_FUNCTIONS = 3
generator=random.randint
args={'a': 50, 'b': 100}
completion_times = {i: generator(**args) for i in range(N_FUNCTIONS)}
print(completion_times)
print("Optimal latency:", sum(completion_times.values()))

def run_simulation(
    n_functions=N_FUNCTIONS, n_cores=2, trace_len=10, transition_proba=1,
    skip_duration=2):

    functions = [new_function(id=i, duration=completion_times[i]) for i in range(n_functions)]

    transitions = generate_order_transition(transition_proba, functions)

    if type(transition_proba) == list:
        transitions = transition_proba

    traces = [generate_sched(functions, transitions, length=trace_len) for _ in range(n_cores)]
    n_cores = len(traces)

    def new_job():
        return [None for i in range(n_functions)]

    working_jobs = []
    cpu_clocks = [0 for _ in range(n_cores)]
    cpu_cursor = [0 for _ in range(n_cores)]
    core_done = [False for _ in range(n_cores)]
    max_endtime = 0

    def all_core_done():
        for is_done in core_done:
            if not is_done:
                return False
        return True

    while not all_core_done():
        for core in range(n_cores):
            if cpu_cursor[core] >= trace_len:
                core_done[core] = True
                continue

            f = traces[core][cpu_cursor[core]]
            if f['id'] == 0: # First function. Starting processing...
                # Update functions' info
                duration = completion_times[f['id']]
                f['starttime'] = cpu_clocks[core]
                endtime = cpu_clocks[core] + duration
                f['endtime'] = endtime

                # Update jobs' info
                job = new_job()
                job[f['id']] = f
                working_jobs.append(job)
                max_endtime = max(endtime, max_endtime)

            else: # Intermediate functions. Continue working on jobs...
                # check if can work on something...
                current = cpu_clocks[core]
                skipped = True
                work_on = None

                for i, job in enumerate(working_jobs):
                    if job[f['id']] is None: # Step not already done
                        if job[f['id']-1] is None: # Last step not done, skip...
                            continue
                        else: # last step completed:
                            if job[f['id']-1]['endtime'] > current: # Last step not finished
                                continue
                            else: # Last step finished and current step not done yet
                                skipped = False
                                work_on = i
                    else: # Step already done:
                        continue

                # Update functions' info
                if skipped: # Did not processed any work
                    duration = skip_duration
                    f['hit'] = False
                else:
                    duration = completion_times[f['id']]

                f['starttime'] = current
                endtime = cpu_clocks[core] + duration
                f['endtime'] = endtime
                
                # Update jobs' info
                if not skipped: # Job entirely processed.
                    job = working_jobs[work_on]
                    job[f['id']] = f
                max_endtime = max(endtime, max_endtime)

            # Update core's clock and cursor
            cpu_clocks[core] += duration
            cpu_cursor[core] += 1

    return traces, working_jobs, max_endtime

def compute_metrics(working_jobs):
    job_dones = 0
    latencies = []
    
    def is_done(job):
        for step in job:
            if step is None:
                return False
        return True
    
    def compute_latency(job):
        return job[-1]['endtime'] - job[0]['starttime']
    
    for job in working_jobs:
        if is_done(job):
            job_dones += 1
            latencies.append(compute_latency(job))
    
    return job_dones, latencies

def format_job(job):
    def format_step(step):
        if step is not None:
            return f"({step['starttime']:3d}; {step['endtime']:3d})"
        else:
            return "(None, None)"
    return " -> ".join([format_step(step) for step in job])

def format_trace_line(trace_line):
    def format_func(f):
        return f"{f['id']} --> ({f['starttime']:3d}, {f['endtime']:3d})"
    return "\n\t".join([f"{i:2d}: {format_func(f)}" for i, f in enumerate(trace_line)])

def compute_hit_ratios(traces):
    hits = 0
    for cpu_line in traces:
        for f in cpu_line:
            hits += 1 if f['hit'] else 0
    total = len(traces) * len(traces[0])
    return hits * 100 / total


TRACE_LEN = 10_000
SKIP_DURATION = 10
RUNS = 2

# var: transition_proba, skip_duration, n_functions, n_cores, runs
transitions = [.5, .7, .9, 1]
skip_duration = [0, 1, 10, 20, 50, 80, 100]
nf = list(range(3, 10))
n_cores = [4, 8, 16, 32, 64]
runs = 50

for t, s, n, c in itertools.product(transitions, skip_duration, nf, n_cores):
    for run in range(runs):
        traces, working_jobs, max_endtime = run_simulation(transition_proba=.5, trace_len=TRACE_LEN, skip_duration=SKIP_DURATION)
        total_work, latencies = compute_metrics(working_jobs)
        print(",".join(
            t, s, n, c, run,
            total_work, max_endtime, np.quantile(latencies, .99), compute_hit_ratios(traces)
        ))
#         # output = config, total_work, max_endtime, tail_latency, hit_ratio


# # for i, core_line in enumerate(traces):
# #     print(f"Core {i:2d}:\n\t{format_trace_line(core_line)}")
# # print()
# # print('\n'.join([f"{i:6d}: {format_job(j)}" for i, j in enumerate(working_jobs) if j[-1] is not None]))

# traces, working_jobs, max_endtime = run_simulation(transition_proba=.7, trace_len=TRACE_LEN, skip_duration=SKIP_DURATION)
# total_work, latencies = compute_metrics(working_jobs)
# print(total_work, max_endtime)
# print("Throughput:", total_work * 100 / max_endtime)
# print("Tail (99th) Latency:", np.quantile(latencies, .5))
# print(f"Hit ratio: {compute_hit_ratios(traces)}%")

# # for i, core_line in enumerate(traces):
# #     print(f"Core {i:2d}:\n\t{format_trace_line(core_line)}")
# # print()
# # print('\n'.join([f"{i:6d}: {format_job(j)}" for i, j in enumerate(working_jobs)]))
