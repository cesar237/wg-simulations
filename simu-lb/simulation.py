#! /usr/bin/python3

# Event: type, servcice_time
# Event type schedulé
# Generate trace = timestamped packet arrivals(?)
# Echelle de la micro seconde sur 5 secondes (5e10^6)
# Throughput = packet par secondes (distribution uniforme des paquets sur la durée totale de la trace)
# Number of event types
# cores: queue, current
# clock
# incrémenter la clock. Faire avancer:
# - les cores
# - les input
# - les outputs = taille des core queues


import random

n_events = 1000
n_cores = 18

event_sched = {i: False for i in range(n_events)}

def new_core():
    return {
        'queue': [],
        'queue_len': 0
    }

def new_event(type=None, service_time=None, arrival_instant=None):
    return {
        'type': type,
        'service_time': service_time,
        'arrival_instant': arrival_instant,
    }

def generate_trace(duration=10**6, inter_arrival=.9, service_time=100):
    # En gros pour chaque timestamp, first check proba de happening.
    # second proba type of event
    # third event service time: constant

    # inter_arrival_times = [random.uniform(1,2) for _ in range(duration)]

    def generate_event(timestamp):
        new = True if random.random() < inter_arrival else False
        if new:
            return new_event(
                random.randint(0, n_events-1),
                service_time=service_time,
                arrival_instant=timestamp
            )
        else:
            return None
    return [generate_event(i) for i in range(duration)]

def enqueue_to_core(core, enqueue_event=None):
    if not event_sched[enqueue_event['type']]:
        cores[core]['queue'].append(enqueue_event)
        cores[core]['queue_len'] += 1
        event_sched[enqueue_event['type']] = True

def update_core(core):
    if core['queue_len'] > 0:
        core['queue'][0]['service_time'] -= 1
        if core['queue'][0]['service_time'] <= 0:
            core['queue_len'] -= 1
            event_sched[core['queue'][0]['type']] = False
            core['queue'].pop(0)

def format_core_lens(cores, timestamp):
    core_lens = [str(c['queue_len']) for c in cores]
    return ','.join(core_lens)



traces = generate_trace(duration=3*10**6, inter_arrival=.9, service_time=10)
n_packets = sum(1 for x in traces if x is not None)
cores = [new_core() for _ in range(n_cores)]
output_freq = 5
loaded_core = 1

def enqueue_to_core(core, enqueue_event=None, loaded_service_time=150):
    if not event_sched[enqueue_event['type']]:
        if core == loaded_core:
            enqueue_event['service_time'] = loaded_service_time
        cores[core]['queue'].append(enqueue_event)
        cores[core]['queue_len'] += 1
        event_sched[enqueue_event['type']] = True

def update_core(core):
    if core['queue_len'] > 0:
        core['queue'][0]['service_time'] -= 1
        if core['queue'][0]['service_time'] <= 0:
            core['queue_len'] -= 1
            event_sched[core['queue'][0]['type']] = False
            core['queue'].pop(0)

for timestamp, event in enumerate(traces):
    if event is not None:
        pick_core = random.randint(0, n_cores-1)
        enqueue_to_core(pick_core, enqueue_event=event, loaded_service_time=100)
    
    for core in cores:
        update_core(core)

    if timestamp % output_freq == 0:
        print(format_core_lens(cores, timestamp))
