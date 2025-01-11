"""
Use this file to run a standard VNS
"""


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import random

from loaders import load_subset_from_ordered_nodes, load_graph_from_csv
from structure import ProblemInstance, Vehicle
import utils
import solvers
import operators as ops
import time
import numpy as np



kwargs = {
    'nodes':                50,
    'num_vehicles':         5,
    'vehicle_capacity':     14,
    'vns_timeout':          90,  # seconds

    'stop_duration':     0,
    'load_duration':     0,


    'ordered_nbhs': [ops.intra_two_opt, ops.intra_segment_swap, ops.inter_two_opt, ops.inter_segment_swap, ops.multi_remove_and_insert_station],
    'nbh_change_set': [solvers.change_nbh_cyclic, solvers.change_nbh_pipe, solvers.change_nbh_sequential, solvers.change_nbh_check_all],
    'large_nbhs': [0.1, 0.15, 0.20, 0.30],
    'large_timeout': 200,
}

run_data = []

time_start = time.time()

TIME_LIMIT = 1800

N_STATIONS = 50
# N_STATIONS = 75

i = 0

while time.time() - time_start < TIME_LIMIT:
    i += 1
    run_info = {}
    run_info['seed'] = i

    random.seed(i)

    graph, node_info = load_graph_from_csv(stations_csv=os.path.join(os.path.dirname(__file__), f'../CSV_DATA/stations_{N_STATIONS}.csv'), edges_csv=os.path.join(os.path.dirname(__file__), f'../CSV_DATA/edges_{N_STATIONS}.csv'))
    start_time = time.time()

    vehicles = [Vehicle(capacity=kwargs['vehicle_capacity'], vehicle_id=str(i),
                        depot='0') for i in range(kwargs['num_vehicles'])]
    problem = ProblemInstance(input_graph=graph, vehicles=vehicles, node_data=node_info, verbose=0)


    solvers.greedy_routing(problem, randomness=True)
    problem.calculate_loading_mf()

    run_info['init_dist'] = round(problem.calculate_costs(), 3)

    operator_seq = [ops.inter_segment_swap, ops.intra_segment_swap, ops.inter_two_opt, ops.intra_two_opt]
    
    distance_hist, time_hist, operation_hist = solvers.general_variable_nbh_search(
        problem, operator_seq, change_nbh=solvers.change_nbh_cyclic, timeout=kwargs['vns_timeout'], verbose=0)
    problem.calculate_loading_mf()

    end_time = time.time()

    run_info['runtime'] = end_time - start_time
    run_info['res_dist'] = round(problem.calculate_costs(), 3)

    time_run = round(end_time - start_time, 3)
    time_out = 'Converged before timeout' if time_run < kwargs['vns_timeout'] else 'Stopped due to timeout'

    run_data.append(run_info)

    print(f"Run {i} finished at {time.time() - time_start:.2f} seconds.")


# Сompute the mean runtime and distance
mean_runtime = sum([run['runtime'] for run in run_data]) / len(run_data)
mean_init_dist = sum([run['init_dist'] for run in run_data]) / len(run_data)
mean_res_dist = sum([run['res_dist'] for run in run_data]) / len(run_data)

# Сompute the standard error of the mean
sem_runtime = np.std([run['runtime'] for run in run_data]) / np.sqrt(len(run_data))
sem_init_dist = np.std([run['init_dist'] for run in run_data]) / np.sqrt(len(run_data))
sem_res_dist = np.std([run['res_dist'] for run in run_data]) / np.sqrt(len(run_data))

# Сompute the minimum resulting distance
min_res_dist = min([run['res_dist'] for run in run_data])
n_runs = len(run_data)



print(f"Number of runs: {n_runs}")
print(f"Runtime: {mean_runtime} ± {sem_runtime}")
print(f"Initial Distance: {mean_init_dist} ± {sem_init_dist}")
print(f"Resulting Distance: {mean_res_dist} ± {sem_res_dist}")
print(f"Min Resulting Distance: {min_res_dist}")

