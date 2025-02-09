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



# Load a subset of the data from MVG dataset and create the Vehicles and Problem Instance objects

graph, node_info = load_graph_from_csv(stations_csv=os.path.join(os.path.dirname(__file__), '../CSV_DATA/stations.csv'), edges_csv=os.path.join(os.path.dirname(__file__), '../CSV_DATA/edges.csv'))
# graph, node_info = load_subset_from_ordered_nodes(nodes=kwargs['nodes'], cost='dist', randomness=False)

vehicles = [Vehicle(capacity=kwargs['vehicle_capacity'], vehicle_id=str(i),
                    depot='0') for i in range(kwargs['num_vehicles'])]
# vehicles = [Vehicle(capacity=kwargs['vehicle_capacity'], vehicle_id=str(i),
#                     depot=str(random.randint(0, kwargs['nodes'] - 1))) for i in range(kwargs['num_vehicles'])]
problem = ProblemInstance(input_graph=graph, vehicles=vehicles, node_data=node_info, verbose=1)


# Create and initial set of solutions using the greedy algorithm and calculate the loading instrunctions

solvers.greedy_routing(problem, randomness=True)
problem.calculate_loading_mf()
print("\nInitial Solution using greedy alogrithm:")
problem.display_results(show_instructions=True)


# Run the VNS and time it
operator_seq = [ops.inter_segment_swap, ops.intra_segment_swap, ops.inter_two_opt, ops.intra_two_opt]
# operator_seq = [ops.inter_segment_swap, ops.intra_segment_swap, ops.intra_two_opt]
start_time = time.time()

distance_hist, time_hist, operation_hist = solvers.general_variable_nbh_search(
    problem, operator_seq, change_nbh=solvers.change_nbh_cyclic, timeout=kwargs['vns_timeout'], verbose=1)
problem.calculate_loading_mf()

end_time = time.time()


# Show the final Results
time_run = round(end_time - start_time, 3)
time_out = 'Converged before timeout' if time_run < kwargs['vns_timeout'] else 'Stopped due to timeout'
print("\nSolution after applying VNS for {} seconds ({}):".format(time_run, time_out))
problem.display_results(show_instructions=True)


# Plot basic routes
# utils.visualize_routes(problem.get_all_routes(), node_info)

# Plot routes in browser
# utils.visualize_routes_go(problem.get_all_routes(), node_info)

# Plot VNS improvement vs time graph
# utils.show_improvement_graph(distance_hist, time_hist, operation_hist, operator_seq, change_nbh_name='Pipe')

