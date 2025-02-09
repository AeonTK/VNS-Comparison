"""
This is a test for a window model to see how it will affect our last route and imbalance.
This test will show the results in the following format:
```
------------------------------------------------------------
      Dist  Window=0  Window=1  Window=2  Window=3  Window=4
      Mean 226011.48 199532.96 167573.51 146517.96 130082.58
       Std  21428.47  32561.90  20143.69  22451.09  27301.54
------------------------------------------------------------
 Total Imb  Window=0  Window=1  Window=2  Window=3  Window=4
      Mean      0.00     41.80     75.00     98.60    119.40
       Std      0.00      4.85      8.59     13.97     15.13
------------------------------------------------------------
   Max Imb  Window=0  Window=1  Window=2  Window=3  Window=4
      Mean      0.00      1.00      2.00      3.00      4.00
       Std      0.00      0.00      0.00      0.00      0.00
------------------------------------------------------------
Efficiency  Window=0  Window=1  Window=2  Window=3  Window=4
      Mean      1.05      0.99      0.97      0.94      0.89
       Std      0.10      0.11      0.13      0.10      0.12
------------------------------------------------------------
```
where
    each column corresponds to window size,
    `Dist` is a total distance of all vehicles calculated by `calculate_costs`, 
    `Total Imb` is the total number of supply/demand (imbalance),
    `Max Imb` is the maximum number of supply/demand (imbalance) of one node (which should usually coincide with the number of window),
    `Efficiency` is calculated by (the total number of allocation)/(total distance) * 1000 
        (multiply by 1000 is just for making them more than 1 to show decimal 2)

You can change `kwargs` values used in the tests. The important one's are
    nodes:              The number of nodes (stations) in the problem.
    number_of_vehicles: The number of vehicles available for routing.
    vehicle_capacity:   The maximum capacity of each vehicle.
    ordered_nbhs:       A list of ordered neighborhoods (operator functions) used for local search.
    local_timeout:      The timeout value for the local search in seconds.
    change_local_nbh:   A function to change the local neighborhood during the search.
    read_only:          Specifies if the test is read-only (read results from CSV).
    filename:           The name of the CSV file for storing test results.
    num_try:            The number of times to perform the test.
    max_window:         The max window size you want to try
    window_model:       all window model or partial window model. Check original function to see the difference.
More kwargs, check the parent class.
"""

import os
import time
import csv
from tqdm import tqdm
from copy import deepcopy
from itertools import product
import sys
sys.path.append(os.getcwd())

import solvers
from structure import Vehicle, ProblemInstance
import operators as ops
from loaders import load_subset_from_ordered_nodes, load_graph
import utils

from tests.test_base import TestLNS
from utils import (
    update_problem_with_all_window,
    update_problem_with_partial_window, 
    get_graph_after_rebalance, 
    get_total_imbalance_from_aux_graph,
    get_max_imbalance_from_aux_graph,
    assert_total_imbalance,
)

DATASETS = ['munich', 'nyc_dummy', 'nyc']
BALANCE_FIX = ['inside', 'outside']

KWARGS = {
    'nodes': 500,
    'dataset': 'munich',  # should be among ['munich', 'nyc_dummy', 'nyc'] = DATASETS
    'centeredness': 5,
    'number_of_vehicles': 5,
    'vehicle_capacity': 15,
    'ordered_nbhs': [ops.intra_two_opt,
                     ops.intra_segment_swap,
                     ops.inter_two_opt,
                     ops.inter_segment_swap],
    'distance_limit': 200000,  # metre
    'local_timeout': 2*60,  # second
    'local_verbose': 0,
    'change_local_nbh': solvers.change_nbh_cyclic,
    'root': os.path.join(os.getcwd(), 'results'),
    'read_only': False,
    'filename': 'test_window_model_for_map_check_ff.csv',
    'num_try': 20,
    'max_window': 4,
    'window_model': update_problem_with_all_window,
    'balance_fix': 'outside',  # should be in ['inside', outside] = BALANCE_FIX
}

class TestWindowModel(TestLNS):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_window = kwargs['max_window']
        self.window_model = kwargs['window_model']
        self.dataset = kwargs['dataset']
        self.balance_fix = kwargs['balance_fix']
        assert self.dataset in DATASETS
        assert self.balance_fix in BALANCE_FIX

    def get_original_problem(self):
        if self.dataset == 'nyc_dummy':
            original_graph, node_info, _depot = load_graph('nyc_instance_dummy', location='nyc_dummy')
        elif self.dataset == 'nyc':
            original_graph, node_info, _depot = load_graph('nyc_instance', location='nyc')
        elif self.dataset == 'munich':
            original_graph, node_info = load_subset_from_ordered_nodes(nodes=self.nodes, cost='time', centeredness=self.centeredness, randomness=self.randomness)
            _depot = '0'
        else:
            raise Exception("Unexpected dataset name. It should be among 'munich', 'nyc_dummy', or 'nyc'")
        return original_graph, node_info, _depot

    def get_problem_instance(self, graph, node_info, depot='0'):
        vehicles = [Vehicle(capacity=self.vehicle_capacity, vehicle_id=str(i), distance_limit=self.distance_limit)
                    for i in range(self.number_of_vehicles)]
        problem = ProblemInstance(input_graph=graph, vehicles=vehicles, node_data=node_info, verbose=0, depot=depot)
        return problem

    def solver(self, original_problem, window_problem):
        solvers.greedy_routing(window_problem, randomness=False)
        self.run_vns(window_problem)
        original_problem.vehicles = window_problem.vehicles
        without_dist = original_problem.calculate_costs(use_ff_ratio=False)/3600
        with_dist = original_problem.calculate_costs(use_ff_ratio=True)/3600
        total_bikes = get_total_imbalance_from_aux_graph(original_problem.model, original_problem.depot)
        aux_graph = get_graph_after_rebalance(original_problem)
        imbalance = get_total_imbalance_from_aux_graph(aux_graph, original_problem.depot)
        max_imbalance = get_max_imbalance_from_aux_graph(aux_graph, original_problem.depot)
        efficiency = (total_bikes - imbalance) / without_dist
        return without_dist, with_dist, imbalance, max_imbalance, efficiency

    def write_results_to_csv(self, filename, header, num_try):
        writer = csv.writer(open(os.path.join(self.root, filename), "w", newline=''))
        writer.writerow(header)
        results = []
        for _ in tqdm(range(num_try)):
            result = []
            original_graph, node_info, _depot = self.get_original_problem()
            original_problem = self.get_problem_instance(original_graph, node_info, depot=_depot)
            assert assert_total_imbalance(original_problem) == 0
            st = time.time()
            find, window_graph = self.window_model(deepcopy(original_problem), 0, 'outside')
            window_problem = self.get_problem_instance(window_graph, node_info, depot=_depot)
            if assert_total_imbalance(window_problem) != 0:
                print(f"assertoion error. delta {delta}. total {assert_total_imbalance(window_problem)}.")
            without_dist, with_dist, imbalance, max_imbalance, efficiency = self.solver(original_problem, window_problem)
            et = time.time()
            # utils.visualize_routes_go(original_problem.get_all_routes(), node_info)
            result += [without_dist, with_dist, 2*original_problem.imbalance, imbalance, max_imbalance, efficiency, et-st]
            for delta in range(1, self.max_window):
                for balance_fix in BALANCE_FIX:
                    st = time.time()
                    find, window_graph = self.window_model(deepcopy(original_problem), delta, balance_fix)
                    window_problem = self.get_problem_instance(window_graph, node_info, depot=_depot)
                    without_dist, with_dist, imbalance, max_imbalance, efficiency = self.solver(original_problem, window_problem)
                    et = time.time()
                    # utils.visualize_routes_go(original_problem.get_all_routes(), node_info)
                    result += [without_dist, with_dist, 2*original_problem.imbalance, imbalance, max_imbalance, efficiency, et-st]
            if not find:
                continue
            writer.writerow(result)
            results.append(result)
        return results


def main():
    for dataset in ['munich']:
        KWARGS['dataset'] = dataset
        sections = ['Deliv w/o [h]', 'Deliv w/ [h]', 'Origi Imb', 'Total Imb', 'Max Imb', 'Efficiency', 'Time [s]']
        N = len(sections)
        test_instance = TestWindowModel(**KWARGS)
        if KWARGS.get('read_only'):
            header, results = test_instance.read_results_from_csv(KWARGS.get('filename', 'test_window_model.csv'))
        else:
            header = ['W=0'] + [f'W={delta},{balance_fix}'
                                     for delta in range(1, KWARGS['max_window'])
                                     for balance_fix in ['in', 'out']]
            results = test_instance.write_results_to_csv(
                filename=KWARGS.get('filename', 'test_window_model.csv'),
                header=header,
                num_try=KWARGS.get('num_try', 100),
            )
        mean, std, max, min = test_instance.get_stats(results)

        print(''.join(['{:>15}'.format('----------')] + ['{:>10}'.format('----------') for i, x in enumerate(std) if i%N == 0]))
        for k, section in enumerate(sections):
            print(''.join(['{:>15}'.format(section)] + ['{:>10}'.format(x) for x in header]))
            print(''.join(['{:>15}'.format('Mean')] + ['{:>10.2f}'.format(float(x)) for i, x in enumerate(mean) if i%N == k]))
            print(''.join(['{:>15}'.format('Std')] + ['{:>10.2f}'.format(float(x)) for i, x in enumerate(std) if i%N == k]))
            print(''.join(['{:>15}'.format('Max')] + ['{:>10.2f}'.format(float(x)) for i, x in enumerate(max) if i%N == k]))
            print(''.join(['{:>15}'.format('Min')] + ['{:>10.2f}'.format(float(x)) for i, x in enumerate(min) if i%N == k]))
            print(''.join(['{:>15}'.format('----------')] + ['{:>10}'.format('----------') for i, x in enumerate(std) if i%N == k]))

if __name__ == '__main__':
    main()