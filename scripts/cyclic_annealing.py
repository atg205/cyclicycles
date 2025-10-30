#!/usr/bin/env python3
"""
Script for running cyclic quantum annealing on problem instances.
"""

import argparse
from pathlib import Path
from cyclicycles.runner import Runner

def main():
    parser = argparse.ArgumentParser(description='Run cyclic quantum annealing')
    parser.add_argument('--n_nodes', type=str, default=None,
                       help='Number of nodes in the instance to run. If not specified, uses first available.')
    parser.add_argument('--num_reads', type=int, default=1000,
                       help='Number of annealing reads per cycle')
    parser.add_argument('--num_cycles', type=int, default=5,
                       help='Number of annealing cycles to perform')
    parser.add_argument('--sampler', type=str, default='6.4',
                       choices=['1.6', '4.1', '6.4'],
                       help='D-Wave sampler to use (1.6=Advantage2, 4.1/6.4=Advantage)')
    
    args = parser.parse_args()
    
    # Initialize runner with specified sampler
    runner = Runner(sampler=args.sampler)
    
    print(f"Running cyclic annealing with parameters:")
    print(f"Number of nodes: {args.n_nodes if args.n_nodes else 'First available'}")
    print(f"Number of reads per cycle: {args.num_reads}")
    print(f"Number of cycles: {args.num_cycles}")
    print(f"Sampler: {args.sampler}")
    print("\nStarting annealing process...")
    
    # Run cyclic annealing
    response, results, cycle_energies = runner.execute_cyclic_annealing(
        n_nodes=args.n_nodes,
        num_reads=args.num_reads,
        num_cycles=args.num_cycles
    )
    
    print("\nCyclic annealing completed!")
    print("\nEnergy progression across cycles:")
    for cycle, energy in enumerate(cycle_energies, 1):
        print(f"Cycle {cycle}: {energy:.6f}")
    
    print(f"\nBest energy found: {results['best_energy']:.6f}")
    print(f"Results saved to: {results['save_path'] if 'save_path' in results else 'results directory'}")

if __name__ == '__main__':
    main()