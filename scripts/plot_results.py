#!/usr/bin/env python3
"""
Script for plotting annealing results.
"""

import argparse
from pathlib import Path
from cyclicycles.plotter import Plotter
import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(src_dir))

from cyclicycles.config import RESULT_DIR

def main():
    parser = argparse.ArgumentParser(description='Plot annealing results')
    parser.add_argument('--solver', type=str, default='6.4',
                       choices=['1.6', '4.1', '6.4'],
                       help='D-Wave solver results to plot')
    parser.add_argument('--n_nodes', type=int, default=None,
                       help='Specific instance to plot. If not specified, plots all instances.')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save plots. If not specified, displays plots.')
    
    args = parser.parse_args()
    
    # Initialize plotter
    plotter = Plotter(RESULT_DIR)
    
    # Create save directory if specified
    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot results
    if args.n_nodes is not None:
        plotter.plot_instance(args.solver, args.n_nodes, save_dir)
    else:
        plotter.plot_all_instances(args.solver, save_dir)

if __name__ == '__main__':
    main()