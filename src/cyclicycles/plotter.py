from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from .config import INSTANCE_DIR

class Plotter:
    def __init__(self, result_dir: Path | str):
        """Initialize plotter with result directory.
        
        Args:
            result_dir: Path to the results directory containing solver subdirectories
        """
        self.result_dir = Path(result_dir)
        
        # Load ground state energies
        all_data_path = INSTANCE_DIR / 'all_data.npz'
        try:
            data = np.load(all_data_path, allow_pickle=True)
            self.ground_energies = {
                str(N): energy for N, energy in 
                zip(data['N_list'], data['ground_energy_list'])
            }
        except Exception as e:
            print(f"Warning: Could not load ground state energies: {e}")
            self.ground_energies = {}
        
    def load_results(self, solver_id: str, n_nodes: int) -> tuple[pd.DataFrame, tuple[float, float, int]]:
        """Load all results for a specific node count and solver.
        
        Args:
            solver_id: The solver ID (e.g., '4.1', '6.4')
            n_nodes: Number of nodes in the instance
            
        Returns:
            tuple: (DataFrame with cyclic results, (forward_mean, forward_std, forward_count))
        """
        # Load cyclic annealing results
        cyclic_path = self.result_dir / solver_id / f'N_{n_nodes}_realization_1'
        cyclic_data = []
        
        if cyclic_path.exists():
            for result_file in cyclic_path.glob('*.npz'):
                data = np.load(result_file, allow_pickle=True)
                cycle_energies = data['cycle_energies']
                cyclic_data.append(cycle_energies)
        
        # Convert to DataFrame
        max_cycles = max(len(e) for e in cyclic_data) if cyclic_data else 0
        if max_cycles > 0:
            # Pad shorter sequences with NaN
            padded_data = [np.pad(e, (0, max_cycles - len(e)), 
                                constant_values=np.nan) for e in cyclic_data]
            df = pd.DataFrame(padded_data).T  # Transpose to get cycles as rows
        else:
            df = pd.DataFrame()
            
        # Load forward annealing results
        forward_path = self.result_dir / 'forward' / solver_id / f'N_{n_nodes}_realization_1'
        forward_stats = (float('inf'), 0.0, 0)  # mean, std, count
        
        if forward_path.exists():
            energies = []
            for result_file in forward_path.glob('*.npz'):
                data = np.load(result_file)
                energies.append(min(data['energies']))
            if energies:
                forward_stats = (
                    float(np.mean(energies)),  # mean
                    float(np.std(energies)),   # std
                    len(energies)              # count
                )
                
        return df, forward_stats
    
    def plot_instance(self, solver_id: str, n_nodes: int, save_dir: Path | str | None = None):
        """Create plot for a specific instance.
        
        Args:
            solver_id: The solver ID (e.g., '4.1', '6.4')
            n_nodes: Number of nodes in the instance
            save_dir: Optional directory to save the plot
        """
        df, forward_stats = self.load_results(solver_id, n_nodes)
        
        if df.empty:
            print(f"No data found for N={n_nodes}, solver {solver_id}")
            return
            
        # Calculate statistics
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        cycles = range(1, len(mean) + 1)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot cyclic annealing results
        n_cyclic = len(df.columns)  # number of cyclic annealing runs
        plt.plot(cycles, mean, 'b-', 
                label=f'Cyclic Annealing (n={n_cyclic})')
        plt.fill_between(cycles, mean - std, mean + std, alpha=0.2, color='b')
        
        # Plot forward annealing result with error bar
        forward_mean, forward_std, forward_count = forward_stats
        if forward_mean != float('inf'):
            plt.axhline(y=forward_mean, color='r', linestyle='--', 
                       label=f'Forward Annealing (n={forward_count})')
            if forward_count > 1:  # Only show error band if we have multiple runs
                plt.axhspan(forward_mean - forward_std, 
                          forward_mean + forward_std,
                          color='r', alpha=0.2)
            
        # Plot ground state energy if available
        str_n_nodes = str(n_nodes)
        if str_n_nodes in self.ground_energies:
            ground_energy = self.ground_energies[str_n_nodes]
            plt.axhline(y=ground_energy, color='g', linestyle=':', 
                       label='Known Ground State')
        
        plt.xlabel('Cycle')
        plt.ylabel('Energy')
        plt.title(f'Energy vs Cycle (N={n_nodes}, Solver {solver_id})')
        plt.legend()
        plt.grid(True)
        
        if save_dir:
            save_path = Path(save_dir) / f'energy_N{n_nodes}_solver{solver_id}.png'
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show(block=True)
            
    def plot_all_instances(self, solver_id: str, save_dir: Path | str | None = None):
        """Create plots for all available instances.
        
        Args:
            solver_id: The solver ID (e.g., '4.1', '6.4')
            save_dir: Optional directory to save the plots
        """
        # Find all instance directories
        instance_dirs = list(self.result_dir.glob(f"{solver_id}/N_*_realization_1"))
        node_counts = [int(d.name.split('_')[1]) for d in instance_dirs]
        
        for n_nodes in sorted(node_counts):
            self.plot_instance(solver_id, n_nodes, save_dir)