from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from .config import INSTANCE_DIR, PROJECT_ROOT

class Plotter:
    def __init__(self, result_dir: Path | str):
        """Initialize plotter with result directory.
        
        Args:
            result_dir: Path to the results directory containing solver subdirectories
        """
        self.result_dir = Path(result_dir)
        
        
    def load_results(self, solver_id: str, n_nodes: int | None = None, num_samples: int | None = None,
                   init_type: str = 'all', instance_type: str = 'static', 
                   instance_id: str | None = None, num_timepoints: int | None = None) -> tuple[pd.DataFrame, tuple[float, float, float, int]]:
        """Load all results for a specific instance and solver.
        
        Args:
            solver_id: The solver ID (e.g., '4.1', '6.4')
            n_nodes: Number of nodes in the instance (for static instances)
            num_samples: Only include results with this exact number of samples (100 or 1000)
            init_type: Which initialization type to load: 'forward', 'zero', or 'all'
            instance_type: Either 'static' or 'dynamics'
            instance_id: ID of the dynamics instance (required if instance_type='dynamics')
            num_timepoints: Number of timepoints for dynamics instances
            
        Returns:
            tuple: (DataFrame with cyclic results, (forward_mean, forward_std, forward_min, forward_count))
        """
        if instance_type == 'dynamics':
            if instance_id is None or num_timepoints is None:
                raise ValueError("instance_id and num_timepoints must be specified for dynamics instances")
            # Load cyclic annealing results
            cyclic_path = self.result_dir / solver_id / f'dynamics_{instance_id}_timepoints_{num_timepoints}_realization_1'
            forward_path = self.result_dir / 'forward' / solver_id / f'dynamics_{instance_id}_timepoints_{num_timepoints}'
        else:
            if n_nodes is None:
                raise ValueError("n_nodes must be specified for static instances")
            # Load cyclic annealing results
            cyclic_path = self.result_dir / solver_id / f'N_{n_nodes}_realization_1'
            forward_path = self.result_dir / 'forward' / solver_id / f'N_{n_nodes}_realization_1'
        
        cyclic_data = []
        
        if cyclic_path.exists():
            for result_file in cyclic_path.glob('*.npz'):
                data = np.load(result_file, allow_pickle=True)
                
                # Check initialization type
                if init_type != 'all':
                    used_forward = bool(data.get('used_forward_init', False))
                    if (init_type == 'forward' and not used_forward) or \
                       (init_type == 'zero' and used_forward):
                        continue
                
                # Check if this file has the required number of samples
                if num_samples is not None:
                    total_samples = sum(data['num_occurrences']) if 'num_occurrences' in data else None
                    if total_samples != num_samples:
                        continue
                        
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
        forward_stats = (float('inf'), 0.0, float('inf'), 0)  # mean, std, min, count
        
        if forward_path.exists():
            energies = []
            for result_file in forward_path.glob('*.npz'):
                data = np.load(result_file)
                # Check if this file has the required number of samples
                if num_samples is not None:
                    total_samples = sum(data['num_occurrences']) if 'num_occurrences' in data else None
                    if total_samples != num_samples:
                        continue
                energies.append(min(data['energies']))
            if energies:
                min_energy = float(min(energies))
                forward_stats = (
                    float(np.mean(energies)),  # mean
                    float(np.std(energies)),   # std
                    min_energy,                # min
                    len(energies)              # count
                )
                
        return df, forward_stats
    
    def load_ground_state_energy(self, instance_type: str = 'static', instance_id: str | None = None, 
                                num_timepoints: int | None = None, n_nodes: int | None = None) -> float | None:
        """Load ground state energy for an instance.
        
        Args:
            instance_type: Either 'static' or 'dynamics'
            instance_id: ID of the dynamics instance (required if instance_type='dynamics')
            num_timepoints: Number of timepoints for dynamics instances (required if instance_type='dynamics')
            n_nodes: Number of nodes for static instances (required if instance_type='static')
            
        Returns:
            Ground state energy or None if not found
        """
        if instance_type == 'dynamics':
            if instance_id is None or num_timepoints is None:
                return None
            
            # Load from CSV file
            csv_path = PROJECT_ROOT / 'data' / 'results' / 'dynamics_velox' / instance_id / 'velox' / f'best_results_hessian_{instance_id}_native.csv'
            
            if not csv_path.exists():
                print(f"Warning: Ground state file not found: {csv_path}")
                return None
            
            try:
                df = pd.read_csv(csv_path)
                
                # Find row matching the timepoints pattern
                # Look for rows where instance column matches *_timepoints_{num_timepoints}.coo
                matching_rows = df[df['instance'].str.contains(f'_timepoints_{num_timepoints}\\.coo', regex=True, na=False)]
                
                if matching_rows.empty:
                    print(f"Warning: No matching ground state entry for timepoints={num_timepoints}")
                    return None
                
                # Get the first matching row's gnd_energy
                gnd_energy = matching_rows.iloc[0]['gnd_energy']
                return float(gnd_energy)
            except Exception as e:
                print(f"Warning: Could not load ground state energy from {csv_path}: {e}")
                return None
        
        else:  # static instances
            if n_nodes is None:
                return None
            
            # Load from npz file
            all_data_path = INSTANCE_DIR / 'all_data.npz'
            try:
                data = np.load(all_data_path, allow_pickle=True)
                energies_dict = {
                    str(N): energy for N, energy in 
                    zip(data['N_list'], data['ground_energy_list'])
                }
                energy_val = energies_dict.get(str(n_nodes))
                return float(energy_val) if energy_val is not None else None
            except Exception as e:
                print(f"Warning: Could not load ground state energies: {e}")
                return None
    
    def plot_instance(self, solver_id: str, n_nodes: int | None = None, save_dir: Path | str | None = None,
                     num_samples: int | None = None, init_type: str = 'all', instance_type: str = 'static',
                     instance_id: str | None = None, num_timepoints: int | None = None):
        """Create plot for a specific instance.
        
        Args:
            solver_id: The solver ID (e.g., '4.1', '6.4')
            n_nodes: Number of nodes in the instance (for static instances)
            save_dir: Optional directory to save the plot
            num_samples: Only include results with this exact number of samples (100 or 1000)
            init_type: Which initialization type to show: 'forward', 'zero', or 'all'
            instance_type: Either 'static' or 'dynamics'
            instance_id: ID of the dynamics instance (required if instance_type='dynamics')
            num_timepoints: Number of timepoints for dynamics instances
        """
        df, forward_stats = self.load_results(solver_id, n_nodes=n_nodes, num_samples=num_samples, 
                                            init_type=init_type, instance_type=instance_type,
                                            instance_id=instance_id, num_timepoints=num_timepoints)

        # Load ground state energy
        ground_energy = self.load_ground_state_energy(
            instance_type=instance_type,
            instance_id=instance_id,
            num_timepoints=num_timepoints,
            n_nodes=n_nodes
        )
        
        if df.empty:
            print(f"No data found for N={n_nodes}, solver {solver_id}")
            return
            
        # Calculate statistics
        mean = df.mean(axis=1)
        min_per_cycle = df.min(axis=1)
        max_per_cycle = df.max(axis=1)
        cycles = range(1, len(mean) + 1)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot cyclic annealing results
        n_cyclic = len(df.columns)  # number of cyclic annealing runs
        plt.plot(cycles, mean, 'b-', 
                label=f'Cyclic Annealing Mean (n={n_cyclic})')
        plt.fill_between(cycles, min_per_cycle, max_per_cycle, alpha=0.2, color='b')
        
        # Plot lowest cyclic energy found
        min_cyclic = df.min().min()  # minimum across all cycles and runs
        plt.axhline(y=min_cyclic, color='b', linestyle=':', 
                   label='Best Cyclic Result')
        # Add annotation for cyclic minimum
        plt.annotate(f'E = {min_cyclic:.6f}',
                    xy=(len(cycles)-2, min_cyclic),
                    xytext=(0, 10), textcoords='offset points',
                    ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='b', alpha=0.8),
                    color='b')
        
        # Plot forward annealing result with error bar
        forward_mean, forward_std, forward_min, forward_count = forward_stats
        if forward_mean != float('inf'):
            plt.axhline(y=forward_mean, color='r', linestyle='--', 
                       label=f'Forward Annealing Mean (n={forward_count})')
            if forward_count > 1:  # Only show error band if we have multiple runs
                plt.axhspan(forward_min, forward_mean + forward_std,
                          color='r', alpha=0.2)
                # Plot lowest forward energy found
                plt.axhline(y=forward_min, color='r', linestyle=':', 
                          label='Best Forward Result')
                # Add annotation for forward minimum
                plt.annotate(f'E = {forward_min:.6f}',
                           xy=(3, forward_min),
                           xytext=(-10, -10), textcoords='offset points',
                           ha='right', va='top',
                           bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='r', alpha=0.8),
                           color='r')
            
        plt.xlim(1,len(cycles))
        # Plot ground state energy if available
        if ground_energy is not None:
            plt.axhline(y=ground_energy, color='g', linestyle=':', 
                       label='Known Ground State')
            
            # annotate
            plt.annotate(f'E = {ground_energy:.6f}',
                        xy=(len(cycles)/2, ground_energy),
                        xytext=(0, 10), textcoords='offset points',
                        ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='b', alpha=0.8),
                        color='gray')
        
        plt.xlabel('Cycle')
        plt.ylabel('Energy')
        samples_str = f", {num_samples} samples" if num_samples else ""
        init_str = f", {init_type} init" if init_type != 'all' else ""
        
        if instance_type == 'dynamics':
            title = f'Energy vs Cycle (dynamics_{instance_id}, timepoints={num_timepoints}, Solver {solver_id}{samples_str}{init_str})'
        else:
            title = f'Energy vs Cycle (N={n_nodes}, Solver {solver_id}{samples_str}{init_str})'
        
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        if save_dir:
            if instance_type == 'dynamics':
                save_path = Path(save_dir) / f'energy_dynamics_{instance_id}_timepoints_{num_timepoints}_solver{solver_id}.png'
            else:
                save_path = Path(save_dir) / f'energy_N{n_nodes}_solver{solver_id}.png'
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show(block=True)
