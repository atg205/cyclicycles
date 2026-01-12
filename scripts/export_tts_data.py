#!/usr/bin/env python3
"""
Export TTS data as JSON from calculate_tts results.

This script calculates TTS metrics and exports the x,y data points (num_qubits, tts)
as a nicely formatted JSON file.

Usage:
    python scripts/export_tts_data.py --solver 6
    python scripts/export_tts_data.py --solver 4.1 --output ./data_export/
    python scripts/export_tts_data.py --solver 1 --ancilla --output ./results/
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys

# Add the src directory to Python path
src_dir = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(src_dir))

from cyclicycles.config import RESULT_DIR
from cyclicycles.instance import Instance
from cyclicycles.plotter import Plotter


def calculate_tts(p_success: float, p_target: float = 0.99, runtime_ms: float = 1.0):
    """Calculate Time To Solution.
    
    Args:
        p_success: Probability of success (0 to 1)
        p_target: Target success probability (default 0.99)
        runtime_ms: Runtime in milliseconds
        
    Returns:
        TTS in milliseconds, or None if p_success is 0 or 1
    """
    if p_success <= 0 or p_success >= 1:
        return None
    
    if p_success == 1.0:
        return None
    
    tts = np.log(1 - p_target) / np.log(1 - p_success) * runtime_ms
    return tts


def load_and_analyze_results(solver_versions, instance_id, num_timepoints, use_ancilla=False, filter_cycles=None):
    """Load and analyze forward and cyclic annealing results.
    
    Args:
        solver_versions: List of solver versions
        instance_id: Instance ID
        num_timepoints: Number of timepoints
        use_ancilla: Whether to use ancilla
        filter_cycles: If specified, only include cyclic annealing samples with this many cycles
    
    Returns:
        tuple: (forward_analysis, cyclic_analysis, num_qubits)
    """
    plotter = Plotter(RESULT_DIR)
    instance = Instance(solver=solver_versions[0])
    
    # Get number of qubits
    dynamics_instances = instance.load_dynamics_instances(number_time_points=num_timepoints)
    if instance_id not in dynamics_instances:
        return None, None, None
    
    num_qubits = dynamics_instances[instance_id]['num_variables']
    
    # Load results to get offset
    cyclic_df, forward_stats, offset, cyclic_best_percentage = plotter.load_results(
        solver_id=solver_versions[0],
        instance_type='dynamics',
        instance_id=instance_id,
        num_timepoints=num_timepoints,
        use_ancilla=use_ancilla
    )
    
    gnd_energy_spin = 0.0
    
    # Forward annealing analysis
    forward_analysis = None
    total_time_ms_fw = 0.0
    total_samples_fw = 0
    successful_samples_fw = 0
    num_calls_fw = 0
    
    forward_tts_values = []
    
    for solver_version in solver_versions:
        solver_result_dir = RESULT_DIR / solver_version / 'forward' / f'instance_{instance_id}'
        if solver_result_dir.exists():
            for file_path in solver_result_dir.glob('*.csv'):
                num_files_fw = 1
                try:
                    df = __import__('pandas').read_csv(file_path)
                    
                    total_time_ms_fw += df['runtime_ms'].sum()
                    total_samples_fw += len(df)
                    
                    # Count ground state samples: gap == 0 (where gap = energy + offset)
                    gap = df['energy'] + offset
                    successful_samples_fw += len(df[gap == gnd_energy_spin])
                    num_calls_fw += 1
                    
                except Exception as e:
                    continue
    
    if num_calls_fw > 0 and total_samples_fw > 0:
        p_success_fw = successful_samples_fw / total_samples_fw
        tts_fw = calculate_tts(p_success_fw, runtime_ms=total_time_ms_fw)
        
        if tts_fw is not None:
            forward_tts_values.append(tts_fw)
        
        forward_analysis = {
            'p_success': p_success_fw,
            'tts': np.mean(forward_tts_values) if forward_tts_values else None
        }
    
    # Cyclic annealing analysis
    cyclic_analysis = None
    cyclic_tts_values = []
    
    for solver_version in solver_versions:
        cyclic_result_dir = RESULT_DIR / solver_version / 'dynamics_velox' / f'instance_{instance_id}'
        if cyclic_result_dir.exists():
            for cycle_dir in cyclic_result_dir.glob(f'*_timepoints_{num_timepoints}'):
                try:
                    # Extract number of cycles from directory name (e.g., "5_cycles_timepoints_5")
                    cycle_dir_name = cycle_dir.name
                    num_cycles = None
                    if '_cycles_' in cycle_dir_name:
                        num_cycles = int(cycle_dir_name.split('_')[0])
                    
                    # Filter by cycles if specified
                    if filter_cycles is not None and num_cycles != filter_cycles:
                        continue
                    
                    df = __import__('pandas').read_csv(cycle_dir / 'results.csv')
                    
                    total_time_ms_ca = df['runtime_ms'].sum()
                    total_samples_ca = len(df)
                    
                    gap = df['energy'] + offset
                    successful_samples_ca = len(df[gap == gnd_energy_spin])
                    
                    if total_samples_ca > 0:
                        p_success_ca = successful_samples_ca / total_samples_ca
                        tts_ca = calculate_tts(p_success_ca, runtime_ms=total_time_ms_ca)
                        
                        if tts_ca is not None:
                            cyclic_tts_values.append(tts_ca)
                
                except Exception as e:
                    continue
    
    if cyclic_tts_values:
        avg_tts_ca = np.mean(cyclic_tts_values)
        cyclic_analysis = {
            'p_success': None,
            'tts': avg_tts_ca,
            'num_realizations': len(cyclic_tts_values)
        }
    
    return forward_analysis, cyclic_analysis, num_qubits


def main():
    parser = argparse.ArgumentParser(description='Export TTS data as JSON')
    parser.add_argument('--solver', type=str, default="6",
                       help='D-Wave solver to analyze')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for JSON file')
    parser.add_argument('--ancilla', action='store_true',
                       help='Export results with ancilla transformation')
    parser.add_argument('--cycles', type=int, default=None,
                       help='Filter to only include samples with this many cycles')
    
    args = parser.parse_args()
    
    # Get solver versions
    solver_versions = []
    for solver_dir in RESULT_DIR.iterdir():
        if solver_dir.is_dir() and not solver_dir.name.startswith('.'):
            if solver_dir.name.split('.')[0] == args.solver or solver_dir.name == args.solver:
                solver_versions.append(solver_dir.name)
    
    if not solver_versions:
        print(f"Error: No results found for solver {args.solver}")
        return
    
    solver_versions.sort()
    print(f"Found solver versions: {solver_versions}")
    
    # Collect data
    forward_data = []
    cyclic_data = []
    
    # Get timepoints
    timepoints_set = set()
    for solver_version in solver_versions:
        solver_dir = RESULT_DIR / solver_version / 'dynamics_velox'
        if solver_dir.exists():
            for instance_dir in solver_dir.iterdir():
                if instance_dir.is_dir():
                    for cycle_dir in instance_dir.iterdir():
                        if cycle_dir.is_dir() and '_timepoints_' in cycle_dir.name:
                            parts = cycle_dir.name.split('_')
                            if len(parts) > 1:
                                try:
                                    timepoints_str = parts[1].split('_')[0]
                                    if timepoints_str.isdigit():
                                        timepoints_set.add(int(timepoints_str))
                                except:
                                    pass
    
    timepoints_list = sorted(list(timepoints_set))
    
    if not timepoints_list:
        print("No dynamics instances found")
        return
    
    # Process each timepoint and instance
    for num_timepoints in timepoints_list:
        instance = Instance(solver=solver_versions[0])
        dynamics_instances = instance.load_dynamics_instances(number_time_points=num_timepoints)
        
        if not dynamics_instances:
            continue
        
        for instance_id in sorted(dynamics_instances.keys()):
            forward_analysis, cyclic_analysis, num_qubits = load_and_analyze_results(
                solver_versions, instance_id, num_timepoints, args.ancilla, filter_cycles=args.cycles
            )
            
            if num_qubits is None:
                continue
            
            if forward_analysis is not None and forward_analysis['tts'] is not None:
                forward_data.append({
                    'num_qubits': num_qubits,
                    'tts': float(forward_analysis['tts']),
                    'instance_id': instance_id,
                    'num_timepoints': num_timepoints
                })
            
            if cyclic_analysis is not None and cyclic_analysis['tts'] is not None:
                cyclic_data.append({
                    'num_qubits': num_qubits,
                    'tts': float(cyclic_analysis['tts']),
                    'instance_id': instance_id,
                    'num_timepoints': num_timepoints
                })
    
    # Average by num_qubits
    def average_by_qubits(data):
        from collections import defaultdict
        grouped = defaultdict(list)
        for point in data:
            grouped[point['num_qubits']].append(point['tts'])
        
        result = []
        for num_qubits in sorted(grouped.keys()):
            avg_tts = np.mean(grouped[num_qubits])
            result.append({
                'x': num_qubits,
                'y': float(avg_tts)
            })
        return result
    
    forward_avg = average_by_qubits(forward_data) if forward_data else []
    cyclic_avg = average_by_qubits(cyclic_data) if cyclic_data else []
    
    # Create output structure
    output_data = {
        'metadata': {
            'solver': args.solver,
            'solver_versions': solver_versions,
            'with_ancilla': args.ancilla,
            'cycles_filter': args.cycles,
            'description': 'TTS data points: x = number of qubits, y = Time To Solution (milliseconds)'
        },
        'forward_annealing': {
            'data': forward_avg,
            'num_points': len(forward_avg)
        },
        'cyclic_annealing': {
            'data': cyclic_avg,
            'num_points': len(cyclic_avg)
        }
    }
    
    # Determine output path
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path('.')
    
    # Create filename
    ancilla_str = "_with_ancilla" if args.ancilla else ""
    cycles_str = f"_{args.cycles}cycles" if args.cycles else ""
    filename = f"tts_data_{args.solver}{ancilla_str}{cycles_str}.json"
    output_path = output_dir / filename
    
    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Data exported to: {output_path}")
    print(f"  Forward annealing points: {len(forward_avg)}")
    print(f"  Cyclic annealing points: {len(cyclic_avg)}")
    print("\nJSON structure:")
    print(json.dumps(output_data, indent=2)[:500] + "...")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
