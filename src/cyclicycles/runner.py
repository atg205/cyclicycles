from pathlib import Path
import numpy as np
import json
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod
from .instance import Instance
from config import RESULT_DIR, INSTANCE_DIR, DATA_DIR, ensure_dir
import re
class Runner:
    def __init__(self, sampler='6.4'):
        self.time_path = DATA_DIR / 'time.json'
        ensure_dir(self.time_path.parent)
        if not self.time_path.exists():
            with self.time_path.open('w') as f:
                json.dump({"time_ms": 0}, f)
                
    
        self.sampler = sampler
        # Default annealing schedule and h_gain
        self.anneal_schedule = [
            [0.0,   1.0],   # start at s=1  (Bx ~ 0)
            [0.1,   1.0],   # turn on Bz while keeping s=1
            [0.6,   0.35],  # ramp down to s_min -> Bx high
            [300.0, 1.0],   # ramp back to s=1 (Bx -> 0)
        ]  # times in microseconds

        self.h_gain_schedule = [
            [0.0,   0.0],   # Bz=0 at t=0 per table
            [0.1,   1],  # raise Bz by 0.1 μs
            [0.6,   1],  # keep Bz ~ constant while Bx rises
            [300.0, 0],   # bring Bz back to 0 by the end
        ]

        # Configure D-Wave sampler based on solver ID
        if self.sampler == "1.7":  # zephyr
            self.qpu = DWaveSampler(solver="Advantage2_system1.7")
        elif self.sampler == "6.4":
            self.qpu = DWaveSampler(solver="Advantage_system6.4")
        elif self.sampler == "4.1":
            self.qpu = DWaveSampler(solver="Advantage_system4.1")
        else:
            raise ValueError(f"Invalid solver id: {self.sampler}")
        self.dw_sampler = self.qpu
        

    def _log_access_time(self, access_time_us: float):
        """Log the D-Wave access time to time.json.
        
        Args:
            access_time_ms (float): Access time in milliseconds.
        """
        try:
            with self.time_path.open('r') as f:
                time_dict = json.load(f)
            time_dict['time_ms'] += access_time_us * 1e-3
            with self.time_path.open('w') as f:
                json.dump(time_dict, f)
        except Exception as e:
            print(f"Error logging access time: {e}")

    def  execute_cyclic_annealing(self, n_nodes: str | None = None, num_cycles: int = 5, num_reads: int = 1000, 
                             use_forward_init: bool = False, instance_type: str = 'static', instance_id: str | None = None,
                             num_timepoints: int = 5):
        """Execute cyclic annealing on a problem instance.
        
        Args:
            n_nodes (str, optional): Number of nodes to select specific static instance.
                If None, executes first available instance.
            num_cycles (int): Number of cyclic annealing iterations. Defaults to 5.
            num_reads (int): Number of samples per cycle. Defaults to 1000.
            use_forward_init (bool): If True, run forward annealing first and use its best
                solution as initial state. Defaults to False.
            instance_type (str): Either 'static' or 'dynamics'. Defaults to 'static'.
            instance_id (str, optional): ID of the dynamics instance (required if instance_type='dynamics').
            num_timepoints (int): Number of timepoints for dynamics instances. Defaults to 5.
            
        Returns:
            tuple: (final_response, result_data, cycle_energies)
        """
    
        # Load instances
        instance = Instance(solver=self.sampler)
        
        if instance_type == 'dynamics':
            self.dw_sampler = EmbeddingComposite(self.dw_sampler)
            if instance_id is None:
                raise ValueError("instance_id must be specified for dynamics instances")
            
            dynamics_instances = instance.load_dynamics_instances(number_time_points=num_timepoints)
            if instance_id not in dynamics_instances:
                raise ValueError(f"Dynamics instance {instance_id} not found")
            
            dyn_instance = dynamics_instances[instance_id]
            # Create BINARY BQM first, then convert to SPIN for D-Wave
            bqm_binary = dimod.BQM(dyn_instance['h'], dyn_instance['J'], 
                                   dyn_instance['offset'], vartype='BINARY')
            bqm_spin = bqm_binary.copy()
            bqm_spin.change_vartype(dimod.SPIN)
            h = dict(bqm_spin.linear)
            J = dict(bqm_spin.quadratic)
            offset = bqm_spin.offset
            vartype = 'SPIN'
            
        else:  # static instances
            J_terms = instance.load_instances()
            
            if not J_terms:
                raise ValueError("No static instances found")
                
            # Select instance
            if n_nodes is None:
                n_nodes = list(J_terms.keys())[0]
            elif n_nodes not in J_terms:
                raise ValueError(f"Instance with {n_nodes} nodes not found")
                
            J = J_terms[n_nodes]
            h = {}  # No linear terms for static instances
            offset = 0.0
            vartype = 'SPIN'
        
        used_qubits = set([i for (i,j) in J.keys()] + [j for (i,j) in J.keys()])
        initial_state = {qubit: 0 if qubit in used_qubits else 3 for qubit in self.qpu.nodelist}

        # Initialize state for first cycle
        num_variables = len(used_qubits)
        cycle_energies = []
        best_state = initial_state
        best_energy = float('inf')
        
        # Run forward annealing first if requested
        if use_forward_init:
            print("Running forward annealing for initialization...")
            forward_response, _ = self.execute_instance(n_nodes=n_nodes, num_reads=num_reads, 
                                                       instance_type=instance_type, instance_id=instance_id,
                                                       num_timepoints=num_timepoints)
            
            # Get best solution from forward annealing
            min_energy_idx = np.argmin(forward_response.record.energy)
            forward_energy = forward_response.record.energy[min_energy_idx]
            forward_state = {qubit: int(forward_response.record.sample[min_energy_idx][i])
                           for i, qubit in enumerate(forward_response.variables)}
            
            print(f"Forward annealing found solution with energy: {forward_energy:.6f}")
            
            # Use forward solution as initial state
            best_state = forward_state
            best_energy = forward_energy
            cycle_energies.append(forward_energy)  # Count forward annealing as first cycle

        final_response = None
        for cycle in range(num_cycles):
            # Set up reverse annealing parameters
            reverse_params = dict(
                anneal_schedule=self.anneal_schedule,
                initial_state=best_state,
                reinitialize_state=True,
                h_gain_schedule=self.h_gain_schedule
            )
            
            # Execute reverse annealing
            response = self.dw_sampler.sample_ising(
                h=h,
                J=J,
                num_reads=num_reads,
                anneal_schedule=self.anneal_schedule,
                initial_state=best_state,
                reinitialize_state=True,
                h_gain_schedule=self.h_gain_schedule
            )
            final_response = response  # Keep track of last response
            
            # Log access time
            if 'timing' in response.info and 'qpu_access_time' in response.info['timing']:
                self._log_access_time(response.info['timing']['qpu_access_time'])
            
            # Find best solution from this cycle
            min_energy_idx = np.argmin(response.record.energy)
            cycle_min_energy = response.record.energy[min_energy_idx]
            cycle_energies.append(cycle_min_energy)
            
            print(f"Cycle {cycle + 1}/{num_cycles} - Minimum energy: {cycle_min_energy:.6f}")
            
            # Update best state if we found a better solution
            if cycle_min_energy < best_energy:
                best_energy = cycle_min_energy
                best_state = {qubit: int(response.record.sample[min_energy_idx][i])
                              for i,qubit in enumerate(response.variables)}
        
        # Save final results
        if instance_type == 'dynamics':
            results_dir = ensure_dir(RESULT_DIR / str(self.sampler) / f'dynamics_{instance_id}_timepoints_{num_timepoints}_realization_1')
        else:
            results_dir = ensure_dir(RESULT_DIR / str(self.sampler) / f'N_{n_nodes}_realization_1')
        
        # Find next available file number
        existing_files = list(results_dir.glob('[0-9]*.npz'))
        next_number = 1 if not existing_files else max(int(re.findall('[0-9]*',f.stem)[0]) for f in existing_files) + 1
        results_path = results_dir / f'{next_number}.npz'
        
        # Add a suffix if forward initialization was used
        if use_forward_init:
            results_path = results_dir / f'{next_number}_forward_init.npz'
        
        if final_response is None:
            raise RuntimeError("No annealing cycles were completed")
            
        # Get last response info for metadata
        final_response_info = {
            'energies': final_response.record.energy,
            'solutions': final_response.record.sample,
            'num_occurrences': final_response.record.num_occurrences,
            'timing': final_response.info['timing']
        }
        
        # Save results with cyclic annealing information
        result_data = {
            **final_response_info,
            'anneal_schedule': self.anneal_schedule,
            'h_gain_schedule': self.h_gain_schedule,
            'cycle_energies': np.array(cycle_energies),
            'num_cycles': num_cycles,
            'best_state': best_state,
            'best_energy': best_energy,
            'used_forward_init': use_forward_init,
            'offset': offset,
            'instance_type': instance_type,
            'num_timepoints': num_timepoints if instance_type == 'dynamics' else None
        }
        
        np.savez_compressed(results_path, **result_data)
        print(f"\nResults saved as: {results_path}")
        
        return final_response, result_data, cycle_energies
        
    def execute_instance(self, n_nodes: str | None = None, num_reads: int = 1000, instance_type: str = 'static', 
                        instance_id: str | None = None, num_timepoints: int = 5):
        """Execute a single annealing run on a problem instance.
        
        Args:
            n_nodes (str, optional): Number of nodes to select specific static instance.
                If None, executes first available instance.
            num_reads (int, optional): Number of samples to collect. Defaults to 1000.
            instance_type (str): Either 'static' or 'dynamics'. Defaults to 'static'.
            instance_id (str, optional): ID of the dynamics instance (required if instance_type='dynamics').
            num_timepoints (int): Number of timepoints for dynamics instances. Defaults to 5.
            
        Returns:
            tuple: (response object, instance_info)
        """
        # Load instances
        instance = Instance(solver=self.sampler)
        
        if instance_type == 'dynamics':
            self.dw_sampler = EmbeddingComposite(self.dw_sampler)
            if instance_id is None:
                raise ValueError("instance_id must be specified for dynamics instances")
            
            dynamics_instances = instance.load_dynamics_instances(number_time_points=num_timepoints)
            if instance_id not in dynamics_instances:
                raise ValueError(f"Dynamics instance {instance_id} not found")
            
            dyn_instance = dynamics_instances[instance_id]
            # Create BINARY BQM first, then convert to SPIN for D-Wave
            bqm_binary = dimod.BQM(dyn_instance['h'], dyn_instance['J'], 
                                   dyn_instance['offset'], vartype='BINARY')
            bqm_spin = bqm_binary.copy()
            bqm_spin.change_vartype(dimod.SPIN)
            h = dict(bqm_spin.linear)
            J = dict(bqm_spin.quadratic)
            offset = bqm_spin.offset

            response = self.dw_sampler.sample_ising(
                h=h,
                J=J,
                num_reads=num_reads,
            )

        else:  # static instances
            J_terms = instance.load_instances()
            
            if not J_terms:
                raise ValueError("No static instances found")
                
            # Select instance
            if n_nodes is None:
                n_nodes = list(J_terms.keys())[0]
            elif n_nodes not in J_terms:
                raise ValueError(f"Instance with {n_nodes} nodes not found")
                
            J = J_terms[n_nodes]
            h = {}  # No linear terms for static instances
            offset = 0.0
            
            # Execute on D-Wave with custom schedule
            response = self.dw_sampler.sample_ising(
                h=h,
                J=J,
                num_reads=num_reads,
            )
    
        # Log access time
        if 'timing' in response.info and 'qpu_access_time' in response.info['timing']:
            self._log_access_time(response.info['timing']['qpu_access_time'])
        
        # Save results
        if instance_type == 'dynamics':
            results_dir = ensure_dir(RESULT_DIR / 'forward' / str(self.sampler) / f'dynamics_{instance_id}_timepoints_{num_timepoints}')
        else:
            results_dir = ensure_dir(RESULT_DIR / 'forward' / str(self.sampler) / f'N_{n_nodes}_realization_1')
         
        # Find the next available file number
        existing_files = list(results_dir.glob('[0-9]*.npz'))
        if not existing_files:
            next_number = 1
        else:
            # Extract numbers from filenames and find the maximum
            numbers = [int(f.stem) for f in existing_files]
            next_number = max(numbers) + 1
            
        results_path = results_dir / f'{next_number}.npz'
        
        # Extract relevant information
        result_data = {
            'energies': response.record.energy,
            'solutions': response.record.sample,
            'num_occurrences': response.record.num_occurrences,
            'timing': response.info['timing'],
            'offset': offset,
            'num_timepoints': num_timepoints if instance_type == 'dynamics' else None
        }
        
        # Save results
        np.savez_compressed(results_path, **result_data)
        print(f"Results saved as: {results_path}")
        
        return response, result_data
