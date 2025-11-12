from pathlib import Path
import re
import numpy as np
from config import INSTANCE_DIR, DYNAMICS_INSTANCE_DIR
import json
import dimod

class Instance:
    def __init__(self,solver='4.1'):
        self.J_terms = {}
        self.solver = solver
        self.instance_dir = INSTANCE_DIR / solver

        
    def load_instances(self, realization_number: int = 1):
        """Load all instances with N_ in folder name from the data directory.
        
        Args:
            realization_number (int): The realization number to load. Defaults to 1.
            
        Returns:
            dict: Dictionary with node numbers as keys and J terms as values.
        """
        # Get all directories that match the pattern N_*_realization_*
        for path in self.instance_dir.glob(f'N_*_realization_{realization_number}'):
            if not path.is_dir():
                continue
                
            # Extract N from the directory name using regex
            match = re.match(r'N_(\d+)_realization_', path.name)
            if not match:
                continue
                
            n_nodes = match.group(1)
            
            # Load J terms
            j_path = path / 'J.npz'
            if j_path.exists():
                self.J_terms[n_nodes] = np.load(j_path, allow_pickle=True)['J'].item()
        
        return self.J_terms
    
    def load_dynamics_instances(self, number_time_points: int = 5):
        """Load all dynamic instances from the dynamics directory.
        
        Args:
            number_time_points (int): Number of time points. Defaults to 5.
            
        Returns:
            dict: Dictionary with instance IDs as keys and BQM dicts (h, J, offset) as values.
        """
        dynamics_instances = {}
        
        if not DYNAMICS_INSTANCE_DIR.exists():
            print(f"Warning: Dynamics instance directory not found: {DYNAMICS_INSTANCE_DIR}")
            return dynamics_instances
        
        # Get all subdirectories (instance IDs)
        for instance_path in DYNAMICS_INSTANCE_DIR.iterdir():
            if not instance_path.is_dir():
                continue
            
            instance_id = instance_path.name
            
            # Find the file with the correct timepoints (ignoring precision)
            # Files follow pattern: precision_{x}_timepoints_{timepoints}.json
            matching_files = list(instance_path.glob(f"*_timepoints_{number_time_points}.json"))
            
            if not matching_files:
                continue
            
            # Use the first (and should be only) matching file
            file_path = matching_files[0]
            
            try:
                with open(file_path, 'r') as f:
                    bqm_data = json.load(f)
                
                # Convert to BQM and extract h, J, offset
                bqm = dimod.BQM.from_serializable(bqm_data)
                
                dynamics_instances[instance_id] = {
                    'h': dict(bqm.linear),
                    'J': dict(bqm.quadratic),
                    'offset': bqm.offset,
                    'num_variables': len(bqm.linear)
                }
            except Exception as e:
                print(f"Warning: Could not load dynamics instance {instance_id}: {e}")
                continue
        
        return dynamics_instances
    
    def load_all_instances(self, realization_number: int = 1, include_dynamics: bool = True):
        """Load both static and dynamic instances.
        
        Args:
            realization_number (int): The realization number for static instances. Defaults to 1.
            include_dynamics (bool): Whether to include dynamic instances. Defaults to True.
            
        Returns:
            tuple: (static_instances_dict, dynamics_instances_dict)
        """
        static = self.load_instances(realization_number)
        dynamics = self.load_dynamics_instances() if include_dynamics else {}
        
        return static, dynamics
