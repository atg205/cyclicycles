from pathlib import Path
import re
import numpy as np
from config import DATA_DIR

class Instance:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.J_terms = {}
        
    def load_instances(self, realization_number: int = 1):
        """Load all instances with N_ in folder name from the data directory.
        
        Args:
            realization_number (int): The realization number to load. Defaults to 1.
            
        Returns:
            dict: Dictionary with node numbers as keys and J terms as values.
        """
        # Get all directories that match the pattern N_*_realization_*
        print(self.data_dir)
        print([file for file in self.data_dir.glob(f'N_*_realization_{realization_number}')])
        for path in self.data_dir.glob(f'N_*_realization_{realization_number}'):
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
