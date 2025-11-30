# Ancilla Transformation Parameter Changes

## Summary

Added a `use_ancilla` parameter throughout the codebase to distinguish between results with and without ancilla transformation. Results are now saved in separate directories and can be plotted independently for comparison.

## Changes Made

### 1. **runner.py** - Result Directory Structure

Modified result saving logic to include ancilla status in directory names:

**Cyclic Annealing (`execute_cyclic_annealing`):**
- **With ancilla:** `results/{solver_id}/dynamics_{instance_id}_timepoints_{num_timepoints}_with_ancilla_realization_1/`
- **Without ancilla:** `results/{solver_id}/dynamics_{instance_id}_timepoints_{num_timepoints}_no_ancilla_realization_1/`
- Static instances: `results/{solver_id}/N_{n_nodes}_with_ancilla_realization_1/` or `N_{n_nodes}_no_ancilla_realization_1/`

**Forward Annealing (`execute_instance`):**
- **With ancilla:** `results/forward/{solver_id}/dynamics_{instance_id}_timepoints_{num_timepoints}_with_ancilla/`
- **Without ancilla:** `results/forward/{solver_id}/dynamics_{instance_id}_timepoints_{num_timepoints}_no_ancilla/`
- Static instances: `results/forward/{solver_id}/N_{n_nodes}_with_ancilla_realization_1/` or `N_{n_nodes}_no_ancilla_realization_1/`

### 2. **plotter.py** - Enhanced Plotting Functions

#### `load_results()` method
- **New parameter:** `use_ancilla: bool | None = None`
  - `True`: Load only results with ancilla transformation
  - `False`: Load only results without ancilla transformation
  - `None`: Automatically detect and load whichever exists (prefers no_ancilla if both exist)

#### `plot_instance()` method
- **New parameter:** `use_ancilla: bool | None = None`
- Title includes ancilla status (e.g., ", with_ancilla" or ", no_ancilla")
- Saved filenames include ancilla suffix (e.g., `_with_ancilla.png` or `_no_ancilla.png`)

### 3. **scripts/plot_dynamics.py**

Added new command-line argument:
```bash
--ancilla {with,without,both}    # Default: 'both'
```

**Example usage:**
```bash
python scripts/plot_dynamics.py --instance_id 2 --num_timepoints 10 --ancilla with
python scripts/plot_dynamics.py --instance_id 2 --num_timepoints 10 --ancilla without
python scripts/plot_dynamics.py --instance_id 2 --num_timepoints 10 --ancilla both
```

### 4. **scripts/plot_results.py**

Added new command-line argument:
```bash
--ancilla {with,without,both}    # Default: 'both'
```

Enhanced directory globbing to handle both ancilla and non-ancilla variants when plotting all instances.

**Example usage:**
```bash
python scripts/plot_results.py --solver 6.4 --ancilla with
python scripts/plot_results.py --solver 6.4 --n_nodes 263 --ancilla without
python scripts/plot_results.py --solver 6.4 --ancilla both
```

### 5. **src/cyclicycles/main.py** - Example Usage

Updated example to demonstrate testing both variants:
```python
for use_ancilla in [True, False]:
    response, results, cycle_energies = runner.execute_cyclic_annealing(
        use_ancilla_transformation=use_ancilla,
        # ... other parameters
    )
```

## Directory Structure Example

After running experiments with both variants:

```
results/
├── 6.4/
│   ├── dynamics_2_timepoints_10_with_ancilla_realization_1/
│   │   ├── 1.npz
│   │   ├── 2.npz
│   │   └── 3.npz
│   ├── dynamics_2_timepoints_10_no_ancilla_realization_1/
│   │   ├── 1.npz
│   │   ├── 2.npz
│   │   └── 3.npz
│   ├── N_263_with_ancilla_realization_1/
│   │   └── ...
│   └── N_263_no_ancilla_realization_1/
│       └── ...
└── forward/
    └── 6.4/
        ├── dynamics_2_timepoints_10_with_ancilla/
        │   └── 1.npz
        ├── dynamics_2_timepoints_10_no_ancilla/
        │   └── 1.npz
        └── ...
```

## Plotting Capabilities

### Separate Comparisons

Plot results with ancilla:
```bash
python scripts/plot_dynamics.py --instance_id 2 --num_timepoints 10 --ancilla with --save_dir ./plots
```

Plot results without ancilla:
```bash
python scripts/plot_dynamics.py --instance_id 2 --num_timepoints 10 --ancilla without --save_dir ./plots
```

This generates:
- `gap_dynamics_2_timepoints_10_solver6.4_with_ancilla.png`
- `gap_dynamics_2_timepoints_10_solver6.4_no_ancilla.png`

### Manual Comparison in Python

```python
from pathlib import Path
from cyclicycles.plotter import Plotter
from cyclicycles.config import RESULT_DIR

plotter = Plotter(RESULT_DIR)

# Plot both variants side by side (conceptually - user can run twice)
plotter.plot_instance(
    solver_id='6.4',
    instance_type='dynamics',
    instance_id='2',
    num_timepoints=10,
    use_ancilla=False,
    save_dir='./plots'
)

plotter.plot_instance(
    solver_id='6.4',
    instance_type='dynamics',
    instance_id='2',
    num_timepoints=10,
    use_ancilla=True,
    save_dir='./plots'
)
```

## Backward Compatibility

- Default `use_ancilla=None` in plotting functions attempts automatic detection
- When `use_ancilla=None` and both directories exist, `no_ancilla` is preferred
- When `use_ancilla=None` and only one exists, that variant is used

## File Metadata

All result `.npz` files continue to include:
- `used_ancilla` (boolean): Whether ancilla transformation was applied
- `ancilla_ratio` (int or None): Ratio used if ancilla transformation applied
- `offset`: Problem offset for gap calculations

These fields help verify which variant was used even without relying solely on directory structure.

## Migration Notes

Since you've deleted all previous record files, you can now:
1. Run experiments with `use_ancilla_transformation=False` to establish baseline
2. Run experiments with `use_ancilla_transformation=True` to test new method
3. Compare results directly using the plotting scripts with `--ancilla with` vs `--ancilla without`
