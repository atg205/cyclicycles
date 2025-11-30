# Quick Reference: Ancilla Transformation Parameter

## Running Experiments

### Without Ancilla (Baseline)
```python
from cyclicycles.runner import Runner

runner = Runner(sampler='6.4')

response, results, cycle_energies = runner.execute_cyclic_annealing(
    num_reads=1000,
    num_cycles=5,
    instance_type='dynamics',
    instance_id='2',
    num_timepoints=10,
    use_ancilla_transformation=False  # No ancilla
)
```

**Results saved to:** `results/6.4/dynamics_2_timepoints_10_no_ancilla_realization_1/`

### With Ancilla Transformation
```python
response, results, cycle_energies = runner.execute_cyclic_annealing(
    num_reads=1000,
    num_cycles=5,
    instance_type='dynamics',
    instance_id='2',
    num_timepoints=10,
    use_ancilla_transformation=True  # With ancilla
)
```

**Results saved to:** `results/6.4/dynamics_2_timepoints_10_with_ancilla_realization_1/`

## Plotting Results

### Plot Without Ancilla Only
```bash
python scripts/plot_dynamics.py \
    --instance_id 2 \
    --num_timepoints 10 \
    --solver 6.4 \
    --ancilla without \
    --save_dir ./plots
```

Output: `gap_dynamics_2_timepoints_10_solver6.4_no_ancilla.png`

### Plot With Ancilla Only
```bash
python scripts/plot_dynamics.py \
    --instance_id 2 \
    --num_timepoints 10 \
    --solver 6.4 \
    --ancilla with \
    --save_dir ./plots
```

Output: `gap_dynamics_2_timepoints_10_solver6.4_with_ancilla.png`

### Auto-Select (Uses First Available)
```bash
python scripts/plot_dynamics.py \
    --instance_id 2 \
    --num_timepoints 10 \
    --solver 6.4 \
    --ancilla both
```

Automatically detects and plots whichever variant exists.

## Static Instances

All the same logic applies to static instances:

```bash
# Static without ancilla
python scripts/plot_results.py --n_nodes 263 --solver 6.4 --ancilla without

# Static with ancilla
python scripts/plot_results.py --n_nodes 263 --solver 6.4 --ancilla with

# Plot all static instances (without ancilla)
python scripts/plot_results.py --solver 6.4 --ancilla without
```

## Forward Annealing

Results for forward annealing are organized the same way:

```python
response, results = runner.execute_instance(
    instance_type='dynamics',
    instance_id='2',
    num_timepoints=10,
    use_ancilla_transformation=False  # or True
)
```

**Results saved to:** `results/forward/6.4/dynamics_2_timepoints_10_no_ancilla/`

## Common Arguments Reference

### plot_dynamics.py
```
--solver {1.8,4.1,6.4}          Solver to plot (default: 6.4)
--instance_id STR                Instance ID (required)
--num_timepoints INT             Number of timepoints (required)
--ancilla {with,without,both}   Ancilla variant (default: both)
--init {forward,zero,all}        Initialization type (default: all)
--num_samples {100,1000}         Filter by sample count (optional)
--save_dir PATH                  Save to directory (optional)
```

### plot_results.py
```
--solver {1.8,4.1,6.4}          Solver to plot (default: 6.4)
--n_nodes INT                    Plot specific N (optional)
--ancilla {with,without,both}   Ancilla variant (default: both)
--init {forward,zero,all}        Initialization type (default: all)
--num_samples {100,1000}         Filter by sample count (default: 1000)
--save_dir PATH                  Save to directory (optional)
```

## Filename Convention

- **With Ancilla:** `*_with_ancilla.png`
- **Without Ancilla:** `*_no_ancilla.png`
- **Auto-detected:** No suffix (or suffix based on which was found)

## Verify Your Setup

Check that you can load results for both variants:

```python
from cyclicycles.plotter import Plotter
from cyclicycles.config import RESULT_DIR

plotter = Plotter(RESULT_DIR)

# This should work after running experiments with both variants
df_with, stats_with, offset = plotter.load_results(
    solver_id='6.4',
    instance_type='dynamics',
    instance_id='2',
    num_timepoints=10,
    use_ancilla=True
)

df_without, stats_without, offset = plotter.load_results(
    solver_id='6.4',
    instance_type='dynamics',
    instance_id='2',
    num_timepoints=10,
    use_ancilla=False
)

print(f"With ancilla: {len(df_with.columns)} runs")
print(f"Without ancilla: {len(df_without.columns)} runs")
```

## Notes

- Gap is automatically calculated as `min_energy + offset` for all plots
- Results include metadata (`used_ancilla`, `ancilla_ratio`) for verification
- Forward annealing results are stored in `results/forward/` subdirectories
- Both variants use the same solver and instance, so comparison is valid
