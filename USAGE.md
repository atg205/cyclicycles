# Ancilla Transformation - Usage Guide

## Running Experiments

### Without Ancilla (default)
```python
runner.execute_cyclic_annealing(
    use_ancilla_transformation=False,
    # ... other parameters
)
```

Results saved to: `results/{solver}/dynamics_{id}_timepoints_{n}_no_ancilla_realization_1/`

### With Ancilla
```python
runner.execute_cyclic_annealing(
    use_ancilla_transformation=True,
    # ... other parameters
)
```

Results saved to: `results/{solver}/dynamics_{id}_timepoints_{n}_with_ancilla_realization_1/`

## Plotting Results

### Plot Without Ancilla (default)
```bash
python scripts/plot_dynamics.py --instance_id 2 --num_timepoints 10 --solver 6.4
```

### Plot With Ancilla
```bash
python scripts/plot_dynamics.py --instance_id 2 --num_timepoints 10 --solver 6.4 --ancilla
```

### Plot All Static Instances Without Ancilla
```bash
python scripts/plot_results.py --solver 6.4
```

### Plot All Static Instances With Ancilla
```bash
python scripts/plot_results.py --solver 6.4 --ancilla
```

## Key Parameters

- `use_ancilla_transformation` (bool): Set to True/False in runner methods
- `--ancilla` flag (scripts): Add flag to plot with ancilla, omit to plot without
