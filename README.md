# Cyclicycles

Implementation of cyclic quantum annealing using D-Wave quantum processors.

## Overview

This project implements and compares forward and cyclic quantum annealing approaches for solving Ising spin glass problems. Cyclic annealing iteratively uses the best solution from the previous cycle as the initial state for the next cycle.

## Requirements

- Python 3.11+
- Poetry for dependency management
- D-Wave Ocean SDK
- Access to D-Wave quantum processors

## Installation

```bash
# Clone the repository
git clone https://github.com/atg205/cyclicycles.git
cd cyclicycles

# Install dependencies using Poetry
poetry install
```

## Usage

### Running Annealing

Forward annealing:
```bash
./scripts/forward_annealing.py --n_nodes 263 --num_reads 1000 --sampler 6.4
```

Cyclic annealing:
```bash
./scripts/cyclic_annealing.py --n_nodes 263 --num_cycles 5 --num_reads 1000 --sampler 6.4
```

Parameters:
- `--n_nodes`: Number of nodes in the instance (263, 678, 958, 1312, 2084, 5627)
- `--num_reads`: Number of annealing reads per cycle
- `--num_cycles`: Number of cycles (cyclic annealing only)
- `--sampler`: D-Wave processor version (1.6=Advantage2, 4.1/6.4=Advantage)

### Plotting Results

Generate plots comparing forward and cyclic annealing:
```bash
./scripts/plot_results.py --solver 6.4 --n_nodes 263 --save_dir plots/
```

Parameters:
- `--solver`: D-Wave processor version
- `--n_nodes`: Specific instance to plot (optional)
- `--save_dir`: Directory to save plots (optional)

## Project Structure

- `src/cyclicycles/`
  - `runner.py`: Implementation of forward and cyclic annealing
  - `instance.py`: Problem instance handling
  - `plotter.py`: Visualization and analysis
- `scripts/`: Command-line interfaces
- `data/`: Problem instances and results
