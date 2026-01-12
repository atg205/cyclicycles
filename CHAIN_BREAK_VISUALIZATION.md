# Chain Break Visualization Script

## Overview

The `visualize_chain_breaks.py` script performs a single annealing sample on a D-Wave problem instance and then visualizes any **chain breaks** that occur. This is useful for understanding embedding quality and which logical variables have conflicts in their qubit chains.

## What are Chain Breaks?

In D-Wave quantum annealers:
- **Logical qubits** are the problem variables you define
- **Physical qubits** are the actual qubits on the D-Wave hardware
- An **embedding** maps each logical qubit to a chain of physical qubits
- A **chain break** occurs when qubits in the same chain have different values after annealing

Chain breaks indicate embedding issues and usually result in lower solution quality.

## Features

The script performs these steps:

1. **Executes a sample**: Runs a single annealing operation on the specified problem instance
2. **Extracts embedding**: Retrieves the embedding mapping from the sampler
3. **Identifies chain breaks**: Analyzes the solution to find which chains are broken
4. **Creates visualization**: Generates an interactive HTML visualization showing:
   - **Blue nodes**: Used qubits in working (non-broken) chains
   - **Red/salmon nodes**: Qubits in broken chains
   - **Grey nodes**: Unused qubits
   - **Black edges**: Qubit connections on the QPU
5. **Generates report**: Saves a JSON file with detailed chain break information

## Installation

The script requires these packages (should already be installed):
- `networkx` - for graph operations
- `matplotlib` - for static visualization
- `plotly` (optional) - for interactive HTML visualization
- `dimod` - for BQM handling
- `dwave-system` - for D-Wave sampler access

If plotly is not installed, the script will fall back to matplotlib and save PNG images instead.

To install plotly:
```bash
pip install plotly
```

## Usage

### Basic Usage

Run with default parameters (dynamics instance 1, 5 timepoints):
```bash
./scripts/visualize_chain_breaks.py
```

### With Custom Parameters

```bash
./scripts/visualize_chain_breaks.py \
    --solver 1.10 \
    --instance-type dynamics \
    --instance-id 1 \
    --num-timepoints 5 \
    --num-reads 10 \
    --output-dir ./chain_break_visualizations
```

### Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--solver` | str | `1.10` | D-Wave solver: 1.6, 1.7, 1.8, 1.9, 1.10, 4.1, or 6.4 |
| `--instance-type` | str | `dynamics` | Instance type: 'static' or 'dynamics' |
| `--instance-id` | str | `1` | Instance ID for dynamics instances (1-8) |
| `--num-timepoints` | int | `5` | Number of timepoints for dynamics instances |
| `--output-dir` | str | `./chain_break_visualizations` | Output directory for visualizations |
| `--num-reads` | int | `10` | Number of samples to read from QPU |

## Output Files

The script generates three files in the output directory:

1. **chain_breaks_[type]_[id]_timepoints_[n].html** - Interactive visualization (Plotly)
   - Hover over nodes to see details
   - Pan and zoom supported
   
2. **chain_breaks_[type]_[id]_timepoints_[n].png** - Static visualization (Matplotlib) - *if not using Plotly*

3. **chain_breaks_[type]_[id]_timepoints_[n].json** - Detailed report containing:
   - Total number of logical variables
   - Total number of broken chains
   - Full embedding mapping
   - List of broken chains
   - Best energy found
   - Number of samples

## Examples

### Example 1: Visualize Dynamics Instance with Solver 1.10
```bash
./scripts/visualize_chain_breaks.py \
    --solver 1.10 \
    --instance-type dynamics \
    --instance-id 2 \
    --num-timepoints 5
```

### Example 2: Visualize Static Instance with Solver 6.4
```bash
./scripts/visualize_chain_breaks.py \
    --solver 6.4 \
    --instance-type static
```

### Example 3: Compare Multiple Solvers
```bash
for solver in 1.10 4.1 6.4; do
    ./scripts/visualize_chain_breaks.py --solver $solver --output-dir ./comparisons/$solver
done
```

## Interpreting the Visualization

### Color Legend

- **Light Blue (with dark blue border)**: Qubits in working chains - these chains had all qubits agree on the final value
- **Salmon/Red (with dark red border)**: Qubits in broken chains - these qubits disagreed with others in their chain
- **Light Grey (with black border)**: Unused qubits - not part of the embedding

### What to Look For

1. **High density of red nodes**: Embedding quality is poor, causing many chain breaks
2. **Clustered red nodes**: The broken chains are concentrated in certain regions
3. **No red nodes**: Perfect embedding quality (all chains are unbroken)

## Troubleshooting

### Issue: "Could not extract embedding from sampler"
- This can happen with certain D-Wave sampler configurations
- The script will still work but may not show the full topology
- Try different solver IDs with `--solver`

### Issue: "Could not extract full QPU topology"
- QPU information may not be accessible
- Visualization will be skipped but JSON report will still be generated

### Issue: "Connection timeout" or "Access denied"
- Ensure you have D-Wave token configured: `dwave config`
- Check your internet connection and D-Wave Leap credentials

### Issue: Plotly not available
- Script will automatically fall back to matplotlib
- PNG files will be generated instead of interactive HTML
- Install plotly for better interactive visualization: `pip install plotly`

## Performance Notes

- Script execution time depends on D-Wave queue time
- First run may take 30 seconds - several minutes to queue and execute
- Subsequent runs may be faster if D-Wave service is responsive
- Lower `--num-reads` values give faster results but may show chains as broken due to randomness

## Tips for Better Results

1. **Use more reads**: `--num-reads 100` or `--num-reads 1000` for more reliable chain break detection
2. **Test multiple instances**: Compare different instance IDs to understand embedding patterns
3. **Compare solvers**: Try different solver IDs (1.10 vs 4.1 vs 6.4) to see topology differences
4. **Save visualizations**: Keep output files for documentation and comparison

## Understanding the Code

### Key Functions

- `get_embedding_and_response()` - Executes sample and extracts embedding
- `identify_chain_breaks()` - Detects which chains have conflicting qubit values
- `get_qpu_topology()` - Retrieves QPU edge/qubit information
- `create_plotly_visualization()` - Creates interactive HTML visualization
- `create_matplotlib_visualization()` - Creates static PNG visualization
- `save_chain_break_report()` - Saves JSON report

## References

- [D-Wave Documentation: Embeddings](https://docs.dwavesys.com/docs/latest/c_embedding.html)
- [D-Wave Documentation: Chain Breaks](https://docs.dwavesys.com/docs/latest/c_qpu_topology.html)
- [dwave-system Python API](https://docs.ocean.dwavesys.com/en/stable/docs_system/sdk_index.html)

## Notes

- The script uses the `Runner` class from your cyclicycles project
- It respects the solver configuration in your project
- Results are compatible with the existing results directory structure
