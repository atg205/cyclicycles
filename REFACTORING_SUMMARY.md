# Runner.py Code Refactoring Summary

## Problem Identified
The `runner.py` module had significant code duplication when handling dynamic instances:
- **Issue**: Both `execute_cyclic_annealing()` and `execute_instance()` duplicated the exact same logic for:
  1. Loading dynamics instances
  2. Applying ancilla transformation (optional)
  3. Converting BINARY to SPIN BQM
  4. Extracting h, J, offset from converted BQM

This duplication made the code harder to maintain and led to potential for inconsistencies.

## Solution: Extract Helper Method

### New Method: `_load_and_prepare_problem()`

Created a private helper method that consolidates all problem loading and transformation logic:

```python
def _load_and_prepare_problem(self, n_nodes: str | None, instance_type: str, instance_id: str | None,
                              num_timepoints: int, use_ancilla_transformation: bool = False, 
                              ancilla_ratio: int = 1):
    """Load and prepare a problem instance for annealing."""
```

**Responsibilities:**
1. Load instances (static or dynamics) based on `instance_type`
2. Apply ancilla transformation if requested (eliminates linear h-terms)
3. Convert BINARY BQM to SPIN for D-Wave compatibility
4. Return unified problem dict with all necessary data and metadata

**Returns:** Single dict containing:
- `h`: Linear terms (SPIN converted)
- `J`: Quadratic terms (SPIN converted)
- `offset`: Energy offset
- `vartype`: Problem vartype ('SPIN')
- Metadata: `instance_type`, `instance_id`, `num_timepoints`, `n_nodes`
- Ancilla info: `used_ancilla` (bool), `ancilla_ratio` (int or None)

## Refactored Methods

### 1. `execute_cyclic_annealing()`

**Before:** ~80 lines of instance loading + transformation code duplicated
**After:** Single call to `_load_and_prepare_problem()`

New parameters added:
- `use_ancilla_transformation: bool = False`
- `ancilla_ratio: int = 1`

Code flow simplified to:
```python
# Setup embedding
if instance_type == 'dynamics':
    self.dw_sampler = EmbeddingComposite(self.dw_sampler)

# Single unified problem loading
problem = self._load_and_prepare_problem(n_nodes, instance_type, instance_id, 
                                         num_timepoints, use_ancilla_transformation, 
                                         ancilla_ratio)

h = problem['h']
J = problem['J']
offset = problem['offset']
n_nodes = problem['n_nodes']

# Rest of cyclic annealing logic (unchanged)
```

Forward annealing call now passes new parameters:
```python
forward_response, _ = self.execute_instance(
    n_nodes=n_nodes, num_reads=num_reads, 
    instance_type=instance_type, instance_id=instance_id,
    num_timepoints=num_timepoints,
    use_ancilla_transformation=use_ancilla_transformation,  # NEW
    ancilla_ratio=ancilla_ratio  # NEW
)
```

Result saving now includes ancilla metadata:
```python
result_data = {
    **final_response_info,
    # ... existing fields ...
    'used_ancilla': problem['used_ancilla'],  # NEW
    'ancilla_ratio': problem['ancilla_ratio'] if problem['used_ancilla'] else None  # NEW
}
```

### 2. `execute_instance()`

**Before:** ~50 lines of instance loading + transformation code
**After:** Single call to `_load_and_prepare_problem()`

New parameters added:
- `use_ancilla_transformation: bool = False`
- `ancilla_ratio: int = 1`

Code flow simplified to:
```python
# Setup embedding
if instance_type == 'dynamics':
    self.dw_sampler = EmbeddingComposite(self.dw_sampler)

# Single unified problem loading
problem = self._load_and_prepare_problem(n_nodes, instance_type, instance_id, 
                                         num_timepoints, use_ancilla_transformation, 
                                         ancilla_ratio)

h = problem['h']
J = problem['J']
offset = problem['offset']
n_nodes = problem['n_nodes']

# Sample on D-Wave
response = self.dw_sampler.sample_ising(h=h, J=J, num_reads=num_reads)

# Rest of execution logic (unchanged)
```

Result saving now includes ancilla metadata:
```python
result_data = {
    'energies': response.record.energy,
    'solutions': response.record.sample,
    'num_occurrences': response.record.num_occurrences,
    'timing': response.info['timing'],
    'offset': offset,
    'instance_type': problem['instance_type'],  # Now from helper
    'num_timepoints': problem['num_timepoints'],  # Now from helper
    'used_ancilla': problem['used_ancilla']  # NEW
}
```

## Benefits of Refactoring

### 1. **Elimination of Code Duplication**
   - ~60-80 lines of duplicated logic now exist in single location
   - Single source of truth for problem loading and transformation

### 2. **Better Maintainability**
   - Future changes to instance loading logic only need one place update
   - Reduced risk of inconsistencies between methods
   - Easier to test problem loading in isolation

### 3. **Consistent Behavior**
   - Both `execute_instance()` and `execute_cyclic_annealing()` handle:
     - Static/dynamics instances identically
     - Ancilla transformation identically
     - BINARY→SPIN conversion identically
   - Forward initialization in cyclic annealing now uses same code as regular execution

### 4. **Unified Metadata Tracking**
   - Both methods now track ancilla usage consistently
   - Problem information returned in standard dict format
   - Easy to extend with new problem parameters in future

### 5. **Cleaner Method Signatures**
   - `execute_instance()` and `execute_cyclic_annealing()` now have clear contracts
   - All problem setup delegated to helper
   - Core annealing logic remains focused and readable

## Usage Examples

### Execute cyclic annealing on dynamics instance without ancilla transformation:
```python
runner = Runner(sampler='6.4')
response, result_data, cycle_energies = runner.execute_cyclic_annealing(
    instance_type='dynamics',
    instance_id='1',
    num_timepoints=5,
    use_forward_init=True
)
```

### Execute cyclic annealing on dynamics instance with ancilla transformation:
```python
runner = Runner(sampler='6.4')
response, result_data, cycle_energies = runner.execute_cyclic_annealing(
    instance_type='dynamics',
    instance_id='1',
    num_timepoints=5,
    use_forward_init=True,
    use_ancilla_transformation=True,  # NEW
    ancilla_ratio=1  # NEW - one ancilla per qubit
)
```

### Execute single forward annealing with ancilla:
```python
runner = Runner(sampler='6.4')
response, result_data = runner.execute_instance(
    instance_type='dynamics',
    instance_id='1',
    num_timepoints=5,
    use_ancilla_transformation=True,
    ancilla_ratio=2  # NEW - one ancilla per 2 qubits
)
```

## Integration with Other Components

### `Instance.py` Integration
- `_load_and_prepare_problem()` calls:
  - `Instance.load_instances()` - for static instances
  - `Instance.load_dynamics_instances()` - for dynamics instances
  - `Instance.remove_linear_terms_with_ancilla()` - for ancilla transformation

### `Plotter.py` Compatibility
- Result metadata now includes `used_ancilla` and `ancilla_ratio`
- Plotter can use this to filter/annotate results appropriately
- Forward annealing results also track ancilla usage

## Future Extensibility

The refactored structure makes it easy to add new features:
1. **Different problem types**: Add new `elif instance_type == 'xyz'` block
2. **New transformations**: Add similar transformation calls in `_load_and_prepare_problem()`
3. **Problem filtering/validation**: Add preprocessing step before returning problem dict
4. **Custom vartype handling**: Extend return dict with additional fields

## Code Statistics

**Lines Changed:**
- Before: ~130 lines of duplicated logic across 2 methods
- After: ~100 lines of helper + ~60 lines per method = ~220 total (with comments)
- Net effect: Centralized common logic, clearer method responsibilities
