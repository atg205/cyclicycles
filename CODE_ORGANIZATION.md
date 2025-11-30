# Code Organization Diagram

## Before Refactoring

```
runner.py
│
├─ execute_cyclic_annealing()
│  ├─ [DUPLICATE] Load dynamics instances
│  ├─ [DUPLICATE] Apply ancilla transformation
│  ├─ [DUPLICATE] Convert BINARY → SPIN
│  ├─ [DUPLICATE] Prepare h, J, offset
│  ├─ Execute cyclic annealing
│  └─ Save results
│
├─ execute_instance()
│  ├─ [DUPLICATE] Load dynamics instances
│  ├─ [DUPLICATE] Apply ancilla transformation  (currently always uses ancilla_ratio=1)
│  ├─ [DUPLICATE] Convert BINARY → SPIN
│  ├─ [DUPLICATE] Prepare h, J, offset
│  ├─ Execute forward annealing
│  └─ Save results
│
└─ instance.py
   ├─ load_instances()
   ├─ load_dynamics_instances()
   └─ remove_linear_terms_with_ancilla()
```

**Problem:** When forward annealing is called from cyclic annealing, dynamic instances are loaded and transformed TWICE with different ancilla settings.

---

## After Refactoring

```
runner.py
│
├─ _load_and_prepare_problem()  ← NEW HELPER METHOD
│  ├─ Load instances (static or dynamics)
│  ├─ Apply ancilla transformation (if requested)
│  ├─ Convert BINARY → SPIN
│  └─ Return unified problem dict
│
├─ execute_cyclic_annealing()
│  ├─ Call _load_and_prepare_problem()
│  ├─ Execute cyclic annealing
│  ├─ Optionally call execute_instance() for forward init
│  │  └─ execute_instance() also calls _load_and_prepare_problem()
│  │     (Uses SAME helper, guaranteed consistency)
│  └─ Save results (with ancilla metadata)
│
├─ execute_instance()
│  ├─ Call _load_and_prepare_problem()
│  ├─ Execute forward annealing
│  └─ Save results (with ancilla metadata)
│
└─ instance.py
   ├─ load_instances()
   ├─ load_dynamics_instances()
   └─ remove_linear_terms_with_ancilla()
```

**Benefits:**
- ✓ Single source of truth for problem loading
- ✓ Consistent ancilla handling
- ✓ No duplicate transformation code
- ✓ Easy to extend with new problem types or transformations

---

## Data Flow: Cyclic Annealing with Forward Initialization and Ancilla

```
execute_cyclic_annealing(
    use_ancilla_transformation=True,
    ancilla_ratio=1,
    use_forward_init=True
)
│
├─ 1. _load_and_prepare_problem()
│  │  ├─ Load dynamics instance from JSON
│  │  ├─ Extract h, J, offset
│  │  ├─ Call Instance.remove_linear_terms_with_ancilla()
│  │  │  └─ Transforms h-terms to ancilla couplings
│  │  ├─ Create BINARY BQM with transformed problem
│  │  ├─ Convert to SPIN BQM
│  │  └─ Return: {h, J, offset, used_ancilla: True, ...}
│  │
│  └─ problem = {..., used_ancilla: True, ...}
│
├─ 2. [if use_forward_init] Call execute_instance()
│  │
│  └─ execute_instance() also calls _load_and_prepare_problem()
│     ├─ Loads SAME dynamics instance
│     ├─ Applies SAME ancilla transformation
│     ├─ Converts to SPIN with SAME parameters
│     ├─ Sample on QPU
│     └─ Return best solution for cyclic init
│
├─ 3. Cyclic annealing loop
│  └─ Use best_state from forward annealing
│
└─ 4. Save results
   ├─ cycle_energies
   ├─ used_ancilla: True
   ├─ ancilla_ratio: 1
   └─ used_forward_init: True
```

---

## Method Signatures Comparison

### Before
```python
# execute_cyclic_annealing - always applied ancilla!
def execute_cyclic_annealing(self, n_nodes=None, num_cycles=5, num_reads=1000,
                             use_forward_init=False, instance_type='static', 
                             instance_id=None, num_timepoints=5):
    # Hardcoded: ancilla_ratio=1 in remove_linear_terms_with_ancilla()
    dyn_instance = instance.remove_linear_terms_with_ancilla(h, J, ancilla_ratio=1, offset=offset)
    ...

# execute_instance - no ancilla support
def execute_instance(self, n_nodes=None, num_reads=1000, instance_type='static',
                     instance_id=None, num_timepoints=5):
    # No ancilla transformation capability
    bqm_binary = dimod.BQM(dyn_instance['h'], dyn_instance['J'], ...)
    ...
```

### After
```python
# Helper method - centralized logic
def _load_and_prepare_problem(self, n_nodes, instance_type, instance_id, num_timepoints,
                              use_ancilla_transformation=False, ancilla_ratio=1):
    if use_ancilla_transformation:
        dyn_instance = instance.remove_linear_terms_with_ancilla(h, J, 
                                                                  ancilla_ratio=ancilla_ratio, 
                                                                  offset=offset)
    ...

# execute_cyclic_annealing - uses helper
def execute_cyclic_annealing(self, n_nodes=None, num_cycles=5, num_reads=1000,
                             use_forward_init=False, instance_type='static',
                             instance_id=None, num_timepoints=5,
                             use_ancilla_transformation=False, ancilla_ratio=1):  # NEW params
    problem = self._load_and_prepare_problem(n_nodes, instance_type, instance_id,
                                             num_timepoints, use_ancilla_transformation,
                                             ancilla_ratio)  # Delegates to helper
    ...

# execute_instance - uses helper
def execute_instance(self, n_nodes=None, num_reads=1000, instance_type='static',
                     instance_id=None, num_timepoints=5,
                     use_ancilla_transformation=False, ancilla_ratio=1):  # NEW params
    problem = self._load_and_prepare_problem(n_nodes, instance_type, instance_id,
                                             num_timepoints, use_ancilla_transformation,
                                             ancilla_ratio)  # Delegates to helper
    ...
```

**Key Improvements:**
1. ✓ Ancilla transformation is now OPTIONAL (parameter)
2. ✓ Ancilla ratio is now CONFIGURABLE (parameter)
3. ✓ Both methods use IDENTICAL logic via helper
4. ✓ Forward initialization applies same transformation as cyclic

---

## File Dependencies

```
runner.py
├─ imports instance.py
│  ├─ Instance.load_instances()
│  ├─ Instance.load_dynamics_instances()
│  └─ Instance.remove_linear_terms_with_ancilla()
│
├─ imports dimod
│  └─ BQM.change_vartype()
│
└─ imports D-Wave SDK
   ├─ DWaveSampler
   └─ EmbeddingComposite
```

All dependencies remain unchanged. The refactoring is internal to runner.py.
