# Embedding Caching Implementation Summary

## Overview
Implemented transparent embedding caching for the D-Wave quantum annealing runner. This prevents redundant computation of problem embeddings (expensive D-Wave operations) by automatically saving and reusing embeddings for identical problems.

**Key Technical Approach:** 
1. **First run (cache miss):** Use `EmbeddingComposite` to sample with `return_embedding=True`, extract the embedding from response info, cache the embedding dict, then return `FixedEmbeddingComposite` with cached embedding
2. **Subsequent runs (cache hit):** Load cached embedding dict, directly use `FixedEmbeddingComposite` with it (no computation needed)

## Key Changes

### 1. Added `pickle` and `FixedEmbeddingComposite` Imports
**File:** `src/cyclicycles/runner.py` (Line 4 and Line 6)
- Added `import pickle` for serializing/deserializing embedding mapping dictionaries to disk
- Added `FixedEmbeddingComposite` to imports from `dwave.system` (used when loading cached embeddings)

### 2. Implemented `_get_or_create_embedding()` Method
**File:** `src/cyclicycles/runner.py` (Lines 63-145)

This new method handles embedding caching transparently using the proper D-Wave approach:

**Functionality:**
- Takes problem parameters: `instance_type`, `instance_id`, `num_timepoints`, `n_nodes`, `h`, `J`
- Generates a cache filename based on problem parameters:
  - Dynamics: `{solver_id}_dynamics_{instance_id}_timepoints_{num_timepoints}.pkl`
  - Static: `{solver_id}_static_N{n_nodes}.pkl`
- Checks if cached embedding exists in `data/embeddings/` directory

**On Cache Hit:**
- Loads embedding dict from pickle file
- Creates `FixedEmbeddingComposite(self.dw_sampler, embedding=embedding_dict)`
- Returns it (no computation needed - instant)

**On Cache Miss:**
1. Creates `EmbeddingComposite(self.dw_sampler)` 
2. Creates BQM from h, J parameters
3. **Calls `sample(bqm, num_reads=1, return_embedding=True)`** - Returns embedding in response
4. Extracts embedding dict from `response.info['embedding_context']['embedding']`
5. Saves embedding dict to pickle file cache
6. Creates `FixedEmbeddingComposite(self.dw_sampler, embedding=embedding_dict)`
7. Returns it (computation done only once)

**Key Features:**
- **Proper D-Wave API usage:** Uses `return_embedding=True` parameter to extract embedding from response
- **Caches only the embedding dict:** Pickle-safe (no thread locks or unpicklable objects)
- **Graceful fallback:** If extraction fails, returns plain `EmbeddingComposite` without breaking execution
- Automatic cache directory creation via `ensure_dir(cache_dir)`
- Informative console output:
  - `"Loaded cached embedding from {path}"` on cache hit
  - `"Computing new embedding (this may take a moment)..."` on cache miss
  - `"Computed embedding with X logical qubits"` after extraction
  - `"Successfully cached embedding to {path}"` on successful save
- Supports both static and dynamics instances with appropriate cache keys
- Used by both forward and cyclic annealing methods

### 3. Updated `execute_cyclic_annealing()` Method
**File:** `src/cyclicycles/runner.py` (Lines 289-301)

**Changes:**
- **Before:** Created fresh `EmbeddingComposite` only for dynamics instances (non-cached)
- **After:** 
  - Loads and prepares problem first (to get instance details and h, J terms)
  - Extracts h, J, offset, n_nodes from problem dict
  - For dynamics instances, calls `_get_or_create_embedding(instance_type, instance_id, num_timepoints, n_nodes, h, J)` instead of direct `EmbeddingComposite()` creation
  - The returned sampler (either `FixedEmbeddingComposite` with cached mapping or fresh `EmbeddingComposite`) is used for all subsequent sampling

### 4. Updated `execute_instance()` Method
**File:** `src/cyclicycles/runner.py` (Lines 450-457)

**Changes:**
- **Before:** Created fresh `EmbeddingComposite` only for dynamics instances (non-cached)
- **After:**
  - Loads and prepares problem first (to get instance details and h, J terms)
  - Extracts h, J, offset, n_nodes from problem dict
  - For dynamics instances, calls `_get_or_create_embedding(instance_type, instance_id, num_timepoints, n_nodes, h, J)` instead of direct `EmbeddingComposite()` creation
  - The returned sampler (either `FixedEmbeddingComposite` with cached mapping or fresh `EmbeddingComposite`) is used for all subsequent sampling

## How It Works

### Execution Flow

1. **User calls forward or cyclic annealing** on a dynamics instance
2. **Runner loads and prepares the problem** to determine instance type/ID/timepoints and extract h, J terms
3. **For dynamics instances:**
   - Runner calls `_get_or_create_embedding(instance_type='dynamics', instance_id=..., num_timepoints=..., n_nodes=..., h=..., J=...)`
   - Embedding cache checks: Does `data/embeddings/6.4_dynamics_{id}_timepoints_{t}.pkl` exist?
   - **If yes:** 
     - Load cached embedding dict from pickle file
     - Create `FixedEmbeddingComposite(self.dw_sampler, embedding=embedding_dict)`
     - Return it ✓ (instant, no computation)
   - **If no:** 
     - Create `EmbeddingComposite(self.dw_sampler)`
     - Create BQM from h, J
     - Call `sample(bqm, num_reads=1, return_embedding=True)` to get embedding
     - Extract embedding dict from `response.info['embedding_context']['embedding']`
     - Save embedding dict to cache
     - Create `FixedEmbeddingComposite(self.dw_sampler, embedding=embedding_dict)`
     - Return it (computation done, cached for future use)
4. **For static instances:** No embedding is computed (they don't need it)
5. **Sampling proceeds** with the returned sampler (either cached `FixedEmbeddingComposite` or fresh `EmbeddingComposite`)

### Cache Structure

```
data/
  embeddings/              # New cache directory
    6.4_dynamics_N_263_realization_1_timepoints_5.pkl
    6.4_dynamics_N_678_realization_1_timepoints_5.pkl
    6.4_static_N1312.pkl
    6.4_static_N2084.pkl
    ...
```

Each file is a pickled `EmbeddingComposite` object, tagged with:
- Solver ID (e.g., `6.4`, `4.1`, `1.8`)
- Instance type (`static` or `dynamics`)
- Instance ID or node count
- Timepoints (for dynamics)

## Benefits

1. **Performance:** Embeddings computed once per problem, reused infinitely via `FixedEmbeddingComposite`
2. **Correctness:** Uses proper D-Wave API (`return_embedding=True`) to extract embeddings
3. **Transparency:** No user parameters needed - automatic default behavior
4. **Sharing:** Forward and cyclic annealing use same cached embedding (no distinction needed)
5. **Robustness:** Error handling prevents failures (graceful fallback to non-cached if needed)
6. **Clarity:** Console output shows cache hits/misses for debugging

## Technical Details

### Why Reorder Problem Loading?

Both execute methods now load the problem **before** setting up embeddings:
```python
# OLD (embedded creation first, problem details unknown)
if instance_type == 'dynamics':
    self.dw_sampler = EmbeddingComposite(self.dw_sampler)
problem = self._load_and_prepare_problem(...)

# NEW (problem loaded first, h/J available for embedding computation)
problem = self._load_and_prepare_problem(...)
h = problem['h']
J = problem['J']
if instance_type == 'dynamics':
    self.dw_sampler = self._get_or_create_embedding(..., h, J)
```

This is necessary because:
- Problem loading extracts `instance_id` and `num_timepoints` needed for cache keys
- h and J terms are needed to compute/trigger the embedding
- Cannot generate proper cache filename or compute embedding without these details
- Performance impact is negligible (problem loading is fast compared to embedding)

### No Ancilla Distinction

As per user request:
- Embeddings are shared between forward and cyclic methods
- No separate embeddings for ancilla/non-ancilla variants
- `use_ancilla_transformation` parameter doesn't affect embedding (only affects problem conversion)

### Cache Invalidation

Embeddings are automatically invalidated by:
- **Solver change:** Different solver ID = different cache file = new embedding
- **Different instance:** Different ID/timepoints/node count = different cache file
- **Manual deletion:** Deleting files in `data/embeddings/` forces recomputation

## Testing

A test script `test_embedding_cache.py` is provided to verify:
1. ✓ Embeddings are created and cached on first run
2. ✓ Embeddings are loaded from cache on subsequent runs
3. ✓ Different instances get different embeddings
4. ✓ Both dynamics and static instances support caching

## Usage (Unchanged for Users)

Users don't need to change anything - embedding caching is automatic:

```python
runner = Runner(sampler='6.4')

# First call: Creates embedding, caches it
runner.execute_instance(instance_type='dynamics', instance_id='N_263_realization_1')

# Second call: Loads from cache (fast)
runner.execute_instance(instance_type='dynamics', instance_id='N_263_realization_1')

# Cyclic annealing uses same cached embedding
runner.execute_cyclic_annealing(instance_type='dynamics', instance_id='N_263_realization_1')
```

## Future Improvements

Potential enhancements (not implemented):
- Cache statistics (hit rate, storage size)
- Cache versioning (invalidate old embeddings)
- Parallel embedding computation
- Alternative cache backends (database, cloud storage)
