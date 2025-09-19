from cyclicycles.runner import Runner

runner = Runner(sampler='4.1')

# Run cyclic annealing with 5 cycles
response, results, cycle_energies = runner.execute_cyclic_annealing(
    n_nodes="263",  # specific instance
    num_cycles=5,   # number of cycles
    num_reads=1000  # samples per cycle
)

# The cycle_energies list shows the progression of minimum energy across cycles
print("Energy progression:", cycle_energies)
print("Best energy found:", results['best_energy'])