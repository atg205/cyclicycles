from cyclicycles.runner import Runner
from tqdm import tqdm
runner = Runner(sampler='1.6')


for i in tqdm(range(10)):
    for instance in [2084,5627]:
        print("_________ instance __________")
        print(instance)
        # Run cyclic annealing with 5 cycles
        response, results,cycle_energies  = runner.execute_cyclic_annealing(
            n_nodes=str(instance),  # specific instance
            num_reads=1000,  # samples per cycle
            num_cycles=10
        )

        # The cycle_energies list shows the progression of minimum energy across cycles
        print("Energy progression:", cycle_energies)
        print("Best energy found:", results['best_energy'])

        # Run forward annealing 
        response, results  = runner.execute_instance(
            n_nodes=str(instance),  # specific instance
            num_reads=1000,  # samples per cycle
        )

        # The cycle_energies list shows the progression of minimum energy across cycles
        print("Best energy found (forward annealing):", results['energies'][0])