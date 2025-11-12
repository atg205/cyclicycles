from cyclicycles.runner import Runner
from tqdm import tqdm
runner = Runner(sampler='1.7')
num_cycles = 5
instance_type = 'dynamics'
instance = "3"

for instance in ["8"]:
    for timepoints in [5]:
        for i in tqdm(range(3)):
            print("_________ instance __________")
            if True:
                # Run cyclic annealing with 5 cycles
                response, results,cycle_energies  = runner.execute_cyclic_annealing(
                    num_reads=1000,  # samples per cycle
                    num_cycles=num_cycles,
                    use_forward_init=True,
                    instance_type='dynamics',
                    instance_id = instance,
                    num_timepoints=timepoints
                )

                # The cycle_energies list shows the progression of minimum energy across cycles
                print("Energy progression:", cycle_energies)
                print("Best energy found:", results['best_energy'])

            # Run forward annealing 
            for i in range(num_cycles):
                response, results  = runner.execute_instance(
                    instance_type="dynamics",
                    instance_id= instance,
                    num_reads=1000,  # samples per cycle
                    num_timepoints=timepoints
                )

                # The cycle_energies list shows the progression of minimum energy across cycles
                print("Best energy found (forward annealing):", results['energies'][0])