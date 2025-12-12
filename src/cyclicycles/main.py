from cyclicycles.runner import Runner
from tqdm import tqdm
runner = Runner(sampler='1.9')
num_cycles = 10
instance_type = 'dynamics'
instances = ["3"]

# Experiment with and without ancilla
for use_ancilla in [False]:
    print(f"\n\n========== Running with use_ancilla={use_ancilla} ==========\n")
    for instance in instances:
        for timepoints in [12]:
            for i in tqdm(range(3)):
                print("_________ instance __________")
                # Run cycli           c annealing
                response, results, cycle_energies = runner.execute_cyclic_annealing(
                    num_reads=1000,
                    num_cycles=num_cycles,
                    use_forward_init=True,
                    instance_type='dynamics',
                    instance_id=instance,
                    num_timepoints=timepoints,
                    use_ancilla_transformation=use_ancilla
                )
                print(f"Energy progression: {cycle_energies}")
                print(f"Best energy found: {results['best_energy']}")
                continue
                # Run forward annealing
                for _ in range(num_cycles):
                    response, results = runner.execute_instance(
                        instance_type="dynamics",
                        instance_id=instance,
                        num_reads=1000,
                        num_timepoints=timepoints,
                        use_ancilla_transformation=False
                    )
                    print(f"Best energy found (forward annealing): {results['energies'][0]}")