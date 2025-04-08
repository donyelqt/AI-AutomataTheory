import time
import random
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

@dataclass
class ModelResult:
    accuracy: float  # or success rate
    iterations: int  # or epochs
    runtime: float  # in seconds
    memory: int     # in MB

class ModelSimulator:
    def __init__(self):
        self.models = {
            "DFA+RL": self.simulate_dfa_rl,
            "NFA+NN": self.simulate_nfa_nn,
            "PDA+MDP": self.simulate_pda_mdp,
            "Probabilistic+MDP": self.simulate_probabilistic_mdp,
            "TM+Transformer": self.simulate_tm_transformer,
            # Add more as needed for other combinations
        }
        self.baseline = {
            "DFA": 0.85, "NFA": 0.87, "PDA": 0.80, 
            "Probabilistic": 0.82, "TM": 0.90
        }

    def simulate_dfa_rl(self, task: str, scale: int = 1) -> ModelResult:
        base_accuracy = {"String": 0.92, "Path": 0.90, "Sequence": 0.88}[task]
        base_iter = {"String": 150, "Path": 120, "Sequence": 100}[task]
        base_runtime = {"String": 0.8, "Path": 1.0, "Sequence": 0.9}[task]
        base_memory = {"String": 10, "Path": 12, "Sequence": 11}[task]
        
        if scale > 1:
            return ModelResult(
                accuracy=base_accuracy - 0.05,
                iterations=base_iter,
                runtime=base_runtime * 1.1,
                memory=base_memory
            )
        return ModelResult(base_accuracy, base_iter, base_runtime, base_memory)

    def simulate_nfa_nn(self, task: str, scale: int = 1) -> ModelResult:
        base_accuracy = {"String": 0.95, "Path": 0.87, "Sequence": 0.96}[task]
        base_iter = {"String": 200, "Path": 180, "Sequence": 220}[task]
        base_runtime = {"String": 1.2, "Path": 1.4, "Sequence": 1.8}[task]
        base_memory = {"String": 15, "Path": 18, "Sequence": 20}[task]
        
        if scale > 1:
            return ModelResult(
                accuracy=base_accuracy - 0.03,
                iterations=base_iter,
                runtime=base_runtime * 1.2,
                memory=base_memory
            )
        return ModelResult(base_accuracy, base_iter, base_runtime, base_memory)

    # Placeholder for other model simulations
    def simulate_pda_mdp(self, task: str, scale: int = 1) -> ModelResult:
        base_accuracy = {"String": 0.88, "Path": 0.92, "Sequence": 0.91}[task]
        base_iter = {"String": 180, "Path": 140, "Sequence": 160}[task]
        base_runtime = {"String": 1.5, "Path": 1.6, "Sequence": 1.7}[task]
        base_memory = {"String": 20, "Path": 22, "Sequence": 23}[task]
        
        if scale > 1:
            return ModelResult(
                accuracy=base_accuracy - 0.07,
                iterations=base_iter,
                runtime=base_runtime * 1.25,
                memory=base_memory
            )
        return ModelResult(base_accuracy, base_iter, base_runtime, base_memory)

    def simulate_probabilistic_mdp(self, task: str, scale: int = 1) -> ModelResult:
        base_accuracy = {"String": 0.90, "Path": 0.94, "Sequence": 0.93}[task]
        base_iter = {"String": 160, "Path": 130, "Sequence": 140}[task]
        base_runtime = {"String": 1.0, "Path": 1.2, "Sequence": 1.3}[task]
        base_memory = {"String": 12, "Path": 15, "Sequence": 14}[task]
        
        if scale > 1:
            return ModelResult(
                accuracy=base_accuracy - 0.04,
                iterations=base_iter,
                runtime=base_runtime * 1.15,
                memory=base_memory
            )
        return ModelResult(base_accuracy, base_iter, base_runtime, base_memory)

    def simulate_tm_transformer(self, task: str, scale: int = 1) -> ModelResult:
        base_accuracy = {"String": 0.97, "Path": 0.89, "Sequence": 0.98}[task]
        base_iter = {"String": 250, "Path": 200, "Sequence": 260}[task]
        base_runtime = {"String": 2.0, "Path": 2.2, "Sequence": 2.5}[task]
        base_memory = {"String": 25, "Path": 28, "Sequence": 30}[task]
        
        if scale > 1:
            return ModelResult(
                accuracy=base_accuracy - 0.02,
                iterations=base_iter,
                runtime=base_runtime * 1.3,
                memory=base_memory
            )
        return ModelResult(base_accuracy, base_iter, base_runtime, base_memory)

    def run_task(self, task_name: str, scale: int = 1) -> Dict[str, ModelResult]:
        results = {}
        task_map = {
            "Task 1": "String",
            "Task 2": "Path",
            "Task 3": "Sequence"
        }
        task = task_map[task_name]

        for model_name, simulate_fn in self.models.items():
            start_time = time.time()
            result = simulate_fn(task, scale)
            elapsed = time.time() - start_time
            # Simulate runtime by adding a small delay if needed
            if elapsed < result.runtime:
                time.sleep(result.runtime - elapsed)
            results[model_name] = result
        
        return results

    def print_results(self, task_name: str, results: Dict[str, ModelResult]):
        print(f"\n{task_name} Results:")
        for model, result in results.items():
            print(f"{model}:")
            print(f"  Accuracy/Success: {result.accuracy:.2%}")
            print(f"  Iterations/Epochs: {result.iterations}")
            print(f"  Runtime: {result.runtime:.2f}s")
            print(f"  Memory: {result.memory} MB")

def main():
    simulator = ModelSimulator()
    
    # Run all tasks at baseline scale
    for task in ["Task 1", "Task 2", "Task 3"]:
        results = simulator.run_task(task)
        simulator.print_results(task, results)
    
    # Scalability test (10x input size)
    print("\nScalability Test (10x Input Size):")
    for task in ["Task 1", "Task 2", "Task 3"]:
        results = simulator.run_task(task, scale=10)
        simulator.print_results(task, results)

if __name__ == "__main__":
    main()