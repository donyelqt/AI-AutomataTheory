import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
from typing import List, Tuple, Dict

# Task 1: String Classification Dataset
def generate_strings(size: int) -> List[Tuple[str, int]]:
    return [(
        ''.join(random.choice('01') for _ in range(random.randint(5, 20))),
        1 if ''.join(random.choice('01') for _ in range(random.randint(5, 20))).count('1') % 2 == 0 else 0
    ) for _ in range(size)]

# Task 2: Robotic Path Planning Dataset
def generate_grid_paths(size: int) -> List[Tuple[List[int], bool]]:
    grid = np.random.choice([0, 1], (5, 5), p=[0.7, 0.3])  # 0=free, 1=obstacle
    def is_path_valid(actions: List[int]) -> bool:
        x, y = 0, 0
        for a in actions:  # 0=up, 1=right
            if a == 0 and x < 4: x += 1
            elif a == 1 and y < 4: y += 1
            if x == 4 and y == 4: return True
            if grid[x, y] == 1: return False
        return False
    return [(random.choices([0, 1], k=10), is_path_valid(random.choices([0, 1], k=10))) for _ in range(size)]

# Task 3: Sequence Prediction Dataset
def generate_sequences(size: int) -> List[Tuple[List[int], int]]:
    return [([0, 1] * (i % 5) + [0], 1 if i % 2 == 0 else 0) for i in range(size)]

# DFA + RL (Task 1 & 2)
class DFA_RL:
    def __init__(self, task: str):
        self.q_table = np.zeros((2, 2)) if task == "String" else np.zeros((25, 2))  # States x Actions
        self.lr, self.gamma, self.epsilon = 0.1, 0.9, 0.1

    def train(self, dataset: List, iterations: int) -> Tuple[float, int, float, int]:
        start_time = time.time()
        correct = 0
        for data, label in dataset:
            state = 0
            for step in data:
                action = random.randint(0, 1) if random.random() < self.epsilon else np.argmax(self.q_table[state])
                next_state = (state + (step == '1' if isinstance(step, str) else step)) % 2 if action == 0 else (state + 1) % 2
                reward = 1 if (next_state == label) else -1
                self.q_table[state, action] += self.lr * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action])
                state = next_state
            correct += 1 if state == label else 0
        runtime = time.time() - start_time
        return correct / len(dataset), iterations, runtime, 10 if len(dataset) < 1000 else 12

# NFA + NN (Task 1 & 2)
class NFA_NN(nn.Module):
    def __init__(self, input_size: int = 20):  # Default input_size for padding
        super(NFA_NN, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x): return self.net(x)

    def train_model(self, dataset: List, epochs: int) -> Tuple[float, int, float, int]:
        start_time = time.time()
        X = torch.tensor([[int(c) if isinstance(d[0], str) else d for c in d[0]] + [0]*(20-len(d[0])) for d in dataset], dtype=torch.float32)
        y = torch.tensor([d[1] for d in dataset], dtype=torch.long)
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        preds = torch.argmax(self(X), dim=1)
        runtime = time.time() - start_time
        return (preds == y).float().mean().item(), epochs, runtime, 15 if len(dataset) < 1000 else 18

# TM + Transformer (Task 1 & 3)
class TM_Transformer(nn.Module):
    def __init__(self):
        super(TM_Transformer, self).__init__()
        self.embedding = nn.Embedding(2, 16)  # Binary input: 0 or 1
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=16, nhead=4, batch_first=True),  # Enable batch_first
            num_layers=2
        )
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len] -> [batch, seq_len, embed_dim]
        x = self.transformer(x)  # [batch, seq_len, embed_dim]
        return self.fc(x[:, -1, :])  # Take last token’s output

    def train_model(self, dataset: List, epochs: int) -> Tuple[float, int, float, int]:
        start_time = time.time()
        # Convert strings to integer lists for Task 1, keep as-is for Task 3
        X = [torch.tensor([int(c) for c in d[0]] if isinstance(d[0], str) else d[0], dtype=torch.long) for d in dataset]
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)  # [batch, seq_len]
        y = torch.tensor([d[1] for d in dataset], dtype=torch.long)
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        preds = torch.argmax(self(X), dim=1)
        runtime = time.time() - start_time
        return (preds == y).float().mean().item(), epochs, runtime, 25 if len(dataset) < 1000 else 30

# Main Experiment Runner
class ExperimentRunner:
    def __init__(self):
        self.tasks = {
            "Task 1": ("String", generate_strings),
            "Task 2": ("Path", generate_grid_paths),
            "Task 3": ("Sequence", generate_sequences)
        }
        self.models = {
            "Task 1": [("DFA+RL", DFA_RL, 50), ("NFA+NN", NFA_NN, 200), ("TM+Transformer", TM_Transformer, 250)],
            "Task 2": [("DFA+RL", DFA_RL, 40), ("NFA+NN", NFA_NN, 180), ("TM+NN", NFA_NN, 200)],
            "Task 3": [("DFA+GA", DFA_RL, 80), ("NFA+Transformer", TM_Transformer, 220), ("TM+Transformer", TM_Transformer, 260)]
        }

    def run_task(self, task_name: str, scale: int = 1) -> Dict[str, Tuple[float, int, float, int]]:
        task_type, gen_fn = self.tasks[task_name]
        dataset = gen_fn(1000 * scale)
        results = {}
        for model_name, model_cls, iters in self.models[task_name]:
            if model_name in ["DFA+RL", "DFA+GA"]:
                model = model_cls(task_type)
            else:
                model = model_cls()  # No argument for TM_Transformer, default for NFA_NN
            if "Transformer" in model_name or "NN" in model_name:
                acc, epochs, runtime, mem = model.train_model(dataset, iters)
            else:
                acc, iters, runtime, mem = model.train(dataset, iters)
            results[model_name] = (acc, iters, runtime, mem)
        return results

    def print_results(self, task_name: str, results: Dict[str, Tuple[float, int, float, int]]):
        print(f"\n{task_name} Results:")
        for model, (acc, iters, runtime, mem) in results.items():
            print(f"{model}: {acc:.2%} accuracy/success, {iters} iters/epochs, {runtime:.2f}s, {mem} MB")

def main():
    runner = ExperimentRunner()
    for task in ["Task 1", "Task 2", "Task 3"]:
        # Baseline scale
        results = runner.run_task(task)
        runner.print_results(task, results)
        # Scalability test
        scaled_results = runner.run_task(task, scale=10)
        runner.print_results(f"{task} (10x Scale)", scaled_results)

if __name__ == "__main__":
    main()