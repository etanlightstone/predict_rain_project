#!/usr/bin/env python3
"""
Example experiment configurations for rain prediction model training.

This file contains various experiment setups that can be used to explore
different model architectures and hyperparameters.
"""

# Example 1: Quick baseline experiment
BASELINE_EXPERIMENT = [
    "python", "train.py",
    "--model_type", "small",
    "--epochs", "50",
    "--learning_rate", "0.001",
    "--batch_size", "32"
]

# Example 2: Medium model with early stopping
MEDIUM_EXPERIMENT = [
    "python", "train.py",
    "--model_type", "medium",
    "--epochs", "100",
    "--learning_rate", "0.001",
    "--batch_size", "64",
    "--early_stopping",
    "--patience", "15",
    "--use_scheduler"
]

# Example 3: Large model with regularization
LARGE_EXPERIMENT = [
    "python", "train.py",
    "--model_type", "large",
    "--epochs", "150",
    "--learning_rate", "0.0005",
    "--batch_size", "128",
    "--weight_decay", "1e-4",
    "--dropout_rate", "0.5",
    "--early_stopping",
    "--use_scheduler"
]

# Example 4: Custom architecture experiment
CUSTOM_EXPERIMENT = [
    "python", "train.py",
    "--model_type", "custom",
    "--hidden_dims", "96", "48", "24",
    "--epochs", "100",
    "--learning_rate", "0.002",
    "--batch_size", "64",
    "--activation", "leaky_relu",
    "--use_batch_norm",
    "--early_stopping"
]

# Example 5: SGD optimizer experiment
SGD_EXPERIMENT = [
    "python", "train.py",
    "--model_type", "medium",
    "--optimizer", "sgd",
    "--learning_rate", "0.01",
    "--momentum", "0.9",
    "--epochs", "150",
    "--batch_size", "64",
    "--weight_decay", "1e-4",
    "--use_scheduler"
]

# Example 6: Deep network experiment
DEEP_EXPERIMENT = [
    "python", "train.py",
    "--model_type", "deep",
    "--epochs", "200",
    "--learning_rate", "0.0008",
    "--batch_size", "64",
    "--dropout_rate", "0.3",
    "--early_stopping",
    "--patience", "25",
    "--use_scheduler"
]

# Example 7: Hyperparameter sweep experiments
HYPERPARAMETER_SWEEP = [
    # Different learning rates
    {
        "name": "lr_0001",
        "cmd": ["python", "train.py", "--model_type", "medium", "--learning_rate", "0.001", "--epochs", "100"]
    },
    {
        "name": "lr_0005",
        "cmd": ["python", "train.py", "--model_type", "medium", "--learning_rate", "0.0005", "--epochs", "100"]
    },
    {
        "name": "lr_002",
        "cmd": ["python", "train.py", "--model_type", "medium", "--learning_rate", "0.002", "--epochs", "100"]
    },
    
    # Different batch sizes
    {
        "name": "batch_32",
        "cmd": ["python", "train.py", "--model_type", "medium", "--batch_size", "32", "--epochs", "100"]
    },
    {
        "name": "batch_128",
        "cmd": ["python", "train.py", "--model_type", "medium", "--batch_size", "128", "--epochs", "100"]
    },
    
    # Different dropout rates
    {
        "name": "dropout_02",
        "cmd": ["python", "train.py", "--model_type", "medium", "--dropout_rate", "0.2", "--epochs", "100"]
    },
    {
        "name": "dropout_05",
        "cmd": ["python", "train.py", "--model_type", "medium", "--dropout_rate", "0.5", "--epochs", "100"]
    }
]


def print_experiment_commands():
    """Print all experiment commands for easy copy-paste."""
    
    experiments = [
        ("Baseline", BASELINE_EXPERIMENT),
        ("Medium", MEDIUM_EXPERIMENT),
        ("Large", LARGE_EXPERIMENT),
        ("Custom", CUSTOM_EXPERIMENT),
        ("SGD", SGD_EXPERIMENT),
        ("Deep", DEEP_EXPERIMENT)
    ]
    
    print("=== EXPERIMENT COMMANDS ===\n")
    
    for name, cmd in experiments:
        print(f"# {name} Experiment")
        print(" ".join(cmd))
        print()
    
    print("=== HYPERPARAMETER SWEEP ===\n")
    for exp in HYPERPARAMETER_SWEEP:
        print(f"# {exp['name']}")
        print(" ".join(exp['cmd']))
        print()


if __name__ == "__main__":
    print_experiment_commands() 