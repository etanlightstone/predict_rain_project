import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.pytorch

from model import RainPredictionModel, create_model, MODEL_CONFIGS


class WeatherDataset:
    """Weather dataset loader and preprocessor."""
    
    def __init__(self, data_path: str = '/mnt/data/predict-rain/weather_data.csv'):
        self.data_path = data_path
        self.scaler = StandardScaler()
        
    def load_and_preprocess(self, test_size: float = 0.2, val_size: float = 0.1, random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load data and create train/validation/test dataloaders.
        
        Args:
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of training data for validation
            random_seed (int): Random seed for reproducibility
            
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: train, validation, test dataloaders
        """
        # Load data
        print(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Separate features and target
        feature_cols = ['Temperature', 'Humidity', 'WindSpeed', 'Pressure', 'CloudCover', 'UVIndex']
        X = df[feature_cols].values
        y = df['RainToday'].values
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target distribution: {np.bincount(y)} (class 0: {np.mean(y == 0):.3f}, class 1: {np.mean(y == 1):.3f})")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)  # Add dimension for binary classification
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split dataset
        total_size = len(dataset)
        test_size_int = int(test_size * total_size)
        train_val_size = total_size - test_size_int
        val_size_int = int(val_size * train_val_size)
        train_size_int = train_val_size - val_size_int
        
        torch.manual_seed(random_seed)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size_int, val_size_int, test_size_int]
        )
        
        print(f"Data splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoader objects."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Store predictions and targets for metrics
        predictions = torch.sigmoid(output).cpu().detach().numpy()
        all_predictions.extend(predictions.flatten())
        all_targets.extend(target.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics
    predictions_binary = (np.array(all_predictions) > 0.5).astype(int)
    accuracy = accuracy_score(all_targets, predictions_binary)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
    }


def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                  device: torch.device) -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Store predictions and targets for metrics
            predictions = torch.sigmoid(output).cpu().numpy()
            all_predictions.extend(predictions.flatten())
            all_targets.extend(target.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate comprehensive metrics
    predictions_binary = (np.array(all_predictions) > 0.5).astype(int)
    accuracy = accuracy_score(all_targets, predictions_binary)
    precision = precision_score(all_targets, predictions_binary, zero_division=0)
    recall = recall_score(all_targets, predictions_binary, zero_division=0)
    f1 = f1_score(all_targets, predictions_binary, zero_division=0)
    
    try:
        auc = roc_auc_score(all_targets, all_predictions)
    except:
        auc = 0.0  # In case of single class in validation set
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }


def train_model(config: Dict) -> Dict:
    """
    Main training function.
    
    Args:
        config (Dict): Training configuration containing all hyperparameters
        
    Returns:
        Dict: Training results and metrics
    """
    print("=== Starting Training ===")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Start MLflow run
    with mlflow.start_run():
        # Set experiment name and tags
        if config.get('use_mlflow', True):
            experiment_name = f"rain-prediction-{config['model_type']}"
            mlflow.set_tag("model_type", config['model_type'])
            mlflow.set_tag("timestamp", config['timestamp'])
            
            # Log all configuration parameters
            mlflow.log_params({
                "model_type": config['model_type'],
                "learning_rate": config['learning_rate'],
                "batch_size": config['batch_size'],
                "epochs": config['epochs'],
                "optimizer": config['optimizer'],
                "weight_decay": config['weight_decay'],
                "early_stopping": config['early_stopping'],
                "use_scheduler": config.get('use_scheduler', False),
                "random_seed": config['random_seed'],
                "test_size": config['test_size'],
                "val_size": config['val_size']
            })
            
            # Log model architecture parameters
            model_config = config['model_config']
            mlflow.log_params({
                "hidden_dims": str(model_config['hidden_dims']),
                "dropout_rate": model_config['dropout_rate'],
                "activation": model_config['activation'],
                "use_batch_norm": model_config['use_batch_norm'],
                "input_dim": model_config['input_dim'],
                "output_dim": model_config['output_dim']
            })
        
        # Set random seeds for reproducibility
        torch.manual_seed(config['random_seed'])
        np.random.seed(config['random_seed'])
        
        # Device selection
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        if config.get('use_mlflow', True):
            mlflow.log_param("device", str(device))
        
        # Load and preprocess data
        dataset_loader = WeatherDataset(config['data_path'])
        train_dataset, val_dataset, test_dataset = dataset_loader.load_and_preprocess(
            test_size=config['test_size'],
            val_size=config['val_size'],
            random_seed=config['random_seed']
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset, config['batch_size']
        )
        
        # Create model
        model_config = config['model_config'].copy()
        model = create_model(model_config).to(device)
        print(f"Model created with {model.count_parameters():,} parameters")
        
        if config.get('use_mlflow', True):
            mlflow.log_param("model_parameters", model.count_parameters())
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        
        if config['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                                  weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], 
                                 momentum=config.get('momentum', 0.9), 
                                 weight_decay=config['weight_decay'])
        else:
            raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
        
        # Learning rate scheduler
        if config.get('use_scheduler', False):
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        train_history = {'loss': [], 'accuracy': []}
        val_history = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'auc': []}
        
        print(f"\n=== Training for {config['epochs']} epochs ===")
        
        for epoch in range(config['epochs']):
            start_time = time.time()
            
            # Train
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_metrics = evaluate_model(model, val_loader, criterion, device)
            
            # Update learning rate scheduler
            if config.get('use_scheduler', False):
                scheduler.step(val_metrics['loss'])
            
            # Store history
            train_history['loss'].append(train_metrics['loss'])
            train_history['accuracy'].append(train_metrics['accuracy'])
            
            for key in val_history.keys():
                val_history[key].append(val_metrics[key])
            
            # Log metrics to MLflow
            if config.get('use_mlflow', True):
                mlflow.log_metrics({
                    "train_loss": train_metrics['loss'],
                    "train_accuracy": train_metrics['accuracy'],
                    "val_loss": val_metrics['loss'],
                    "val_accuracy": val_metrics['accuracy'],
                    "val_precision": val_metrics['precision'],
                    "val_recall": val_metrics['recall'],
                    "val_f1_score": val_metrics['f1_score'],
                    "val_auc": val_metrics['auc'],
                    "learning_rate": optimizer.param_groups[0]['lr']
                }, step=epoch)
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            epoch_time = time.time() - start_time
            
            # Print progress
            if epoch % config.get('print_every', 10) == 0 or epoch == config['epochs'] - 1:
                print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Train Acc: {train_metrics['accuracy']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val Acc: {val_metrics['accuracy']:.4f} | "
                      f"Val F1: {val_metrics['f1_score']:.4f} | "
                      f"Time: {epoch_time:.2f}s")
            
            # Early stopping check
            if config.get('early_stopping', False) and patience_counter >= config.get('patience', 20):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model for final evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation on test set
        print("\n=== Final Evaluation ===")
        test_metrics = evaluate_model(model, test_loader, criterion, device)
        
        print("Test Set Results:")
        for metric, value in test_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        # Log final test metrics to MLflow
        if config.get('use_mlflow', True):
            mlflow.log_metrics({
                "test_loss": test_metrics['loss'],
                "test_accuracy": test_metrics['accuracy'],
                "test_precision": test_metrics['precision'],
                "test_recall": test_metrics['recall'],
                "test_f1_score": test_metrics['f1_score'],
                "test_auc": test_metrics['auc'],
                "best_val_loss": best_val_loss
            })
            
            # Save and log the trained model
            model_path = "rain_prediction_model"
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=model_path,
                conda_env={
                    'channels': ['defaults', 'conda-forge', 'pytorch'],
                    'dependencies': [
                        'python=3.8',
                        'pytorch>=2.0.0',
                        'pandas>=2.0.0',
                        'numpy>=1.24.0',
                        'scikit-learn>=1.3.0',
                        {'pip': ['mlflow>=2.8.0']}
                    ],
                    'name': 'rain_prediction_env'
                }
            )
            
            # Log model configuration as artifact
            config_path = "model_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            mlflow.log_artifact(config_path)
            
            # Clean up temporary file
            if os.path.exists(config_path):
                os.remove(config_path)
            
            # Log the MLflow run URL
            run_id = mlflow.active_run().info.run_id
            experiment_id = mlflow.active_run().info.experiment_id
            mlflow_url = f"http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}"
            print(f"\nMLflow Run URL: {mlflow_url}")
        
        # Prepare results
        results = {
            'config': config,
            'model_parameters': model.count_parameters(),
            'best_val_loss': best_val_loss,
            'test_metrics': test_metrics,
            'train_history': train_history,
            'val_history': val_history,
            'training_completed': True
        }
        
        if config.get('use_mlflow', True):
            results['mlflow_run_id'] = mlflow.active_run().info.run_id
            results['mlflow_experiment_id'] = mlflow.active_run().info.experiment_id
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Train Rain Prediction Model')
    
    # Model architecture
    parser.add_argument('--model_type', type=str, default='medium', 
                       choices=['small', 'medium', 'large', 'deep', 'custom'],
                       help='Predefined model architecture')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=None,
                       help='Hidden layer dimensions (for custom model)')
    parser.add_argument('--dropout_rate', type=float, default=None,
                       help='Dropout rate')
    parser.add_argument('--activation', type=str, default=None,
                       choices=['relu', 'tanh', 'leaky_relu'],
                       help='Activation function')
    parser.add_argument('--use_batch_norm', action='store_true',
                       help='Use batch normalization')
    
    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'],
                       help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD optimizer')
    
    # Training options
    parser.add_argument('--early_stopping', action='store_true',
                       help='Use early stopping')
    parser.add_argument('--patience', type=int, default=20,
                       help='Patience for early stopping')
    parser.add_argument('--use_scheduler', action='store_true',
                       help='Use learning rate scheduler')
    
    # Data options
    parser.add_argument('--data_path', type=str, default='/mnt/data/predict-rain/weather_data.csv',
                       help='Path to data file')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Validation set proportion')
    
    # Other options
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--print_every', type=int, default=10,
                       help='Print progress every N epochs')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Save results to JSON file')
    parser.add_argument('--use_mlflow', action='store_true', default=True,
                       help='Use MLflow for experiment tracking')
    parser.add_argument('--no_mlflow', action='store_true',
                       help='Disable MLflow tracking')
    
    args = parser.parse_args()
    
    # Build model configuration
    if args.model_type == 'custom':
        if args.hidden_dims is None:
            raise ValueError("Must specify --hidden_dims for custom model")
        model_config = {
            'input_dim': 6,
            'hidden_dims': args.hidden_dims,
            'dropout_rate': args.dropout_rate if args.dropout_rate is not None else 0.3,
            'activation': args.activation if args.activation is not None else 'relu',
            'use_batch_norm': args.use_batch_norm,
            'output_dim': 1
        }
    else:
        model_config = MODEL_CONFIGS[args.model_type].copy()
        # Override with command line arguments if provided
        if args.dropout_rate is not None:
            model_config['dropout_rate'] = args.dropout_rate
        if args.activation is not None:
            model_config['activation'] = args.activation
        if args.use_batch_norm:
            model_config['use_batch_norm'] = True
    
    # Build training configuration
    config = {
        'model_type': args.model_type,
        'model_config': model_config,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'optimizer': args.optimizer,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'early_stopping': args.early_stopping,
        'patience': args.patience,
        'use_scheduler': args.use_scheduler,
        'data_path': args.data_path,
        'test_size': args.test_size,
        'val_size': args.val_size,
        'random_seed': args.random_seed,
        'print_every': args.print_every,
        'use_mlflow': args.use_mlflow and not args.no_mlflow,
        'timestamp': datetime.now().isoformat()
    }
    
    # Train model
    results = train_model(config)
    
    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.save_results}")
    
    # Print final summary
    print(f"\n=== Training Summary ===")
    print(f"Model: {args.model_type} ({results['model_parameters']:,} parameters)")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Test F1-score: {results['test_metrics']['f1_score']:.4f}")
    print(f"Test AUC: {results['test_metrics']['auc']:.4f}")


if __name__ == "__main__":
    main() 