#!/usr/bin/env python3
"""
MLflow setup script for rain prediction experiments.

This script helps set up MLflow tracking for the rain prediction model experiments.
It creates the necessary experiment and provides instructions for viewing results.
"""

import mlflow
import mlflow.pytorch
import os


def setup_mlflow():
    """Set up MLflow experiment for rain prediction."""
    
    # Set MLflow tracking URI (default is local file store)
    if not os.getenv('MLFLOW_TRACKING_URI'):
        mlflow.set_tracking_uri("file:./mlruns")
        print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
    
    # Create or set experiment
    experiment_name = "rain-prediction-experiments"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
        
        mlflow.set_experiment(experiment_name)
        
    except Exception as e:
        print(f"Error setting up experiment: {e}")
        return False
    
    return True


def print_mlflow_instructions():
    """Print instructions for using MLflow."""
    
    print("\n" + "="*60)
    print("MLflow Setup Complete!")
    print("="*60)
    
    print("\n📊 To start the MLflow UI and view your experiments:")
    print("   mlflow ui")
    print("   Then open: http://localhost:5000")
    
    print("\n🚀 To run training with MLflow tracking:")
    print("   python train.py --model_type medium --epochs 50")
    
    print("\n🔧 To run training without MLflow tracking:")
    print("   python train.py --model_type medium --epochs 50 --no_mlflow")
    
    print("\n📈 MLflow will automatically track:")
    print("   • All hyperparameters and model configuration")
    print("   • Training and validation metrics per epoch")
    print("   • Final test metrics")
    print("   • The trained PyTorch model")
    print("   • Model configuration files")
    
    print("\n💾 Data location:")
    print(f"   MLflow data: {mlflow.get_tracking_uri()}")
    
    print("\n" + "="*60)


def test_mlflow_connection():
    """Test MLflow connection with a dummy run."""
    
    print("\n🧪 Testing MLflow connection...")
    
    try:
        with mlflow.start_run(run_name="test_connection"):
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.95)
            
            run_id = mlflow.active_run().info.run_id
            print(f"✅ Test run successful! Run ID: {run_id}")
            
        return True
        
    except Exception as e:
        print(f"❌ Test run failed: {e}")
        return False


def main():
    """Main setup function."""
    
    print("🔧 Setting up MLflow for Rain Prediction Experiments")
    print("-" * 50)
    
    # Setup MLflow
    if not setup_mlflow():
        print("❌ Failed to set up MLflow")
        return
    
    # Test connection
    if not test_mlflow_connection():
        print("❌ Failed to test MLflow connection")
        return
    
    # Print instructions
    print_mlflow_instructions()


if __name__ == "__main__":
    main() 