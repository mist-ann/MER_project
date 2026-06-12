#!/usr/bin/env python3
"""
Night Training Script
Uruchomia wszystkie eksperymenty na noc:
1. Advanced Random Forest + Gradient Boosting
2. Full LSTM training
3. Hyperparameter tuning (grid search)
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import json


class NightTraining:
    """Manager do nocnego trenowania."""
    
    def __init__(self, log_file='night_training_log.txt'):
        self.log_file = log_file
        self.start_time = datetime.now()
        self.results = {}
        
        # Prepare log
        with open(log_file, 'w') as f:
            f.write(f"Night Training Session - {self.start_time}\n")
            f.write("=" * 70 + "\n\n")
    
    def log(self, message):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] {message}"
        
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + "\n")
    
    def run_experiment(self, name, script, args=None):
        """
        Run experiment script.
        
        Args:
            name: Experiment name
            script: Script to run
            args: Command line arguments
        """
        self.log(f"\n{'='*70}")
        self.log(f"Starting: {name}")
        self.log(f"{'='*70}")
        
        start = time.time()
        
        try:
            cmd = ['python', script]
            if args:
                cmd.extend(args)
            
            self.log(f"Command: {' '.join(cmd)}\n")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Log output
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.log(f"  {line}")
            
            if result.stderr:
                for line in result.stderr.split('\n'):
                    if line.strip():
                        self.log(f"  [ERROR] {line}")
            
            elapsed = time.time() - start
            
            if result.returncode == 0:
                self.log(f"✓ {name} completed successfully ({elapsed/60:.1f} min)")
                self.results[name] = {
                    'status': 'success',
                    'time_min': elapsed / 60
                }
            else:
                self.log(f"✗ {name} failed (return code {result.returncode})")
                self.results[name] = {
                    'status': 'failed',
                    'time_min': elapsed / 60
                }
        
        except Exception as e:
            self.log(f"✗ Exception in {name}: {e}")
            self.results[name] = {
                'status': 'exception',
                'error': str(e)
            }
    
    def summary(self):
        """Print summary at end."""
        elapsed = (datetime.now() - self.start_time).total_seconds() / 60
        
        self.log(f"\n{'='*70}")
        self.log("NIGHT TRAINING SUMMARY")
        self.log(f"{'='*70}")
        self.log(f"Total time: {elapsed:.1f} minutes\n")
        
        for exp_name, result in self.results.items():
            status = result['status']
            time_str = f"{result.get('time_min', 0):.1f} min"
            self.log(f"  {exp_name}: {status} ({time_str})")
        
        # Save results
        with open('night_training_results.json', 'w') as f:
            json.dump({
                'timestamp': str(self.start_time),
                'total_time_min': elapsed,
                'experiments': self.results
            }, f, indent=2)
        
        self.log(f"\n✓ Results saved to: night_training_results.json")


def main():
    """Main execution."""
    
    trainer = NightTraining(log_file='night_training_log.txt')
    
    trainer.log("Night Training Session Starting...")
    trainer.log(f"Python version: {sys.version}")
    trainer.log(f"Working directory: {os.getcwd()}\n")
    
    # Check if data exists
    if not os.path.exists('data/processed/train_dataset.npz'):
        trainer.log("✗ ERROR: Preprocessed data not found!")
        trainer.log("  Please run: python quick_start.py")
        return
    
    # ========================================================================
    # EXPERIMENT 1: Advanced Random Forest + Gradient Boosting (fast)
    # ========================================================================
    
    # trainer.run_experiment(
    #     "Advanced Random Forest + Gradient Boosting",
    #     "advanced_rf.py",
    #     args=[]
    # )
    
    # ========================================================================
    # EXPERIMENT 2: Full LSTM Training (medium)
    # ========================================================================
    
    trainer.run_experiment(
        "LSTM Full Training (improved architecture)",
        "lstm_full_training.py",
        args=[
            '--hidden_size', '256',
            '--num_layers', '3',
            '--dropout', '0.4',
            '--lr', '1e-3',
            '--batch_size', '32',
            '--epochs', '200'
        ]
    )
    
    # ========================================================================
    # EXPERIMENT 3: Hyperparameter Tuning (long - depends on GPU)
    # ========================================================================
    
    # UWAGA: To będzie trwać! Dostosuj parametry jeśli potrzebne:
    # Jeśli masz słaby GPU, zmniejsz kombinacje:
    #   --hidden_sizes 256 512
    #   --num_layers 2 3
    #   --dropouts 0.3 0.5
    #   --lrs 5e-4 1e-3
    #   --batch_sizes 32
    
    trainer.run_experiment(
        "Hyperparameter Tuning (Grid Search)",
        "hyperparameter_tuning.py",
        args=[
            '--hidden_sizes', '128', '256',
            '--num_layers', '2',
            '--dropouts', '0.3', '0.5',
            '--lrs', '5e-4', '1e-3',
            '--batch_sizes', '32',
            '--epochs', '40',
            '--patience', '5'
        ]
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    trainer.summary()
    
    # ========================================================================
    # Notifications (optional - jeśli chcesz znać wynik na rano)
    # ========================================================================
    
    print("\n" + "="*70)
    print("All experiments completed!")
    print("="*70)
    print("\nResults files generated:")
    print("  - night_training_log.txt (full log)")
    print("  - night_training_results.json (summary)")
    print("  - rf_gb_results.json (RF vs GB comparison)")
    print("  - lstm_results.json (LSTM metrics)")
    print("  - hyperparameter_search_results.csv (grid search results)")
    print("  - best_hyperparameters.json (best params found)")
    print("\nModels saved in checkpoints/")
    print("\n✓ Night training finished at: " + datetime.now().strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()