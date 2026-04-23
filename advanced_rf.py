"""
Advanced Random Forest + Gradient Boosting
- Hyperparameter tuning
- Feature importance analysis
- Model serialization
- Cross-validation
"""

import os

import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
from pathlib import Path
import time


def extract_advanced_features(mel_spec: np.ndarray) -> np.ndarray:
    """
    Ulepszona ekstrakcja handcrafted features z mel-spektrogramu.
    
    More sophisticated feature engineering:
    - Statistics per frequency band
    - Temporal dynamics
    - Spectral characteristics
    - Energy distribution
    """
    features = []
    
    # 1. Per-frequency statistics (128 mels)
    mean_energy = mel_spec.mean(axis=1)
    std_energy = mel_spec.std(axis=1)
    max_energy = mel_spec.max(axis=1)
    min_energy = mel_spec.min(axis=1)
    
    features.extend(mean_energy)  # 128
    features.extend(std_energy)   # 128
    features.extend(max_energy)   # 128
    features.extend(min_energy)   # 128
    
    # 2. Temporal features (over time)
    energy_over_time = mel_spec.mean(axis=0)  # (n_frames,)
    
    features.append(energy_over_time.mean())  # Mean energy
    features.append(energy_over_time.std())   # Std energy
    features.append(energy_over_time.max())   # Peak energy
    features.append(np.percentile(energy_over_time, 25))  # Q1
    features.append(np.percentile(energy_over_time, 50))  # Median
    features.append(np.percentile(energy_over_time, 75))  # Q3
    
    # 3. Spectral characteristics
    n_mels = mel_spec.shape[0]
    freqs = np.linspace(0, 1, n_mels)
    
    # Spectral centroid
    total_energy = mel_spec.sum()
    if total_energy > 0:
        spectral_centroid = np.sum(mel_spec * freqs[:, np.newaxis]) / total_energy
    else:
        spectral_centroid = 0.5
    features.append(spectral_centroid)
    
    # Spectral spread
    if total_energy > 0:
        spectral_spread = np.sqrt(
            np.sum(mel_spec * (freqs[:, np.newaxis] - spectral_centroid)**2) / total_energy
        )
    else:
        spectral_spread = 0
    features.append(spectral_spread)
    
    # Low/Mid/High energy ratio
    low_energy = mel_spec[:n_mels//3].sum()
    mid_energy = mel_spec[n_mels//3:2*n_mels//3].sum()
    high_energy = mel_spec[2*n_mels//3:].sum()
    
    total = low_energy + mid_energy + high_energy
    if total > 0:
        features.extend([low_energy/total, mid_energy/total, high_energy/total])
    else:
        features.extend([0, 0, 0])
    
    # 4. Zero crossing rate (approximated)
    sign_changes = np.sum(np.abs(np.diff(np.sign(energy_over_time))))
    features.append(sign_changes / len(energy_over_time))
    
    # 5. Overall statistics
    features.append(mel_spec.mean())  # Global mean
    features.append(mel_spec.std())   # Global std
    features.append(mel_spec.max())   # Global max
    features.append(mel_spec.min())   # Global min
    
    return np.array(features, dtype=np.float32)


class AdvancedRandomForest:
    """
    Tuned Random Forest z feature importance i cross-validation.
    """
    
    def __init__(self, n_estimators=500, max_depth=20, min_samples_leaf=5):
        self.rf_valence = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        self.rf_arousal = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def extract_all_features(self, mel_specs):
        """Extract features dla całego datasetu."""
        print(f"Extracting features from {len(mel_specs)} samples...")
        features = []
        
        for i, mel_spec in enumerate(mel_specs):
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{len(mel_specs)}")
            
            feat = extract_advanced_features(mel_spec)
            features.append(feat)
        
        X = np.array(features, dtype=np.float32)
        
        # Set feature names (for importance)
        if self.feature_names is None:
            n_mels = 128
            self.feature_names = []
            # Per-mel features
            for stat in ['mean', 'std', 'max', 'min']:
                for i in range(n_mels):
                    self.feature_names.append(f'mel_{i}_{stat}')
            # Temporal features
            self.feature_names.extend([
                'energy_mean', 'energy_std', 'energy_max',
                'energy_q25', 'energy_q50', 'energy_q75'
            ])
            # Spectral
            self.feature_names.extend([
                'spectral_centroid', 'spectral_spread',
                'low_ratio', 'mid_ratio', 'high_ratio'
            ])
            self.feature_names.append('zcr')
            self.feature_names.extend([
                'global_mean', 'global_std', 'global_max', 'global_min'
            ])
        
        print(f"✓ Extracted {X.shape[1]} features\n")
        return X
    
    def fit(self, train_specs, train_valences, train_arousals, 
            cv_folds=5, tune_hyperparams=False):
        """Trenuj modele z opcionalnym tuningiem."""
        
        # Feature extraction
        X = self.extract_all_features(train_specs)
        X_scaled = self.scaler.fit_transform(X)
        
        y_val = train_valences
        y_arou = train_arousals
        
        # Hyperparameter tuning (optional)
        if tune_hyperparams:
            print("Tuning hyperparameters...")
            param_grid = {
                'n_estimators': [200, 500],
                'max_depth': [15, 20, 25],
                'min_samples_leaf': [2, 5]
            }
            
            print("\n  Tuning Valence model...")
            gs_val = GridSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                param_grid,
                cv=3,
                verbose=1
            )
            gs_val.fit(X_scaled, y_val)
            print(f"  Best params (Valence): {gs_val.best_params_}")
            self.rf_valence = gs_val.best_estimator_
            
            print("\n  Tuning Arousal model...")
            gs_arou = GridSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                param_grid,
                cv=3,
                verbose=1
            )
            gs_arou.fit(X_scaled, y_arou)
            print(f"  Best params (Arousal): {gs_arou.best_params_}")
            self.rf_arousal = gs_arou.best_estimator_
        
        else:
            # Direct training
            print("Training Valence model...")
            self.rf_valence.fit(X_scaled, y_val)
            
            print("Training Arousal model...")
            self.rf_arousal.fit(X_scaled, y_arou)
        
        # Cross-validation scores
        print("\nCross-validation (5-fold):")
        
        cv_scores_val = cross_val_score(
            self.rf_valence, X_scaled, y_val, cv=cv_folds, scoring='r2'
        )
        print(f"  Valence R² (CV): {cv_scores_val.mean():.4f} (+/- {cv_scores_val.std():.4f})")
        
        cv_scores_arou = cross_val_score(
            self.rf_arousal, X_scaled, y_arou, cv=cv_folds, scoring='r2'
        )
        print(f"  Arousal R² (CV): {cv_scores_arou.mean():.4f} (+/- {cv_scores_arou.std():.4f})")
        
        # Training scores
        print("\nTraining scores:")
        print(f"  Valence R²: {self.rf_valence.score(X_scaled, y_val):.4f}")
        print(f"  Arousal R²: {self.rf_arousal.score(X_scaled, y_arou):.4f}")
        
        return X, X_scaled
    
    def predict(self, test_specs):
        """Predykcja na test secie."""
        X = self.extract_all_features(test_specs)
        X_scaled = self.scaler.transform(X)
        
        valences = self.rf_valence.predict(X_scaled)
        arousals = self.rf_arousal.predict(X_scaled)
        
        return valences, arousals
    
    def evaluate(self, test_specs, test_valences, test_arousals):
        """Ewaluacja."""
        val_pred, arou_pred = self.predict(test_specs)
        
        metrics = {
            'valence_mae': mean_absolute_error(test_valences, val_pred),
            'arousal_mae': mean_absolute_error(test_arousals, arou_pred),
            'valence_rmse': np.sqrt(mean_squared_error(test_valences, val_pred)),
            'arousal_rmse': np.sqrt(mean_squared_error(test_arousals, arou_pred)),
            'valence_r2': r2_score(test_valences, val_pred),
            'arousal_r2': r2_score(test_arousals, arou_pred),
        }
        
        return metrics, val_pred, arou_pred
    
    def save(self, filepath='rf_model.pkl'):
        """Zapisz model."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'rf_valence': self.rf_valence,
                'rf_arousal': self.rf_arousal,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        print(f"✓ Model saved: {filepath}")
    
    def plot_feature_importance(self, top_n=20, save_path='rf_feature_importance.png'):
        """Plot feature importance."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Valence
        importances_val = self.rf_valence.feature_importances_
        top_idx_val = np.argsort(importances_val)[-top_n:]
        
        axes[0].barh(
            [self.feature_names[i] for i in top_idx_val],
            importances_val[top_idx_val]
        )
        axes[0].set_title(f'Top {top_n} Features - Valence')
        axes[0].set_xlabel('Importance')
        
        # Arousal
        importances_arou = self.rf_arousal.feature_importances_
        top_idx_arou = np.argsort(importances_arou)[-top_n:]
        
        axes[1].barh(
            [self.feature_names[i] for i in top_idx_arou],
            importances_arou[top_idx_arou]
        )
        axes[1].set_title(f'Top {top_n} Features - Arousal')
        axes[1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot: {save_path}")
        plt.close()


class GradientBoostingBaseline:
    """
    Gradient Boosting (zwykle lepszy niż RF ale wolniejszy do treningu).
    """
    
    def __init__(self, n_estimators=500, learning_rate=0.1, max_depth=5):
        self.gb_valence = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            verbose=1
        )
        
        self.gb_arousal = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            verbose=1
        )
        
        self.scaler = StandardScaler()
    
    def extract_all_features(self, mel_specs):
        """Użyj tej samej feature extraction co RF."""
        features = []
        for mel_spec in mel_specs:
            feat = extract_advanced_features(mel_spec)
            features.append(feat)
        return np.array(features, dtype=np.float32)
    
    def fit(self, train_specs, train_valences, train_arousals):
        print("Extracting features...")
        X = self.extract_all_features(train_specs)
        X_scaled = self.scaler.fit_transform(X)
        
        print("\nTraining Gradient Boosting models...")
        self.gb_valence.fit(X_scaled, train_valences)
        self.gb_arousal.fit(X_scaled, train_arousals)
        
        print(f"  Valence R²: {self.gb_valence.score(X_scaled, train_valences):.4f}")
        print(f"  Arousal R²: {self.gb_arousal.score(X_scaled, train_arousals):.4f}")
    
    def predict(self, test_specs):
        X = self.extract_all_features(test_specs)
        X_scaled = self.scaler.transform(X)
        
        valences = self.gb_valence.predict(X_scaled)
        arousals = self.gb_arousal.predict(X_scaled)
        
        return valences, arousals
    
    def evaluate(self, test_specs, test_valences, test_arousals):
        val_pred, arou_pred = self.predict(test_specs)
        
        metrics = {
            'valence_mae': mean_absolute_error(test_valences, val_pred),
            'arousal_mae': mean_absolute_error(test_arousals, arou_pred),
            'valence_r2': r2_score(test_valences, val_pred),
            'arousal_r2': r2_score(test_arousals, arou_pred),
        }
        
        return metrics
    
    def save(self, filepath='gb_model.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'gb_valence': self.gb_valence,
                'gb_arousal': self.gb_arousal,
                'scaler': self.scaler
            }, f)
        print(f"✓ Model saved: {filepath}")


def main(data_dir='data/processed'):
    """Main execution."""
    
    os.makedirs('checkpoints', exist_ok=True)
    
    print("="*70)
    print("ADVANCED RANDOM FOREST + GRADIENT BOOSTING")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    import pickle as pkl
    
    train_data = np.load(f'{data_dir}/train_dataset.npz')
    with open(f'{data_dir}/train_mel_specs.pkl', 'rb') as f:
        train_mel_specs = pkl.load(f)
    
    test_data = np.load(f'{data_dir}/test_dataset.npz')
    with open(f'{data_dir}/test_mel_specs.pkl', 'rb') as f:
        test_mel_specs = pkl.load(f)
    
    print(f"✓ Train: {len(train_mel_specs)} samples")
    print(f"✓ Test: {len(test_mel_specs)} samples\n")
    
    # Advanced Random Forest
    print("="*70)
    print("ADVANCED RANDOM FOREST")
    print("="*70 + "\n")
    
    start = time.time()
    
    rf = AdvancedRandomForest(n_estimators=500, max_depth=20)
    X_train, X_train_scaled = rf.fit(
        train_mel_specs,
        train_data['valence'],
        train_data['arousal'],
        cv_folds=5,
        tune_hyperparams=False  # Set to True for grid search
    )
    
    # Evaluate
    test_metrics_rf, pred_val_rf, pred_arou_rf = rf.evaluate(
        test_mel_specs,
        test_data['valence'],
        test_data['arousal']
    )
    
    elapsed_rf = time.time() - start
    
    print(f"\n✓ RF Results:")
    print(f"  Valence: MAE={test_metrics_rf['valence_mae']:.4f}, R²={test_metrics_rf['valence_r2']:.4f}")
    print(f"  Arousal: MAE={test_metrics_rf['arousal_mae']:.4f}, R²={test_metrics_rf['arousal_r2']:.4f}")
    print(f"  Training time: {elapsed_rf:.1f}s\n")
    
    # Save RF
    rf.save('checkpoints/rf_advanced_best.pkl')
    rf.plot_feature_importance(top_n=20, save_path='rf_feature_importance.png')
    
    # Gradient Boosting
    print("\n" + "="*70)
    print("GRADIENT BOOSTING")
    print("="*70 + "\n")
    
    start = time.time()
    
    gb = GradientBoostingBaseline(n_estimators=500, learning_rate=0.1, max_depth=5)
    gb.fit(train_mel_specs, train_data['valence'], train_data['arousal'])
    
    test_metrics_gb = gb.evaluate(
        test_mel_specs,
        test_data['valence'],
        test_data['arousal']
    )
    
    elapsed_gb = time.time() - start
    
    print(f"\n✓ GB Results:")
    print(f"  Valence: MAE={test_metrics_gb['valence_mae']:.4f}, R²={test_metrics_gb['valence_r2']:.4f}")
    print(f"  Arousal: MAE={test_metrics_gb['arousal_mae']:.4f}, R²={test_metrics_gb['arousal_r2']:.4f}")
    print(f"  Training time: {elapsed_gb:.1f}s\n")
    
    gb.save('checkpoints/gb_best.pkl')
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    summary = {
        'random_forest': {
            'valence_mae': test_metrics_rf['valence_mae'],
            'arousal_mae': test_metrics_rf['arousal_mae'],
            'valence_r2': test_metrics_rf['valence_r2'],
            'arousal_r2': test_metrics_rf['arousal_r2'],
            'training_time_s': elapsed_rf
        },
        'gradient_boosting': {
            'valence_mae': test_metrics_gb['valence_mae'],
            'arousal_mae': test_metrics_gb['arousal_mae'],
            'valence_r2': test_metrics_gb['valence_r2'],
            'arousal_r2': test_metrics_gb['arousal_r2'],
            'training_time_s': elapsed_gb
        }
    }
    
    with open('rf_gb_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✓ Results saved:")
    print("  - checkpoints/rf_advanced_best.pkl")
    print("  - checkpoints/gb_best.pkl")
    print("  - rf_feature_importance.png")
    print("  - rf_gb_results.json")


if __name__ == "__main__":
    main()