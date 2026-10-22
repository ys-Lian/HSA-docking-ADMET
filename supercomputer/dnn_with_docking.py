#!/usr/bin/env python3
"""DNN - with docking features"""
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = 'cpu'
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

DATA_PATH   = '/home/Scratch/hsa_docking/ml/open_admet_HPC_with_docking_final1.csv'
OUTPUT_PATH = '/home/Scratch/hsa_docking/ml/results/ADMET_DNN_with_docking_results.xlsx'

TARGET_COLS     = ['MLM CLint', 'HLM CLint', 'KSOL', 'MPPB']
ALL_TARGET_COLS = ['LogD', 'KSOL', 'HLM CLint', 'RLM CLint', 'MLM CLint',
                   'Caco-2 Permeability Papp A_B', 'Caco-2 Permeability Efflux',
                   'MPPB', 'MBPB', 'MGMB']
LOG_COLS     = ['MLM CLint', 'HLM CLint', 'KSOL']
LOG_PPB_COLS = ['MPPB']

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.replace(r'[\[\]<>,]', '_', regex=True)
for col in TARGET_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    if col in LOG_COLS:
        df[col] = np.log10(df[col].clip(lower=0.01))
    elif col in LOG_PPB_COLS:
        df[col] = np.log10((100 - df[col]).clip(lower=0.01))
    df[col] = df[col].fillna(df[col].mean())

feature_cols = [c for c in df.columns if c not in ['Name', 'SMILES'] + ALL_TARGET_COLS]
X_all = df[feature_cols].values.astype(np.float32)
Y_all = df[TARGET_COLS].values.astype(np.float32)
imputer = SimpleImputer(strategy='mean')
X_all = imputer.fit_transform(X_all).astype(np.float32)
n_targets   = len(TARGET_COLS)
n_features  = X_all.shape[1]

def to_linear(col, vals):
    if col in LOG_COLS:       return 10 ** vals
    elif col in LOG_PPB_COLS: return 100 - 10 ** vals
    return vals

ACTIVATIONS = {'ReLU': nn.ReLU(), 'Sigmoid': nn.Sigmoid(), 'Tanh': nn.Tanh()}

class DNN(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers, units, activation):
        super().__init__()
        layers, d = [], in_dim
        act = ACTIVATIONS[activation]
        for i in range(n_layers):
            layers += [nn.Linear(d, units[i]), act]
            d = units[i]
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train_dnn(model, X_tr, Y_tr, lr, batch_size, optimizer_name, weight_decay, epochs=100):
    ds     = TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    optim_cls = {'Adam': torch.optim.Adam, 'RMSprop': torch.optim.RMSprop,
                 'SGD': torch.optim.SGD}[optimizer_name]
    optimizer = optim_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            optimizer.step()

def objective(trial, X_tr, Y_tr):
    kf = KFold(n_splits=5, shuffle=True, random_state=trial.number)
    val_maes = []
    lr           = trial.suggest_float('lr', 1e-4, 0.1, log=True)
    n_layers     = trial.suggest_int('n_layers', 1, 4)
    activation   = trial.suggest_categorical('activation', ['ReLU', 'Sigmoid', 'Tanh'])
    batch_size   = trial.suggest_categorical('batch_size', [16, 32, 64])
    optimizer    = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    units = [trial.suggest_categorical(f'n_units_layer_{k+1}', [16, 32, 64, 128, 256])
             for k in range(n_layers)]

    for train_idx, val_idx in kf.split(X_tr):
        X_t, X_v = X_tr[train_idx], X_tr[val_idx]
        Y_t, Y_v = Y_tr[train_idx], Y_tr[val_idx]
        x_sc = StandardScaler().fit(X_t)
        model = DNN(n_features, n_targets, n_layers, units, activation).to(DEVICE)
        train_dnn(model, x_sc.transform(X_t).astype(np.float32), Y_t,
                  lr, batch_size, optimizer, weight_decay)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(x_sc.transform(X_v).astype(np.float32)).to(DEVICE)).cpu().numpy()
        val_maes.append(mean_absolute_error(Y_v, pred))
    return np.mean(val_maes)

all_results = []
for i in range(10):
    print(f"\n── Run {i+1}/10 ──")
    random.seed(i); np.random.seed(i); torch.manual_seed(i)
    idx = np.random.permutation(len(X_all))
    n_test = int(len(idx) * 0.20)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    Y_train, Y_test = Y_all[train_idx], Y_all[test_idx]

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, Y_train),
                   n_trials=50, show_progress_bar=False)
    best = study.best_params
    print(f"  Best params: {best}  |  inner MAE: {study.best_value:.4f}")

    n_layers   = best['n_layers']
    units      = [best[f'n_units_layer_{k+1}'] for k in range(n_layers)]
    activation = best['activation']
    x_scaler_final = StandardScaler().fit(X_train)
    X_train_s = x_scaler_final.transform(X_train).astype(np.float32)
    X_test_s  = x_scaler_final.transform(X_test).astype(np.float32)

    final_model = DNN(n_features, n_targets, n_layers, units, activation).to(DEVICE)
    train_dnn(final_model, X_train_s, Y_train,
              best['lr'], best['batch_size'], best['optimizer'], best['weight_decay'], epochs=200)
    final_model.eval()
    with torch.no_grad():
        Y_pred = final_model(torch.tensor(X_test_s).to(DEVICE)).cpu().numpy()

    per_mae    = {TARGET_COLS[j]: mean_absolute_error(Y_test[:, j], Y_pred[:, j]) for j in range(n_targets)}
    per_mse    = {TARGET_COLS[j]: mean_squared_error(Y_test[:, j], Y_pred[:, j]) for j in range(n_targets)}
    linear_mae = {col: mean_absolute_error(to_linear(col, Y_test[:, j]), to_linear(col, Y_pred[:, j]))
                  for j, col in enumerate(TARGET_COLS)}
    print(f"  Test MAE (log): { {k: f'{v:.4f}' for k, v in per_mae.items()} }")
    print(f"  Test MAE (lin): { {k: f'{v:.4f}' for k, v in linear_mae.items()} }")

    all_results.append({
        'Run': i + 1, 'Best Params': str(best), 'inner_MAE': study.best_value,
        **{f'MAE_log_{k}': v for k, v in per_mae.items()},
        **{f'MAE_linear_{k}': v for k, v in linear_mae.items()},
        **{f'MSE_{k}': v for k, v in per_mse.items()},
        **{f'y_true_{TARGET_COLS[j]}': Y_test[:, j].tolist() for j in range(n_targets)},
        **{f'y_pred_{TARGET_COLS[j]}': Y_pred[:, j].tolist() for j in range(n_targets)},
    })

results_df = pd.DataFrame(all_results)
results_df.to_excel(OUTPUT_PATH, index=False)
print(f"\nResults saved to {OUTPUT_PATH}")
print("\n── Summary: Mean ± Std MAE across 10 runs ──")
for col in TARGET_COLS:
    m_log = results_df[f'MAE_log_{col}'].mean(); s_log = results_df[f'MAE_log_{col}'].std()
    m_lin = results_df[f'MAE_linear_{col}'].mean(); s_lin = results_df[f'MAE_linear_{col}'].std()
    unit = 'log10(mL/min/kg)' if col in LOG_COLS else 'log10(%bound)'
    print(f"  {col}: {m_log:.4f} ± {s_log:.4f} [{unit}]  |  {m_lin:.4f} ± {s_lin:.4f} [original]")
