#!/usr/bin/env python3
"""XGB - with docking features"""
import numpy as np
import pandas as pd
import random
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
import shap
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

np.random.seed(42)
random.seed(42)

DATA_PATH   = '/home/Scratch/hsa_docking/ml/open_admet_HPC_with_docking_final1.csv'
OUTPUT_PATH = '/home/Scratch/hsa_docking/ml/results/ADMET_XGB_with_docking_results.xlsx'
SHAP_PREFIX = '/home/Scratch/hsa_docking/ml/results/ADMET_XGB_with_docking'

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
n_targets = len(TARGET_COLS)

def to_linear(col, vals):
    if col in LOG_COLS:    return 10 ** vals
    elif col in LOG_PPB_COLS: return 100 - 10 ** vals
    return vals

def objective(trial, X_tr, Y_tr):
    kf = KFold(n_splits=5, shuffle=True, random_state=trial.number)
    val_maes = []
    params = {
        'max_depth':        trial.suggest_int('max_depth', 3, 8),
        'n_estimators':     trial.suggest_categorical('n_estimators', [100, 300, 500, 800]),
        'learning_rate':    trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
    }
    for train_idx, val_idx in kf.split(X_tr):
        X_t, X_v = X_tr[train_idx], X_tr[val_idx]
        Y_t, Y_v = Y_tr[train_idx], Y_tr[val_idx]
        x_sc = StandardScaler().fit(X_t)
        model = MultiOutputRegressor(xgb.XGBRegressor(**params), n_jobs=1)
        model.fit(x_sc.transform(X_t), Y_t)
        val_maes.append(mean_absolute_error(Y_v, model.predict(x_sc.transform(X_v))))
    return np.mean(val_maes)

all_results = []
best_run_info = {'inner_MAE': float('inf'), 'model': None, 'X_train_s': None, 'train_names': None}
for i in range(10):
    print(f"\n── Run {i+1}/10 ──")
    random.seed(i); np.random.seed(i)
    idx = np.random.permutation(len(X_all))
    n_test = int(len(idx) * 0.20)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    Y_train, Y_test = Y_all[train_idx], Y_all[test_idx]

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, Y_train),
                   n_trials=50, show_progress_bar=False)
    best_params = study.best_params
    best_params.update({'random_state': 42, 'n_jobs': -1, 'verbosity': 0})
    print(f"  Best params: {best_params}  |  inner MAE: {study.best_value:.4f}")

    x_scaler_final = StandardScaler().fit(X_train)
    X_train_s = x_scaler_final.transform(X_train)
    X_test_s  = x_scaler_final.transform(X_test)
    final_model = MultiOutputRegressor(xgb.XGBRegressor(**best_params), n_jobs=1)
    final_model.fit(X_train_s, Y_train)
    if study.best_value < best_run_info['inner_MAE']:
        best_run_info['inner_MAE']   = study.best_value
        best_run_info['model']       = final_model
        best_run_info['X_train_s']   = X_train_s
        best_run_info['train_names'] = df['Name'].values[train_idx]
    Y_pred = final_model.predict(X_test_s)

    per_mae   = {TARGET_COLS[j]: mean_absolute_error(Y_test[:, j], Y_pred[:, j]) for j in range(n_targets)}
    per_mse   = {TARGET_COLS[j]: mean_squared_error(Y_test[:, j], Y_pred[:, j]) for j in range(n_targets)}
    linear_mae = {col: mean_absolute_error(to_linear(col, Y_test[:, j]), to_linear(col, Y_pred[:, j]))
                  for j, col in enumerate(TARGET_COLS)}
    print(f"  Test MAE (log): { {k: f'{v:.4f}' for k, v in per_mae.items()} }")
    print(f"  Test MAE (lin): { {k: f'{v:.4f}' for k, v in linear_mae.items()} }")


    all_results.append({
        'Run': i + 1, 'Best Params': str(best_params), 'inner_MAE': study.best_value,
        **{f'MAE_log_{k}': v for k, v in per_mae.items()},
        **{f'MAE_linear_{k}': v for k, v in linear_mae.items()},
        **{f'MSE_{k}': v for k, v in per_mse.items()},
        **{f'y_true_{TARGET_COLS[j]}': Y_test[:, j].tolist() for j in range(n_targets)},
        **{f'y_pred_{TARGET_COLS[j]}': Y_pred[:, j].tolist() for j in range(n_targets)},
    })

results_df = pd.DataFrame(all_results)
results_df.to_excel(OUTPUT_PATH, index=False)
print(f"\nCalculating SHAP for best run (inner_MAE={best_run_info['inner_MAE']:.4f})...")
X_best_df = pd.DataFrame(best_run_info['X_train_s'], columns=feature_cols)
for j, col in enumerate(TARGET_COLS):
    explainer = shap.TreeExplainer(best_run_info['model'].estimators_[j])
    sv = explainer.shap_values(X_best_df)
    shap_out = pd.DataFrame(sv, columns=feature_cols)
    shap_out.insert(0, 'Name', best_run_info['train_names'])
    shap_out = shap_out.groupby('Name').mean()
    fname = f"{SHAP_PREFIX}_{col.replace(' ', '_')}_SHAP.csv"
    shap_out.to_csv(fname)
    print(f"Saved SHAP: {fname}")
print(f"\nResults saved to {OUTPUT_PATH}")

for col in TARGET_COLS:
    shap_out = pd.concat(all_shap_records[col], ignore_index=True)
    shap_out = shap_out.drop(columns=['Run']).groupby('Name').mean()
    fname = f"{SHAP_PREFIX}_{col.replace(' ', '_')}_SHAP.csv"
    shap_out.to_csv(fname)
    print(f"Saved SHAP: {fname}")

print("\n── Summary: Mean ± Std MAE across 10 runs ──")
for col in TARGET_COLS:
    m_log = results_df[f'MAE_log_{col}'].mean(); s_log = results_df[f'MAE_log_{col}'].std()
    m_lin = results_df[f'MAE_linear_{col}'].mean(); s_lin = results_df[f'MAE_linear_{col}'].std()
    unit = 'log10(mL/min/kg)' if col in LOG_COLS else 'log10(%bound)'
    print(f"  {col}: {m_log:.4f} ± {s_log:.4f} [{unit}]  |  {m_lin:.4f} ± {s_lin:.4f} [original]")
