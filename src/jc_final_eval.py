"""Run the frozen 2025 JC holdout confirmation."""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.jc_baseline_models import load_weathered_2024_2025, metrics
from src.jc_baselines import naive_predictions
from src.jc_data import ROOT
from src.jc_features import make_features
from src.jc_torch_models import (
    DEVICE, LSTMForecaster, MLP, SEED, SEQUENCE_COLUMNS, _make_sequences, seed_everything,
)
from src.jc_weather import FIELDS


def _fit_fixed(model, train_inputs, train_y, predict_inputs, epochs, seed):
    model = model.to(DEVICE)
    training = tuple(torch.as_tensor(value, dtype=torch.float32) for value in train_inputs)
    loader = DataLoader(TensorDataset(*training, torch.as_tensor(train_y, dtype=torch.float32)),
                        batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for batch in loader:
            *features, target = (tensor.to(DEVICE) for tensor in batch)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(*features), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model.eval()
    inputs = tuple(torch.as_tensor(value, dtype=torch.float32, device=DEVICE)
                   for value in predict_inputs)
    with torch.no_grad():
        return model(*inputs).detach().cpu().numpy()


def run_final_confirmation():
    final_metadata_path = ROOT / 'results/metrics/jc_final_2025_metadata.json'
    if final_metadata_path.exists():
        raise FileExistsError(
            'Final 2025 confirmation already exists; refusing to rerun or overwrite it.')
    config = json.loads((ROOT / 'configs/jc_experiment.json').read_text(encoding='utf-8'))
    frame = load_weathered_2024_2025()
    x, y = make_features(frame)
    target_time = x.index + pd.Timedelta(hours=1)
    train_mask = ((target_time >= pd.Timestamp(config['final_fit_start_utc'])) &
                  (target_time < pd.Timestamp(config['final_fit_end_utc'])) & y.notna().to_numpy())
    test_mask = ((target_time >= pd.Timestamp(config['test_start_utc'])) &
                 (target_time < pd.Timestamp(config['test_end_utc'])) & y.notna().to_numpy())
    predictions = pd.DataFrame(index=x.index[test_mask])
    predictions['actual'] = y.loc[predictions.index]
    naive = naive_predictions(frame, y)
    for name in ('naive_previous_hour', 'naive_previous_week'):
        predictions[name] = naive.loc[predictions.index, name]

    base_x = x.loc[:, ~x.columns.str.startswith(tuple(FIELDS))]
    model_specs = {
        'linear_regression': (base_x, make_pipeline(SimpleImputer(strategy='median'),
            StandardScaler(), LinearRegression())),
        'random_forest': (base_x, make_pipeline(SimpleImputer(strategy='median'),
            RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1,
                                  min_samples_leaf=2, max_features=0.8))),
        'linear_regression_weather': (x, make_pipeline(SimpleImputer(strategy='median'),
            StandardScaler(), LinearRegression())),
        'random_forest_weather': (x, make_pipeline(SimpleImputer(strategy='median'),
            RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1,
                                  min_samples_leaf=2, max_features=0.8))),
    }
    for name, (features, model) in model_specs.items():
        train_valid = train_mask & features.notna().all(axis=1).to_numpy()
        test_valid = test_mask & features.notna().all(axis=1).to_numpy()
        model.fit(features.loc[train_valid], y.loc[train_valid])
        predictions.loc[features.index[test_valid], name] = model.predict(features.loc[test_valid])

    train_valid = train_mask & x.notna().all(axis=1).to_numpy()
    test_valid = test_mask & x.notna().all(axis=1).to_numpy()
    imputer = SimpleImputer(strategy='median')
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(imputer.fit_transform(x.loc[train_valid]))
    x_test = x_scaler.transform(imputer.transform(x.loc[test_valid]))
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y.loc[train_valid].to_numpy().reshape(-1, 1)).ravel()
    seed_everything(SEED)
    mlp_scaled = _fit_fixed(MLP(x_train.shape[1]), (x_train,), y_train, (x_test,),
                            config['final_mlp_epochs'], SEED)
    predictions.loc[x.index[test_valid], 'mlp'] = y_scaler.inverse_transform(
        mlp_scaled.reshape(-1, 1)).ravel()

    sequences, calendar, targets, seq_valid, _ = _make_sequences(
        frame, x, y, SEQUENCE_COLUMNS)
    lstm_train = train_mask & seq_valid & np.isfinite(targets)
    lstm_test = test_mask & seq_valid & np.isfinite(targets)
    train_seq, test_seq = sequences[lstm_train], sequences[lstm_test]
    medians = np.nanmedian(train_seq.reshape(-1, train_seq.shape[-1]), axis=0)
    train_seq = np.where(np.isfinite(train_seq), train_seq, medians)
    test_seq = np.where(np.isfinite(test_seq), test_seq, medians)
    sequence_mean = train_seq.reshape(-1, train_seq.shape[-1]).mean(axis=0)
    sequence_std = train_seq.reshape(-1, train_seq.shape[-1]).std(axis=0)
    sequence_std[sequence_std < 1e-8] = 1.0
    train_seq = (train_seq - sequence_mean) / sequence_std
    test_seq = (test_seq - sequence_mean) / sequence_std
    calendar_columns = [column for column in x if column.startswith('target_')]
    calendar_scaler = StandardScaler()
    calendar_train = calendar_scaler.fit_transform(calendar[lstm_train])
    calendar_test = calendar_scaler.transform(calendar[lstm_test])
    lstm_y_scaler = StandardScaler()
    lstm_y = lstm_y_scaler.fit_transform(targets[lstm_train].reshape(-1, 1)).ravel()
    seed_everything(SEED)
    lstm_scaled = _fit_fixed(
        LSTMForecaster(train_seq.shape[-1], calendar_train.shape[-1]),
        (train_seq, calendar_train), lstm_y, (test_seq, calendar_test),
        config['final_lstm_epochs'], SEED)
    lstm_index = x.index[lstm_test]
    predictions.loc[lstm_index, 'lstm_168h'] = lstm_y_scaler.inverse_transform(
        lstm_scaled.reshape(-1, 1)).ravel()

    model_columns = [column for column in predictions if column != 'actual']
    common = predictions.dropna(subset=['actual'] + model_columns)
    metrics_table = pd.DataFrame([
        {'model': model, **metrics(common['actual'], common[model])}
        for model in model_columns
    ]).sort_values('MAE')
    output = ROOT / 'results/metrics'
    predictions.to_csv(output / 'jc_final_2025_predictions.csv', index_label='predictor_time_utc')
    common.to_csv(output / 'jc_final_2025_common_predictions.csv', index_label='predictor_time_utc')
    metrics_table.to_csv(output / 'jc_final_2025_metrics.csv', index=False)
    metadata = {
        'evaluation': 'Frozen final holdout confirmation; no tuning after evaluation',
        'holdout_pristine': False,
        'holdout_caveat': 'An earlier exploratory script evaluated and displayed 2025 metrics; those values were excluded from tuning, but the holdout is not a pristine blind test.',
        'train_period': [config['final_fit_start_utc'], config['final_fit_end_utc']],
        'test_period': [config['test_start_utc'], config['test_end_utc']],
        'common_scored_rows': int(len(common)),
        'candidate_predictions': model_columns,
        'deep_model_epochs_from_development_folds': {
            'mlp': config['final_mlp_epochs'], 'lstm_168h': config['final_lstm_epochs']},
        'device': str(DEVICE),
        'metrics': metrics_table.to_dict(orient='records'),
    }
    final_metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    print(metrics_table.to_string(index=False))
    print('common scored rows:', len(common), 'device:', DEVICE)


if __name__ == '__main__':
    run_final_confirmation()
