"""Replay a single historical JC hour with frozen Random Forest and LSTM recipes.

This is an inference demonstration on an observed historical timestamp, not a live
forecast service. The target value is returned only for retrospective comparison.
"""
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.jc_baseline_models import load_weathered_2024_2025
from src.jc_data import ROOT
from src.jc_features import make_features
from src.jc_torch_models import (
    DEVICE, LSTMForecaster, SEED, SEQUENCE_COLUMNS, _make_sequences, seed_everything,
)
from src.jc_weather import FIELDS

TRAIN_END_UTC = pd.Timestamp('2025-01-01T05:00:00Z')


def resolve_historical_target(frame, target_local_time=None):
    if target_local_time is None:
        target_local_time = frame['local_time'].max()
    target_local = pd.Timestamp(target_local_time)
    if target_local.tzinfo is not None:
        raise ValueError('Provide target_local_time as a naive America/New_York wall time')
    target_utc = target_local.tz_localize(
        'America/New_York', ambiguous='raise', nonexistent='raise').tz_convert('UTC')
    if target_utc not in frame.index:
        raise ValueError(f'Target hour is not present in the prepared data: {target_local}')
    predictor_time = target_utc - pd.Timedelta(hours=1)
    if predictor_time not in frame.index:
        raise ValueError('The previous UTC hour is not present in the prepared data')
    if target_utc < TRAIN_END_UTC:
        raise ValueError('Choose a target after the fixed 2024 training period')
    return target_local, target_utc, predictor_time


def _fit_lstm_fixed(model, train_inputs, train_target, input_row, epochs, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = model.to(DEVICE)
    tensors = tuple(torch.as_tensor(value, dtype=torch.float32) for value in train_inputs)
    targets = torch.as_tensor(train_target, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(*tensors, targets), batch_size=128, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for batch in loader:
            *features, actual = (value.to(DEVICE) for value in batch)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(*features), actual)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model.eval()
    inputs = tuple(torch.as_tensor(value, dtype=torch.float32, device=DEVICE)
                   for value in input_row)
    with torch.no_grad():
        return float(model(*inputs).detach().cpu().item())


def predict_historical_hour(frame, target_local_time=None, lstm_epochs=26):
    target_local, target_utc, predictor_time = resolve_historical_target(
        frame, target_local_time)
    x, y = make_features(frame)
    if predictor_time not in x.index or pd.isna(y.loc[predictor_time]):
        raise ValueError('The selected hour is not scoreable (for example, DST ambiguity)')

    target_times = x.index + pd.Timedelta(hours=1)
    train_mask = ((target_times < TRAIN_END_UTC) & y.notna().to_numpy())
    train_x = x.loc[train_mask]
    train_y = y.loc[train_mask]

    # Random Forest uses the frozen non-weather feature recipe from the evaluation.
    rf_columns = [column for column in x if not column.startswith(tuple(FIELDS))]
    rf_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1,
                              min_samples_leaf=2, max_features=0.8))
    rf_train = train_x[rf_columns]
    rf_train_valid = rf_train.notna().all(axis=1)
    if x.loc[predictor_time, rf_columns].isna().any():
        raise ValueError('The selected hour has insufficient Random Forest history')
    rf_pipeline.fit(rf_train.loc[rf_train_valid], train_y.loc[rf_train_valid])
    rf_prediction = float(rf_pipeline.predict(x.loc[[predictor_time], rf_columns])[0])

    # LSTM sees only the 168 hours ending at the predictor hour; the next-hour
    # calendar vector is known in advance. Target-hour outcomes are never inputs.
    sequences, calendar, targets, sequence_valid, calendar_columns = _make_sequences(
        frame, x, y, SEQUENCE_COLUMNS)
    train_positions = np.flatnonzero(train_mask & sequence_valid & np.isfinite(targets))
    prediction_position = x.index.get_loc(predictor_time)
    if not sequence_valid[prediction_position]:
        raise ValueError('The selected 168-hour demand history contains missing values')
    train_sequences = sequences[train_positions]
    prediction_sequence = sequences[[prediction_position]]
    medians = np.nanmedian(train_sequences.reshape(-1, train_sequences.shape[-1]), axis=0)
    train_sequences = np.where(np.isfinite(train_sequences), train_sequences, medians)
    prediction_sequence = np.where(np.isfinite(prediction_sequence), prediction_sequence, medians)
    sequence_mean = train_sequences.reshape(-1, train_sequences.shape[-1]).mean(axis=0)
    sequence_std = train_sequences.reshape(-1, train_sequences.shape[-1]).std(axis=0)
    sequence_std[sequence_std < 1e-8] = 1.0
    train_sequences = (train_sequences - sequence_mean) / sequence_std
    prediction_sequence = (prediction_sequence - sequence_mean) / sequence_std

    from sklearn.preprocessing import StandardScaler
    calendar_scaler = StandardScaler()
    calendar_train = calendar_scaler.fit_transform(calendar[train_positions])
    calendar_prediction = calendar_scaler.transform(calendar[[prediction_position]])
    target_scaler = StandardScaler()
    lstm_train_target = target_scaler.fit_transform(
        targets[train_positions].reshape(-1, 1)).ravel()
    seed_everything(SEED)
    scaled_prediction = _fit_lstm_fixed(
        LSTMForecaster(train_sequences.shape[-1], calendar_train.shape[-1]),
        (train_sequences, calendar_train), lstm_train_target,
        (prediction_sequence, calendar_prediction), lstm_epochs, SEED)
    lstm_prediction = float(target_scaler.inverse_transform([[scaled_prediction]])[0, 0])

    return {
        'target_local_time': target_local.isoformat(sep=' '),
        'target_utc': target_utc.isoformat(),
        'predictor_time_utc': predictor_time.isoformat(),
        'train_end_exclusive_utc': TRAIN_END_UTC.isoformat(),
        'history_hours': 168,
        'device': str(DEVICE),
        'lstm_epochs': lstm_epochs,
        'predictions': {
            'random_forest': max(0.0, rf_prediction),
            'lstm_168h': max(0.0, lstm_prediction),
        },
        'actual_for_historical_replay_only': float(y.loc[predictor_time]),
        'note': 'Historical one-step replay; actual is returned only for display and is never used as an input.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-local', help='Naive America/New_York hour, e.g. 2025-12-31 23:00; defaults to latest available target')
    args = parser.parse_args()
    frame = load_weathered_2024_2025()
    result = predict_historical_hour(frame, args.target_local)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
