"""PyTorch MLP and 168-hour LSTM with fold-local preprocessing and early stopping."""
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.jc_baseline_models import load_prepared, load_weathered_prepared, metrics
from src.jc_data import ROOT
from src.jc_features import make_features
from src.jc_weather import FIELDS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 60
PATIENCE = 8
SEQUENCE_COLUMNS = ['departures'] + FIELDS


class MLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class LSTMForecaster(nn.Module):
    def __init__(self, n_sequence_features, n_calendar_features, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(n_sequence_features, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_calendar_features, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, sequence, calendar):
        _, (hidden, _) = self.lstm(sequence)
        return self.head(torch.cat([hidden[-1], calendar], dim=1)).squeeze(-1)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_torch(model, train_inputs, train_y, valid_inputs, valid_y, seed=SEED):
    seed_everything(seed)
    model = model.to(DEVICE)
    train_tensors = tuple(torch.as_tensor(value, dtype=torch.float32) for value in train_inputs)
    valid_tensors = tuple(torch.as_tensor(value, dtype=torch.float32, device=DEVICE)
                          for value in valid_inputs)
    y_train = torch.as_tensor(train_y, dtype=torch.float32)
    y_valid = torch.as_tensor(valid_y, dtype=torch.float32, device=DEVICE)
    loader = DataLoader(TensorDataset(*train_tensors, y_train), batch_size=BATCH_SIZE,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_loss = float('inf')
    best_state = None
    best_epoch = 0
    wait = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in loader:
            *features, target = (tensor.to(DEVICE) for tensor in batch)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*features)
            loss = loss_fn(prediction, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_prediction = model(*valid_tensors)
            val_loss = float(loss_fn(val_prediction, y_valid).item())
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError('Training did not produce a valid checkpoint')
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(*valid_tensors).detach().cpu().numpy()
    return prediction, {'best_epoch': best_epoch, 'best_validation_mse_scaled': best_loss,
                        'epochs_run': epoch, 'device': str(DEVICE)}


def _fold_masks(index, target_index, fold):
    target_time = index + pd.Timedelta(hours=1)
    train = ((target_time >= pd.Timestamp(fold['train_start_utc'])) &
             (target_time < pd.Timestamp(fold['train_end_utc'])))
    valid = ((target_time >= pd.Timestamp(fold['validation_start_utc'])) &
             (target_time < pd.Timestamp(fold['validation_end_utc'])))
    valid &= target_index.notna().to_numpy()
    return train, valid


def _mlp_fold(x, y, train_mask, valid_mask, seed):
    seed_everything(seed)
    usable = x.notna().all(axis=1).to_numpy()
    train_mask = train_mask & usable & y.notna().to_numpy()
    valid_mask = valid_mask & usable & y.notna().to_numpy()
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(x.loc[train_mask]))
    x_valid = scaler.transform(imputer.transform(x.loc[valid_mask]))
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y.loc[train_mask].to_numpy().reshape(-1, 1)).ravel()
    y_valid = y_scaler.transform(y.loc[valid_mask].to_numpy().reshape(-1, 1)).ravel()
    prediction_scaled, training = _fit_torch(
        MLP(x_train.shape[1]), (x_train,), y_train, (x_valid,), y_valid, seed)
    prediction = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).ravel()
    return x.index[valid_mask], prediction, training, int(train_mask.sum())


def _make_sequences(frame, x, y, required_columns):
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing LSTM sequence columns: {sorted(missing)}')
    values = frame[required_columns].astype(float).to_numpy()
    calendar_columns = [column for column in x if column.startswith('target_')]
    calendar = x[calendar_columns].to_numpy(dtype=float)
    targets = y.to_numpy(dtype=float)
    frame_positions = frame.index.get_indexer(x.index)
    if (frame_positions < 0).any():
        raise ValueError('Feature timestamps must align exactly with the hourly frame')
    samples, sequence_valid = [], []
    for frame_pos in frame_positions:
        start = frame_pos - 167
        if start < 0:
            samples.append(np.full((168, len(required_columns)), np.nan))
            sequence_valid.append(False)
        else:
            window = values[start:frame_pos + 1]
            samples.append(window)
            # Missing demand history invalidates a sample; weather can be imputed fold-locally.
            sequence_valid.append(np.isfinite(window[:, 0]).all())
    return (np.asarray(samples, dtype=np.float32), calendar, targets,
            np.asarray(sequence_valid, dtype=bool), calendar_columns)


def _lstm_fold(frame, x, y, train_mask, valid_mask, seed):
    seed_everything(seed)
    sequences, calendar, targets, sequence_valid, calendar_columns = _make_sequences(
        frame, x, y, SEQUENCE_COLUMNS)
    train_mask = train_mask & sequence_valid & np.isfinite(targets)
    valid_mask = valid_mask & sequence_valid & np.isfinite(targets)
    train_seq = sequences[train_mask]
    valid_seq = sequences[valid_mask]

    # Fit imputation and normalization only on the training fold's sequence observations.
    medians = np.nanmedian(train_seq.reshape(-1, train_seq.shape[-1]), axis=0)
    train_seq = np.where(np.isfinite(train_seq), train_seq, medians)
    valid_seq = np.where(np.isfinite(valid_seq), valid_seq, medians)
    mean = train_seq.reshape(-1, train_seq.shape[-1]).mean(axis=0)
    std = train_seq.reshape(-1, train_seq.shape[-1]).std(axis=0)
    std[std < 1e-8] = 1.0
    train_seq = (train_seq - mean) / std
    valid_seq = (valid_seq - mean) / std

    calendar_scaler = StandardScaler()
    cal_train = calendar_scaler.fit_transform(calendar[train_mask])
    cal_valid = calendar_scaler.transform(calendar[valid_mask])
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(targets[train_mask].reshape(-1, 1)).ravel()
    y_valid = y_scaler.transform(targets[valid_mask].reshape(-1, 1)).ravel()
    pred_scaled, training = _fit_torch(
        LSTMForecaster(train_seq.shape[-1], cal_train.shape[-1]),
        (train_seq, cal_train), y_train, (valid_seq, cal_valid), y_valid, seed)
    prediction = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    return x.index[valid_mask], prediction, training, int(train_mask.sum())


def run_torch_cv(frame, config):
    x, y = make_features(frame)
    results = {}
    predictions = {}
    for fold_number, fold in enumerate(config['folds']):
        train_mask, valid_mask = _fold_masks(x.index, y, fold)
        fold_result = {}
        fold_predictions = pd.DataFrame(index=x.index[valid_mask])
        for name, runner in (('mlp', _mlp_fold), ('lstm_168h', _lstm_fold)):
            if name == 'mlp':
                index, prediction, training, train_rows = runner(
                    x, y, train_mask.copy(), valid_mask.copy(), SEED + fold_number)
            else:
                index, prediction, training, train_rows = runner(
                    frame, x, y, train_mask.copy(), valid_mask.copy(), SEED + fold_number)
            actual = y.loc[index]
            fold_result[name] = {**metrics(actual, pd.Series(prediction, index=index)),
                                 'train_rows': train_rows, **training}
            fold_predictions[f'{name}_prediction'] = pd.Series(prediction, index=index)
        fold_predictions['actual'] = y.loc[fold_predictions.index]
        results[fold['name']] = fold_result
        predictions[fold['name']] = fold_predictions
    return results, predictions


def main():
    config = json.loads((ROOT / 'configs/jc_experiment.json').read_text(encoding='utf-8'))
    frame = load_weathered_prepared()
    # Crop before constructing features so no 2025 targets enter model-selection code.
    frame = frame.loc[frame.index < pd.Timestamp(config['test_start_utc'])]
    results, predictions = run_torch_cv(frame, config)
    output = ROOT / 'results/metrics'
    (output / 'jc_torch_development.json').write_text(
        json.dumps({'device': str(DEVICE), 'seed': SEED, 'results': results}, indent=2),
        encoding='utf-8')
    for fold, prediction in predictions.items():
        prediction.to_csv(output / f'jc_torch_predictions_{fold}.csv', index_label='predictor_time_utc')
    print(json.dumps({'device': str(DEVICE), 'results': results}, indent=2))


if __name__ == '__main__':
    main()
