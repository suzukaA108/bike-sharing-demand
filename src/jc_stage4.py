"""Unified development-fold comparison, feature ablation, and error analysis."""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.jc_baseline_models import (
    load_prepared, load_weathered_prepared, metrics, run_development_cv,
)
from src.jc_data import ROOT
from src.jc_features import feature_groups, make_features
from src.jc_torch_models import run_torch_cv
from src.jc_weather import FIELDS


def _read_weather_predictions(paths):
    frames = []
    for path in paths:
        frame = pd.read_csv(path, parse_dates=['predictor_time_utc'])
        frame = frame.set_index('predictor_time_utc')
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize('UTC')
        frames.append(frame)
    return frames


def common_model_table(prediction_frames):
    if not prediction_frames:
        raise ValueError('No prediction frames supplied')
    actuals = [frame['actual'] for frame in prediction_frames if 'actual' in frame]
    if not actuals:
        raise ValueError('Prediction frames must contain actual values')
    reference = actuals[0]
    for actual in actuals[1:]:
        shared = reference.index.intersection(actual.index)
        if not np.allclose(reference.loc[shared], actual.loc[shared], equal_nan=True):
            raise ValueError('Prediction files disagree on actual targets')
    merged = reference.rename('actual').to_frame()
    for frame in prediction_frames:
        predictions = frame.drop(columns=['actual'], errors='ignore')
        overlap = set(predictions.columns) & set(merged.columns)
        if overlap:
            raise ValueError(f'Duplicate prediction names: {sorted(overlap)}')
        merged = merged.join(predictions, how='inner')
    prediction_columns = [c for c in merged if c != 'actual']
    merged = merged.dropna(subset=['actual'] + prediction_columns)
    return merged


def _fold_masks(x, y, fold):
    target_time = x.index + pd.Timedelta(hours=1)
    train = ((target_time >= pd.Timestamp(fold['train_start_utc'])) &
             (target_time < pd.Timestamp(fold['train_end_utc'])) &
             y.notna().to_numpy())
    valid = ((target_time >= pd.Timestamp(fold['validation_start_utc'])) &
             (target_time < pd.Timestamp(fold['validation_end_utc'])) &
             y.notna().to_numpy())
    return train, valid


def run_rf_ablation(frame, config):
    x, y = make_features(frame)
    groups = feature_groups(x.columns)
    combinations = {
        'calendar': groups['calendar'],
        'calendar+demand': groups['calendar'] + groups['demand_history'],
        'calendar+weather': groups['calendar'] + groups['weather_history'],
        'calendar+demand+weather': groups['calendar'] + groups['demand_history'] + groups['weather_history'],
    }
    results = []
    for fold in config['folds']:
        train_base, valid_base = _fold_masks(x, y, fold)
        valid_common = valid_base.copy()
        for columns in combinations.values():
            valid_common &= x[columns].notna().all(axis=1).to_numpy()
        for name, columns in combinations.items():
            subset = x[columns]
            train_mask = train_base & subset.notna().all(axis=1).to_numpy()
            valid_mask = valid_common
            model = make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=2, max_features=0.8))
            model.fit(subset.loc[train_mask], y.loc[train_mask])
            prediction = pd.Series(model.predict(subset.loc[valid_mask]), index=subset.index[valid_mask])
            scores = metrics(y.loc[prediction.index], prediction)
            results.append({'fold': fold['name'], 'feature_set': name,
                            'features': len(columns), 'train_rows': int(train_mask.sum()), **scores})
    return pd.DataFrame(results)


def _error_groups(common, fold_name, target_weather):
    target_index = common.index + pd.Timedelta(hours=1)
    local = target_index.tz_convert('America/New_York')
    diagnostics = pd.DataFrame(index=common.index)
    diagnostics['actual'] = common['actual']
    diagnostics['hour_local'] = local.hour
    diagnostics['weekday_local'] = local.dayofweek
    diagnostics['is_weekend'] = local.dayofweek >= 5
    holiday_dates = USFederalHolidayCalendar().holidays(
        start=local.min().date(), end=local.max().date()).date
    diagnostics['is_us_federal_holiday'] = np.array([v in holiday_dates for v in local.date])
    weather = target_weather.reindex(target_index)
    diagnostics['precipitation_target_hour'] = weather['precipitation'].to_numpy()
    diagnostics['weather_group'] = np.where(
        diagnostics['precipitation_target_hour'].isna(), 'unknown',
        np.where(diagnostics['precipitation_target_hour'] > 0, 'precipitation', 'no_precipitation'))
    model_columns = [column for column in common if column not in ('actual',)]
    rows = []
    for model in model_columns:
        diagnostics['absolute_error'] = (common['actual'] - common[model]).abs()
        diagnostics['squared_error'] = (common['actual'] - common[model]) ** 2
        groups = {
            'hour_period': pd.cut(diagnostics['hour_local'], [-1, 5, 9, 15, 19, 23],
                                  labels=['night_00_05', 'morning_06_09', 'day_10_15', 'evening_16_19', 'late_20_23']),
            'weekend': diagnostics['is_weekend'].map({True: 'weekend', False: 'weekday'}),
            'us_federal_holiday': diagnostics['is_us_federal_holiday'].map({True: 'holiday', False: 'non_holiday'}),
            'target_weather': diagnostics['weather_group'],
        }
        for group_name, labels in groups.items():
            summary = diagnostics.groupby(labels, observed=False).agg(
                n=('absolute_error', 'count'), MAE=('absolute_error', 'mean'),
                mean_squared_error=('squared_error', 'mean')).reset_index()
            for _, row in summary.iterrows():
                if row['n']:
                    rows.append({'fold': fold_name, 'model': model, 'group': group_name,
                                 'category': str(row.iloc[0]), 'n': int(row['n']),
                                 'MAE': float(row['MAE']),
                                 'RMSE': float(np.sqrt(row['mean_squared_error']))})
    return pd.DataFrame(rows)


def run_stage4():
    config = json.loads((ROOT / 'configs/jc_experiment.json').read_text(encoding='utf-8'))
    raw = load_prepared()
    weathered = load_weathered_prepared()
    dev_end = pd.Timestamp(config['test_start_utc'])
    raw = raw.loc[raw.index < dev_end]
    weathered = weathered.loc[weathered.index < dev_end]

    classical_results, classical_predictions = run_development_cv(raw, config)
    weather_results, weather_predictions = run_development_cv(
        weathered, config, include_weather=True)
    torch_results, torch_predictions = run_torch_cv(weathered, config)

    common_by_fold = {}
    comparison_rows = []
    error_frames = []
    for fold in config['folds']:
        name = fold['name']
        classical = classical_predictions[name]
        weather_frame = weather_predictions[name].drop(
            columns=['naive_previous_hour', 'naive_previous_week'], errors='ignore').rename(
                columns={key: f'{key}_weather' for key in ('linear_regression', 'random_forest')})
        torch_frame = torch_predictions[name].rename(columns={
            'mlp_prediction': 'mlp', 'lstm_168h_prediction': 'lstm_168h'})
        frames = [classical, weather_frame, torch_frame]
        common = common_model_table(frames)
        common_by_fold[name] = common
        for model in common.columns.drop('actual'):
            comparison_rows.append({'fold': name, 'model': model,
                                    **metrics(common['actual'], common[model])})
        weather = pd.read_csv(ROOT / 'data/processed/jc_weather_2024_utc.csv',
                              parse_dates=['timestamp_utc'], index_col='timestamp_utc')
        weather.index = pd.DatetimeIndex(weather.index, tz='UTC')
        error_frames.append(_error_groups(common, name, weather))

    ablation = run_rf_ablation(weathered, config)
    output_dir = ROOT / 'results/metrics'
    pd.DataFrame(comparison_rows).to_csv(output_dir / 'jc_unified_comparison.csv', index=False)
    ablation.to_csv(output_dir / 'jc_rf_feature_ablation.csv', index=False)
    pd.concat(error_frames, ignore_index=True).to_csv(
        output_dir / 'jc_development_error_analysis.csv', index=False)
    for fold, frame in common_by_fold.items():
        frame.to_csv(output_dir / f'jc_unified_predictions_{fold}.csv', index_label='predictor_time_utc')
    metadata = {
        'evaluation_scope': '2024 development folds only; 2025 excluded from fitting, selection, and error analysis',
        'common_timestamp_policy': 'Intersect scored timestamps across every model within each fold; all reported leaderboard metrics use this shared set.',
        'common_rows': {k: int(len(v)) for k, v in common_by_fold.items()},
        'target_weather_policy': 'Target-hour precipitation used only for post-hoc error grouping, never as a model input.',
        'holiday_policy': 'US federal holiday calendar applied to target-hour America/New_York local date.',
        'models': list(pd.DataFrame(comparison_rows)['model'].unique()),
    }
    (output_dir / 'jc_stage4_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(pd.DataFrame(comparison_rows).to_string(index=False))
    print('\nRF feature ablation\n', ablation.to_string(index=False))
    print('\nCommon rows', metadata['common_rows'])
    return metadata


if __name__ == '__main__':
    run_stage4()
