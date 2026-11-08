"""Train and evaluate classical JC one-step baselines."""
import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.jc_baselines import naive_predictions
from src.jc_features import make_features
from src.jc_data import ROOT
from src.jc_weather import FIELDS, fetch_weather_range, join_hourly


def load_prepared():
    frame = pd.read_csv(ROOT / 'data/processed/jc_hourly_2024_2025.csv',
                        parse_dates=['timestamp_utc', 'local_time'], index_col='timestamp_utc')
    frame.index = pd.DatetimeIndex(frame.index, tz='UTC')
    return frame


def load_weathered_prepared():
    frame = load_prepared()
    weather_path = ROOT / 'data/processed/jc_weather_2024_utc.csv'
    if not weather_path.exists():
        start = pd.Timestamp('2024-01-01T05:00:00Z')
        end = pd.Timestamp('2025-01-01T05:00:00Z')
        weather, metadata = fetch_weather_range(start.date(), end.date())
        weather.to_csv(weather_path, index_label='timestamp_utc')
        (ROOT / 'data/processed/jc_weather_2024_utc.json').write_text(
            json.dumps(metadata, indent=2), encoding='utf-8')
    else:
        weather = pd.read_csv(weather_path, parse_dates=['timestamp_utc'], index_col='timestamp_utc')
        weather.index = pd.DatetimeIndex(weather.index, tz='UTC')
    return join_hourly(frame, weather)


def load_weathered_2024_2025():
    frame = load_prepared()
    weather_path = ROOT / 'data/processed/jc_weather_2024_2025_utc.csv'
    if not weather_path.exists():
        from src.jc_weather import fetch_weather_range
        weather, metadata = fetch_weather_range('2024-01-01', '2026-01-01')
        weather.to_csv(weather_path, index_label='timestamp_utc')
        (ROOT / 'data/processed/jc_weather_2024_2025_utc.json').write_text(
            json.dumps(metadata, indent=2), encoding='utf-8')
    else:
        weather = pd.read_csv(weather_path, parse_dates=['timestamp_utc'], index_col='timestamp_utc')
        weather.index = pd.DatetimeIndex(weather.index, tz='UTC')
    return join_hourly(frame, weather)


def metrics(y_true, prediction):
    valid = y_true.notna() & prediction.notna()
    return {'n': int(valid.sum()),
            'MAE': float(mean_absolute_error(y_true[valid], prediction[valid])),
            'RMSE': float(root_mean_squared_error(y_true[valid], prediction[valid])),
            'R2': float(r2_score(y_true[valid], prediction[valid]))}


def run_development_cv(frame, config, include_weather=False):
    x, y = make_features(frame)
    if not include_weather:
        x = x.loc[:, ~x.columns.str.startswith(tuple(FIELDS))]
    target_time = x.index + pd.Timedelta(hours=1)
    naive = naive_predictions(frame, y)
    usable_x = x.notna().all(axis=1)
    usable_y = y.notna()
    models = {
        'linear_regression': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LinearRegression()),
        'random_forest': make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=2, max_features=0.8)),
    }
    results = {}
    predictions = {}
    for fold in config['folds']:
        name = fold['name']
        train_mask = ((target_time >= pd.Timestamp(fold['train_start_utc'])) &
                      (target_time < pd.Timestamp(fold['train_end_utc'])) & usable_x & usable_y)
        valid_mask = ((target_time >= pd.Timestamp(fold['validation_start_utc'])) &
                      (target_time < pd.Timestamp(fold['validation_end_utc'])) & usable_x & usable_y)
        train_x, train_y = x.loc[train_mask], y.loc[train_mask]
        valid_x, valid_y = x.loc[valid_mask], y.loc[valid_mask]
        fold_results = {'train_rows': len(train_x), 'features': x.shape[1]}
        fold_predictions = pd.DataFrame({'actual': valid_y}, index=valid_y.index)
        for model_name, model in models.items():
            model.fit(train_x, train_y)
            prediction = pd.Series(model.predict(valid_x), index=valid_x.index)
            fold_results[model_name] = metrics(valid_y, prediction)
            fold_predictions[model_name] = prediction
        for baseline_name in ('naive_previous_hour', 'naive_previous_week'):
            prediction = naive.loc[valid_y.index, baseline_name]
            fold_results[baseline_name] = metrics(valid_y, prediction)
            fold_predictions[baseline_name] = prediction
        results[name] = fold_results
        predictions[name] = fold_predictions
    return results, predictions


def main():
    frame = load_prepared()
    config = json.loads((ROOT / 'configs/jc_experiment.json').read_text(encoding='utf-8'))
    output, predictions = run_development_cv(frame, config)
    weathered = load_weathered_prepared()
    weather_output, weather_predictions = run_development_cv(weathered, config, include_weather=True)
    output['weather_history'] = weather_output
    path = ROOT / 'results/metrics/jc_baselines.json'
    path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    for fold, prediction in predictions.items():
        prediction.to_csv(ROOT / f'results/metrics/jc_predictions_{fold}.csv', index_label='predictor_time_utc')
    for fold, prediction in weather_predictions.items():
        prediction.to_csv(ROOT / f'results/metrics/jc_weather_predictions_{fold}.csv', index_label='predictor_time_utc')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
