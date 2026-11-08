"""Leakage-safe tabular features for one-hour-ahead JC departures."""
import numpy as np
import pandas as pd

from src.jc_weather import FIELDS

TARGET = 'departures'
DEMAND_LAGS = (1, 2, 3, 24, 168)
WEATHER_LAGS = (1, 2, 3, 24)
ROLLING_WINDOWS = (3, 24, 168)


def make_features(hourly):
    """Feature row at UTC t predicts demand at t+1; all observations are through t."""
    if not isinstance(hourly.index, pd.DatetimeIndex) or str(hourly.index.tz) != 'UTC':
        raise ValueError('Hourly data must use a UTC-aware DatetimeIndex')
    if not hourly.index.is_monotonic_increasing or hourly.index.has_duplicates:
        raise ValueError('Hourly index must be sorted and unique')
    if not hourly.index.equals(pd.date_range(hourly.index[0], hourly.index[-1], freq='h', tz='UTC')):
        raise ValueError('Hourly index must be continuous; do not shift features across gaps')
    if TARGET not in hourly:
        raise ValueError(f'Missing target column: {TARGET}')

    x = pd.DataFrame(index=hourly.index)
    target_local = (hourly['local_time'].shift(-1) if 'local_time' in hourly
                    else pd.Series((hourly.index + pd.Timedelta(hours=1)).tz_convert('America/New_York').tz_localize(None), index=hourly.index))
    x['target_hour'] = target_local.dt.hour
    x['target_weekday'] = target_local.dt.dayofweek
    x['target_month'] = target_local.dt.month
    x['target_is_weekend'] = (target_local.dt.dayofweek >= 5).astype('int8')
    x['target_hour_sin'] = np.sin(2 * np.pi * x['target_hour'] / 24)
    x['target_hour_cos'] = np.cos(2 * np.pi * x['target_hour'] / 24)
    x['target_weekday_sin'] = np.sin(2 * np.pi * x['target_weekday'] / 7)
    x['target_weekday_cos'] = np.cos(2 * np.pi * x['target_weekday'] / 7)
    x['target_month_sin'] = np.sin(2 * np.pi * x['target_month'] / 12)
    x['target_month_cos'] = np.cos(2 * np.pi * x['target_month'] / 12)

    observed_demand = hourly[TARGET].astype(float)
    for lag in DEMAND_LAGS:
        x[f'demand_lag_{lag}'] = observed_demand.shift(lag - 1)
    history = observed_demand.shift(0)
    for window in ROLLING_WINDOWS:
        x[f'demand_mean_{window}'] = history.rolling(window, min_periods=window).mean()
        x[f'demand_std_{window}'] = history.rolling(window, min_periods=window).std()

    for column in FIELDS:
        if column in hourly:
            weather = hourly[column].astype(float)
            for lag in WEATHER_LAGS:
                x[f'{column}_lag_{lag}'] = weather.shift(lag - 1)
            for window in (24, 168):
                x[f'{column}_mean_{window}'] = weather.shift(0).rolling(
                    window, min_periods=window).mean()

    # Every feature row at t predicts y at t+1; never include target-hour outcomes.
    y = observed_demand.shift(-1).rename(TARGET)
    target_valid = hourly.get('dst_ambiguous_target', pd.Series(False, index=hourly.index)).shift(-1).fillna(True).astype(bool)
    valid = y.notna() & ~target_valid
    x = x.loc[valid]
    y = y.loc[valid]
    if not x.index.equals(y.index):
        raise AssertionError('Feature and target timestamps differ')
    return x, y


def feature_groups(columns):
    groups = {'calendar': [], 'demand_history': [], 'weather_history': []}
    for column in columns:
        if column.startswith('target_'):
            groups['calendar'].append(column)
        elif column.startswith('demand_'):
            groups['demand_history'].append(column)
        else:
            groups['weather_history'].append(column)
    return groups
