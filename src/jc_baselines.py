"""Naive one-step demand forecast baselines."""
import pandas as pd


def naive_predictions(hourly, targets):
    """Return previous-hour and previous-week predictions aligned to target UTC times."""
    if not targets.index.isin(hourly.index).all():
        raise ValueError('All targets must exist on the hourly UTC index')
    # `targets.index` stores predictor time t; each prediction is for t+1.
    previous_hour = hourly['departures'].reindex(targets.index)
    previous_week = hourly['departures'].shift(167).reindex(targets.index)
    return pd.DataFrame({'naive_previous_hour': previous_hour,
                         'naive_previous_week': previous_week}, index=targets.index)
