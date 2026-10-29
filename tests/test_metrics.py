import pandas as pd

from src import config
from src.metrics import best_f1_threshold, metrics_at_threshold

Y = [0, 0, 0, 0, 1, 1, 1, 1]
P = [0.1, 0.2, 0.4, 0.6, 0.35, 0.55, 0.7, 0.9]


def test_confusion_matrix_at_half():
    m = metrics_at_threshold(Y, P, 0.5)
    assert m["confusion_matrix"] == [[3, 1], [1, 3]]
    assert m["accuracy"] == 0.75 and m["precision"] == 0.75 and m["recall"] == 0.75


def test_lower_threshold_raises_recall():
    assert metrics_at_threshold(Y, P, 0.3)["recall"] > metrics_at_threshold(Y, P, 0.8)["recall"]


def test_threshold_extremes_do_not_crash():
    assert metrics_at_threshold(Y, P, 0.99)["precision"] == 0.0  # nobody flagged
    assert metrics_at_threshold(Y, P, 0.0)["recall"] == 1.0      # everybody flagged


def test_best_f1_threshold_is_at_least_as_good_as_half():
    t, f1 = best_f1_threshold(Y, P)
    assert f1 >= metrics_at_threshold(Y, P, 0.5)["f1"] and 0 <= t <= 1


def test_saved_evaluation_matches_saved_metrics():
    ev = pd.read_csv(config.EVAL_PATH)
    m = metrics_at_threshold(ev["y_true"], ev["y_prob"], 0.5)
    import json
    saved = json.load(open(config.METRICS_PATH))
    assert m["confusion_matrix"] == saved["confusion_matrix"]
