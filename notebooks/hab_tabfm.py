"""Shared data, forecasts, and evaluation for regional notebooks 08 and 09."""

from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)

REGIONAL_TABLE = "model_table_weekly_fe_reduced.parquet"
FEATURE_SETS = "feature_sets.json"
CLASSES = (0, 1, 2)
PROB_COLUMNS = ["p_class_0", "p_class_1", "p_class_2"]
YEARS = tuple(range(2017, 2026))
EARLY_WARNING_YEARS = tuple(range(2020, 2026))
GAP_WEEKS = 4
CATBOOST_CUTOFF = 0.35
TABFM_WEIGHT = 0.50
TABFM_CHECKPOINT = "google/tabfm-1.0.0-pytorch"
TABFM_SETTINGS = {
    "n_estimators": 32,
    "max_num_rows": None,
    "batch_size": 1,
    "random_state": 42,
}
REPORT_WINDOWS = {
    "Historical comparison (2020–2024)": (2020, 2024),
    "Extended backtest (2017–2024)": (2017, 2024),
    "Final test (2025)": (2025, 2025),
}


def load_regional(data_dir):
    """Read the committed feature table and label only weeks with observed outcomes."""
    spec = (data_dir / FEATURE_SETS).read_text()
    if spec.startswith("version https://git-lfs"):
        raise ValueError("Fetch the feature list with git lfs pull --include=data/final/feature_sets.json.")
    features = json.loads(spec)["regional_core"]
    raw = pd.read_parquet(data_dir / REGIONAL_TABLE)
    labeled = raw.dropna(subset=["y_class"]).copy()
    labeled["target_week"] = pd.to_datetime(labeled["target_week"])
    if not labeled["target_week"].is_unique:
        raise ValueError("Expected one labeled row per regional target week.")
    if not labeled["y_class"].isin(CLASSES).all() or not labeled["last_class"].isin(CLASSES).all():
        raise ValueError("The regional table must use classes 0, 1, and 2.")
    if not labeled["target_week"].dt.year.eq(labeled["target_year"]).all():
        raise ValueError("target_year must describe the year of target_week.")
    labeled["y_class"] = labeled["y_class"].astype(int)
    return raw, labeled, features


def forecast_folds(labeled):
    """Yield the historical forecasts and the final 2025 test."""
    for year in YEARS:
        evaluation = labeled.loc[labeled["target_year"] == year]
        train = labeled.loc[labeled["target_year"] < year]
        if evaluation.empty:
            raise ValueError(f"No labeled weeks in {year}.")
        if year < 2025:
            cutoff = evaluation["target_week"].min() - pd.Timedelta(weeks=GAP_WEEKS)
            train = train.loc[train["target_week"] < cutoff]
        if set(train["y_class"]) != set(CLASSES):
            raise ValueError(f"Training for {year} requires all three classes.")
        yield year, train, evaluation


def load_tabfm():
    import tabfm
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use the same precision on CPU and CUDA. This is the published classifier, in inference mode.
    return tabfm.tabfm_v1_0_0_pytorch.load(
        model_type="classification", device=device, dtype=torch.float32
    ).eval()


def tabfm_features(frame, features, missing_columns):
    result = frame[features].astype(float).copy()
    for column in missing_columns:
        result[f"{column}_missing"] = result[column].isna().astype(float)
    return result


def fit_tabfm(model, train, features):
    from tabfm import TabFMClassifier

    missing_columns = [column for column in features if train[column].isna().any()]
    classifier = TabFMClassifier(model=model, **TABFM_SETTINGS)
    classifier.fit(tabfm_features(train, features, missing_columns), train["y_class"].to_numpy())
    return classifier, missing_columns


def fit_lightgbm(train, features):
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        objective="multiclass", num_class=3, learning_rate=0.03, max_depth=5,
        n_estimators=100, num_leaves=15, class_weight="balanced",
        random_state=42, verbosity=-1, n_jobs=1,
    )
    return model.fit(train[features], train["y_class"])


def fit_catboost(train, features):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        loss_function="MultiClass", iterations=300, learning_rate=0.03, depth=5,
        l2_leaf_reg=5, auto_class_weights="Balanced", random_seed=42,
        verbose=False, allow_writing_files=False, thread_count=1,
    )
    return model.fit(train[features], train["y_class"])


def probability_matrix(probabilities, classes=CLASSES):
    """Align columns to classes 0, 1, and 2 and check that rows are probabilities."""
    probs = np.asarray(probabilities, dtype=float)
    class_order = list(classes)
    if probs.ndim != 2 or probs.shape[1] != len(CLASSES) or set(class_order) != set(CLASSES):
        raise ValueError("Expected one probability column for each severity class.")
    probs = probs[:, [class_order.index(label) for label in CLASSES]]
    if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1):
        raise ValueError("Every forecast must have finite probabilities that sum to 1.")
    return probs / probs.sum(axis=1, keepdims=True)


def as_forecasts(evaluation, classifier, predictor_features):
    probabilities = probability_matrix(
        classifier.predict_proba(predictor_features), classifier.classes_
    )
    return pd.DataFrame(probabilities, index=evaluation["target_week"], columns=PROB_COLUMNS)


def tabfm_backtest(labeled, features, model):
    forecasts = []
    for year, train, evaluation in forecast_folds(labeled):
        classifier, missing_columns = fit_tabfm(model, train, features)
        query = tabfm_features(evaluation, features, missing_columns)
        forecasts.append(as_forecasts(evaluation, classifier, query))
        print(f"{year}: {len(train)} context weeks, {len(evaluation)} forecast weeks", flush=True)
    return pd.concat(forecasts)


def baseline_backtests(labeled, features):
    lightgbm = []
    catboost = []
    for _, train, evaluation in forecast_folds(labeled):
        lightgbm.append(as_forecasts(evaluation, fit_lightgbm(train, features), evaluation[features]))
        catboost.append(as_forecasts(evaluation, fit_catboost(train, features), evaluation[features]))
    return pd.concat(lightgbm), pd.concat(catboost)


def observed_forecast_weeks(labeled):
    columns = ["target_week", "target_year", "y_class", "last_class"]
    return pd.concat([evaluation[columns] for _, _, evaluation in forecast_folds(labeled)]).set_index(
        "target_week"
    )


def artifact_config(data_dir, features):
    return {
        "source_sha256": {
            name: sha256((data_dir / name).read_bytes()).hexdigest()
            for name in (REGIONAL_TABLE, FEATURE_SETS)
        },
        "regional_core": features,
        "classes": list(CLASSES),
        "forecast_years": list(YEARS),
        "gap_weeks_for_historical_folds": GAP_WEEKS,
        "tabfm_checkpoint": TABFM_CHECKPOINT,
        "tabfm_settings": TABFM_SETTINGS,
        "versions": {
            package: version(package) for package in (
                "tabfm", "torch", "numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm", "catboost"
            )
        },
    }


def load_tabfm_forecasts(artifact_dir, data_dir, features, observed_weeks):
    saved = json.loads((artifact_dir / "config.json").read_text())
    expected = artifact_config(data_dir, features)
    if any(saved.get(key) != value for key, value in expected.items() if key != "versions"):
        raise ValueError("The saved TabFM forecasts use different inputs or settings. Rerun notebook 08.")
    forecasts = pd.read_parquet(artifact_dir / "tabfm_probabilities.parquet")
    if not forecasts.index.is_unique or set(forecasts.index) != set(observed_weeks.index):
        raise ValueError("Saved TabFM forecasts must cover the evaluation weeks exactly. Rerun notebook 08.")
    return forecasts.loc[observed_weeks.index, PROB_COLUMNS]


def risk(probabilities):
    return probabilities[:, 1:].sum(axis=1)


def risk_decision(probabilities, cutoff):
    severity = 1 + (probabilities[:, 2] > probabilities[:, 1])
    return np.where(risk(probabilities) >= cutoff, severity, 0)


def prior_year_cutoffs(observed, candidate, reference):
    """Match CatBoost's false-alarm count on earlier predictions for class-0 weeks."""
    cutoffs = {}
    for year in EARLY_WARNING_YEARS:
        negatives = (observed["target_year"] < year) & (observed["y_class"] == 0)
        false_alarms = int((risk(reference)[negatives] >= CATBOOST_CUTOFF).sum())
        ranked = np.sort(risk(candidate)[negatives])[::-1]
        if false_alarms == 0:
            cutoff = float(np.nextafter(1.0, np.inf))
        elif false_alarms == len(ranked):
            cutoff = 0.0
        else:
            cutoff = float(np.mean(ranked[false_alarms - 1:false_alarms + 1]))
        cutoffs[year] = cutoff
    return cutoffs


def evaluate(observed, forecasts):
    """Return aggregate metrics, decisions per week, and fitted alert cutoffs."""
    decisions = {"Persistence": (observed["last_class"].to_numpy(), None)}
    catboost = probability_matrix(forecasts["CatBoost"][PROB_COLUMNS])
    cutoffs = {}
    for name, frame in forecasts.items():
        if not frame.index.equals(observed.index):
            raise ValueError(f"{name} predictions must match the forecast weeks in order.")
        probs = probability_matrix(frame[PROB_COLUMNS])
        if name == "CatBoost":
            decisions["CatBoost (0.35)"] = (risk_decision(probs, CATBOOST_CUTOFF), probs)
            continue
        decisions[name] = (probs.argmax(axis=1), probs)
        cutoffs[name] = prior_year_cutoffs(observed, probs, catboost)
        threshold = observed["target_year"].map(cutoffs[name])
        pred = risk_decision(probs, threshold.to_numpy())
        decisions[f"{name} (matched alert budget)"] = (np.where(threshold.notna(), pred, -1), probs)

    y = observed["y_class"].to_numpy()
    eligible = observed["last_class"].eq(0).to_numpy()
    onset = eligible & (y > 0)
    rows = []
    for window, (start, end) in REPORT_WINDOWS.items():
        mask = observed["target_year"].between(start, end).to_numpy()
        for name, (pred, probs) in decisions.items():
            if (pred[mask] < 0).any():
                continue
            actual, predicted = y[mask], pred[mask]
            hits = int(((pred > 0) & onset & mask).sum())
            total = int((onset & mask).sum())
            misses = int(((actual == 2) & (predicted == 0)).sum())
            false_alarms = int(((actual == 0) & (predicted > 0)).sum())
            rows.append({
                "window": window, "model": name, "weeks": int(mask.sum()),
                "onsets caught": hits, "onsets": total,
                "onset recall": hits / total if total else np.nan,
                "severe misses": misses, "false alarms": false_alarms,
                "cost 2:1": 2 * misses + false_alarms, "cost 3:1": 3 * misses + false_alarms,
                "accuracy": accuracy_score(actual, predicted),
                "balanced accuracy": balanced_accuracy_score(actual, predicted),
                "macro F1": f1_score(actual, predicted, labels=CLASSES, average="macro", zero_division=0),
                "onset AP": average_precision_score(
                    y[eligible & mask] > 0, risk(probs)[eligible & mask]
                ) if probs is not None and total else np.nan,
                "log loss": log_loss(actual, probs[mask], labels=CLASSES) if probs is not None else np.nan,
            })
    weekly = observed.copy()
    for name, (pred, _) in decisions.items():
        weekly[name] = pd.Series(pred, index=observed.index).mask(pred < 0).astype("Int64")
    return pd.DataFrame(rows).set_index(["window", "model"]), weekly, cutoffs


def save_tabfm_artifacts(directory, config, features, labeled, forecasts, summary, weekly, cutoffs):
    directory.mkdir(parents=True, exist_ok=True)
    context = labeled.loc[labeled["target_year"] < 2025, ["target_week", "y_class", *features]]
    context.to_parquet(directory / "tabfm_context_through_2024.parquet", index=False)
    forecasts.to_parquet(directory / "tabfm_probabilities.parquet")
    summary.to_csv(directory / "metrics.csv")
    weekly.to_csv(directory / "weekly_decisions.csv")
    config = {**config, "matched_alert_cutoffs": cutoffs, "context_through_year": 2024}
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def verify_tabfm_context(directory, model, sample, expected):
    config = json.loads((directory / "config.json").read_text())
    features = config["regional_core"]
    context = pd.read_parquet(directory / "tabfm_context_through_2024.parquet")
    classifier, missing_columns = fit_tabfm(model, context, features)
    result = as_forecasts(sample, classifier, tabfm_features(sample, features, missing_columns))
    np.testing.assert_allclose(result, expected.loc[result.index], atol=1e-5, rtol=1e-5)
    return result
