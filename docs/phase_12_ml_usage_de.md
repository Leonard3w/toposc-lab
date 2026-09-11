# Phase 12: ML-Surrogates zur Simulationspriorisierung

Das Paket `toposc_lab.ml` priorisiert teure exakte Rechnungen. Es ersetzt
weder Solver noch Topologie-/Majorana-/Robustheitsdiagnostik. Eine Vorhersage
ist niemals ein exaktes Label und darf nicht als physikalischer Nachweis
gespeichert oder veröffentlicht werden.

## Leakage-freier Grundablauf

```python
from toposc_lab.data import DatasetSplitConfig, load_dataset, split_dataset
from toposc_lab.ml import (
    HandcraftedFeatureSchema,
    RidgeRegressor,
    extract_handcrafted_features,
)

dataset = load_dataset("results/exact_dataset.json")
split = split_dataset(dataset, config=DatasetSplitConfig(seed=2026))
by_id = dataset.by_id()
train = tuple(by_id[item] for item in split.train_ids)
validation = tuple(by_id[item] for item in split.validation_ids)

# Ausschließlich auf Train fitten.
schema = HandcraftedFeatureSchema.fit(train)
x_train = extract_handcrafted_features(train, schema=schema)
x_validation = extract_handcrafted_features(validation, schema=schema)

# `targets` stammt aus extract_regression_targets(...).valid_values_by_id().
model = RidgeRegressor().fit(
    x_train.values,
    [targets[item] for item in split.train_ids],
)
validation_prediction = model.predict(x_validation.values)
```

`HandcraftedFeatureSchema` liest nur Graph/Geometrie und explizite
Modellparameter. Spektrum, Observablen, Topologie, Robustheit und Provenienz
werden bei der Featureextraktion nicht gelesen. Unbekannte Testparameter
werden nicht nachträglich in das Schema aufgenommen.

## Exakte Targets und Masken

```python
from toposc_lab.ml import (
    RegressionTargetDefinition,
    RegressionTargetSource,
    extract_regression_targets,
)

definition = RegressionTargetDefinition(
    name="gap",
    source=RegressionTargetSource.OBSERVABLE,
    result_kind="spectral_gap",
    result_version="1",
    value_key="gap",
)
target_batch = extract_regression_targets(dataset, definition)
targets = target_batch.valid_values_by_id()
```

`valid_mask` und `invalid_reasons` müssen erhalten bleiben. Insbesondere wird
ein nicht anwendbares oder numerisch ungültiges Topologieresultat nie zu `0`
oder „trivial“ konvertiert.

## Unsicherheit, Kalibrierung und OOD

`BootstrapUncertaintyRegressor` liefert Ensemble-Streuung und Abstand zum
Trainingsraum. `IntervalCalibrator` darf ausschließlich mit Validation-
Residualen gefittet werden; `calibration_metrics` misst danach die Coverage
auf einem separaten Satz. Kleine Stichproben und Nullbreiten werden als
Fehlermodi ausgegeben.

`FeatureOODDetector` bestimmt seinen Schwellenwert aus Leave-one-out-Abständen
im Trainingssatz. Sein Ergebnis ist nur ein Akquisitionswarnsignal. Auch ein
OOD-Kandidat kann interessante oder völlig valide Physik besitzen und muss
exakt gerechnet werden.

## GNN-Policy

`MinimalGNNRegressor` ist eine feste Ein-Schicht-Referenz ohne
Architektursuche. `benchmark_simple_models_vs_gnn` vergleicht GNN und einfache
Modelle auf denselben Record-IDs. Eine GNN wird nur gewählt, wenn sie die vorab
definierte relative Verbesserung auf Validation erreicht und diese auf Test
mit gepaartem Bootstrap bestätigt. Andernfalls bleibt das einfachere Modell
Default.

Der reproduzierbare Gate-Lauf ist:

```python
from toposc_lab.ml import run_surrogate_gate_benchmark

result = run_surrogate_gate_benchmark()
assert result.passed
```
