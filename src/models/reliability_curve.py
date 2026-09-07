"""
reliability_curve.py — Curva de Fiabilidad (Reliability Curve / Calibration Curve)
para GradientBoosting y demas modelos.

Genera diagramas de fiabilidad que muestran si el modelo esta bien calibrado,
subconfiado (predicted < actual) o sobreconfiado (predicted > actual) en
cada rango de probabilidad predicha.

Incluye:
  - Curva de fiabilidad (calibration_curve)
  - Histograma de probabilidades predichas
  - ECE (Expected Calibration Error) por modelo
   - Comparacion modelo crudo vs calibrado (isotonic)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score

from src.data.data_pipeline import run_pipeline
from src.data.feature_store import (
    prepare_set_data,
    prepare_match_data,
    enrich_with_team_stats,
    compute_roster_features,
)

OUTPUT_DIR = BASE_DIR / "models" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Funciones de utilidad
# ─────────────────────────────────────────────────────────────


def compute_ece(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error (ECE).
    Divide las predicciones en n_bins equiespaciados y mide la diferencia
    media entre la probabilidad predicha y la frecuencia real.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()

        ece += (n_bin / total) * abs(avg_acc - avg_conf)

    return ece


def print_calibration_analysis(y_true, y_prob, name, n_bins=10):
    """
    Imprime analisis detallado por bin de probabilidad:
    donde el modelo esta subconfiado o sobreconfiado.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    print(f"\n  {'='*68}")
    print(f"  ANALISIS POR BIN DE PROBABILIDAD - {name}")
    print(f"  {'='*68}")
    print(
        f"  {'Rango prob':>18} {'N':>6} {'%total':>7} "
        f"{'Pred_avg':>9} {'Real_avg':>9} {'Gap':>8} {'Diagnostico'}"
    )
    print(f"  {'-'*66}")

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        gap = avg_conf - avg_acc
        pct = 100 * n_bin / len(y_true)

        if abs(gap) < 0.03:
            diag = "[OK] Bien calibrado"
        elif gap > 0.08:
            diag = "[WARN] SOBRECONFIANZA"
        elif gap > 0.03:
            diag = "- Ligera sobreconfianza"
        elif gap < -0.08:
            diag = "[WARN] SUBCONFIANZA"
        elif gap < -0.03:
            diag = "+ Ligera subconfianza"
        else:
            diag = ""

        print(
            f"  [{lo:.1f}-{hi:.1f}] {n_bin:>7} {pct:>6.1f}% "
            f"{avg_conf:>9.3f} {avg_acc:>9.3f} {gap:>+8.3f}  {diag}"
        )

    # Resumen
    ece = compute_ece(y_true, y_prob, n_bins=n_bins)
    print(f"  {'-'*66}")
    print(
        f"  ECE: {ece:.4f}  |  Media real (y): {y_true.mean():.3f}  "
        f"|  Media predicha: {y_prob.mean():.3f}  "
        f"|  Gap global: {y_prob.mean() - y_true.mean():+.3f}"
    )
    print()


def plot_reliability_diagram(
    y_test,
    prob_dict,
    title,
    filename,
    n_bins=15,
):
    """
    Genera el reliability diagram completo para uno o varios modelos.

    Args:
        y_test: array de etiquetas reales (0/1)
        prob_dict: dict {nombre_modelo: array_de_probabilidades}
        title: titulo del grafico
        filename: nombre del archivo de salida (.png)
        n_bins: numero de bins para la curva de fiabilidad
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 11), gridspec_kw={"height_ratios": [2.5, 1]})
    ax_cal = axes[0]
    ax_hist = axes[1]

    colors = {
        "GradientBoosting (crudo)": "crimson",
        "GradientBoosting (calibrado)": "royalblue",
        "GradientBoosting": "forestgreen",
    }

    for name, y_prob in prob_dict.items():
        color = colors.get(name, None)

        # Curva de fiabilidad
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins, strategy="uniform")
        ece = compute_ece(y_test, y_prob, n_bins=n_bins)
        brier = brier_score_loss(y_test, y_prob)

        ax_cal.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=f"{name}  (ECE={ece:.4f}, Brier={brier:.4f})",
            color=color,
        )

        # Histograma
        ax_hist.hist(
            y_prob,
            bins=30,
            alpha=0.55,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

    # Línea de calibración perfecta
    ax_cal.plot([0, 1], [0, 1], "k--", linewidth=1.3, alpha=0.7, label="Calibracion perfecta")

    # Líneas de referencia: sobre/subconfianza
    ax_cal.fill_between([0, 1], [0, 0], [0, 1], alpha=0.04, color="red", label="_nolegend_")
    ax_cal.fill_between([0, 1], [0, 1], [1, 1], alpha=0.04, color="blue", label="_nolegend_")

    # Anotaciones de zonas
    ax_cal.text(
        0.80,
        0.58,
        "SOBRECONFIANZA\n(pred > real)",
        fontsize=9,
        ha="center",
        color="darkred",
        style="italic",
        alpha=0.7,
    )
    ax_cal.text(
        0.80,
        0.28,
        "SUBCONFIANZA\n(pred < real)",
        fontsize=9,
        ha="center",
        color="darkblue",
        style="italic",
        alpha=0.7,
    )

    # Configurar eje de calibración
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_xlabel("Probabilidad predicha (mean predicted probability)", fontsize=12)
    ax_cal.set_ylabel("Frecuencia real de la clase positiva", fontsize=12)
    ax_cal.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax_cal.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax_cal.grid(True, alpha=0.3)
    ax_cal.set_aspect("equal")

    # Configurar histograma
    ax_hist.set_xlabel("Probabilidad predicha P(gana_local=1)", fontsize=12)
    ax_hist.set_ylabel("Número de muestras", fontsize=12)
    ax_hist.set_title("Distribucion de probabilidades predichas (test set)", fontsize=12)
    ax_hist.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_hist.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafico guardado: {filepath}")


def train_and_evaluate_gb_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    name="GradientBoosting",
):
    """
    Entrena GradientBoostingClassifier (crudo y calibrado) y devuelve
    las probabilidades predichas en test para ambos.
    """
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

    # Modelo crudo
    gb.fit(X_train, y_train)
    y_prob_raw = gb.predict_proba(X_test)[:, 1]

    # Modelo calibrado (isotonic, 3-fold CV sobre train+val combinados)
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    gb_cal = CalibratedClassifierCV(
        GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
        cv=3,
        method="isotonic",
    )
    gb_cal.fit(X_train_full, y_train_full)
    y_prob_cal = gb_cal.predict_proba(X_test)[:, 1]

    # Metricas de test
    print(f"\n  [{name}] Resultados en TEST (raw vs calibrado):")
    for label, yp in [("Crudo     ", y_prob_raw), ("Calibrado ", y_prob_cal)]:
        acc = accuracy_score(y_test, yp >= 0.5)
        auc = roc_auc_score(y_test, yp)
        brier = brier_score_loss(y_test, yp)
        ece = compute_ece(y_test, yp)
        print(f"    {label}  Acc={acc:.4f}  AUC={auc:.4f}  " f"Brier={brier:.4f}  ECE={ece:.4f}")

    return {
        "GradientBoosting (crudo)": y_prob_raw,
        "GradientBoosting (calibrado)": y_prob_cal,
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────


def main():
    print("=" * 75)
    print("  CURVA DE FIABILIDAD - GradientBoosting (Set y Match)")
    print("=" * 75)

    # 1. Cargar datos
    print("\n[1] Cargando datos...")
    data = run_pipeline()

    # ────────────────────────────────────────────────────────
    #  A) Fiabilidad en SET prediction
    # ────────────────────────────────────────────────────────
    print("\n" + "-" * 75)
    print("  A) FIABILIDAD EN PREDICCION DE SETS")
    print("-" * 75)

    X_set, y_set = prepare_set_data(data["set_features"])

    # Usar arrays numpy (tree-based no necesita escalado)
    X_tr = X_set["train"].values
    y_tr = y_set["train"].values
    X_va = X_set["val"].values
    y_va = y_set["val"].values
    X_te = X_set["test"].values
    y_te = y_set["test"].values

    probs_set = train_and_evaluate_gb_model(
        X_tr,
        y_tr,
        X_va,
        y_va,
        X_te,
        y_te,
        name="GB-Set",
    )

    print_calibration_analysis(y_te, probs_set["GradientBoosting (crudo)"], "GB SET - Crudo")
    print_calibration_analysis(
        y_te, probs_set["GradientBoosting (calibrado)"], "GB SET - Calibrado"
    )

    plot_reliability_diagram(
        y_te,
        probs_set,
        title="Diagrama de Fiabilidad - GradientBoosting en SET",
        filename="reliability_set.png",
        n_bins=15,
    )

    # ────────────────────────────────────────────────────────
    #  B) Fiabilidad en MATCH prediction
    # ────────────────────────────────────────────────────────
    print("\n" + "-" * 75)
    print("  B) FIABILIDAD EN PREDICCION DE PARTIDOS")
    print("-" * 75)

    match_df = data["match_features"].copy()

    # Enriquecer con team stats y roster features
    print("\n  Enriqueciendo match features...")
    match_df = enrich_with_team_stats(match_df, data["team_stats"])
    match_df = compute_roster_features(match_df, data["player_stats"])

    X_match, y_match = prepare_match_data(match_df)

    X_tr_m = X_match["train"].values
    y_tr_m = y_match["train"].values
    X_va_m = X_match["val"].values
    y_va_m = y_match["val"].values
    X_te_m = X_match["test"].values
    y_te_m = y_match["test"].values

    probs_match = train_and_evaluate_gb_model(
        X_tr_m,
        y_tr_m,
        X_va_m,
        y_va_m,
        X_te_m,
        y_te_m,
        name="GB-Match",
    )

    print_calibration_analysis(y_te_m, probs_match["GradientBoosting (crudo)"], "GB MATCH - Crudo")
    print_calibration_analysis(
        y_te_m, probs_match["GradientBoosting (calibrado)"], "GB MATCH - Calibrado"
    )

    plot_reliability_diagram(
        y_te_m,
        probs_match,
        title="Diagrama de Fiabilidad - GradientBoosting en PARTIDOS",
        filename="reliability_match.png",
        n_bins=15,
    )

    # ────────────────────────────────────────────────────────
    #  C) Solo modelo calibrado (mas limpio para el TFG)
    # ────────────────────────────────────────────────────────
    print("\n" + "-" * 75)
    print("  C) GRAFICO LIMPIO - Solo calibrado (Set + Match)")
    print("-" * 75)

    probs_set_cal = {"GradientBoosting (calibrado)": probs_set["GradientBoosting (calibrado)"]}
    probs_match_cal = {"GradientBoosting (calibrado)": probs_match["GradientBoosting (calibrado)"]}

    plot_reliability_diagram(
        y_te,
        probs_set_cal,
        title="Diagrama de Fiabilidad - GradientBoosting Calibrado en SET",
        filename="reliability_set_calibrado.png",
        n_bins=15,
    )

    plot_reliability_diagram(
        y_te_m,
        probs_match_cal,
        title="Diagrama de Fiabilidad - GradientBoosting Calibrado en PARTIDOS",
        filename="reliability_match_calibrado.png",
        n_bins=15,
    )

    print("\n" + "=" * 75)
    print("  TODOS LOS GRAFICOS GENERADOS EN:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
