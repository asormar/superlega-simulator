"""
data_pipeline.py — Pipeline de limpieza y preprocesamiento de datos.

Lee todos los CSVs crudos de la carpeta DB/, los limpia, normaliza
los nombres de equipos, y produce DataFrames unificados listos para ML.
"""

import glob
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.team_mapper import normalize_team_name

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────
# Rutas base
# ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_DIR = BASE_DIR / "DB"


# ─────────────────────────────────────────────────────────────
# 1. SETS / PARTIDOS
# ─────────────────────────────────────────────────────────────


def load_sets_partidos() -> pd.DataFrame:
    """
    Carga y limpia el archivo sets_partidos.csv.
    Normaliza nombres de equipos y añade columnas derivadas.
    """
    path = DB_DIR / "sets_partidos.csv"
    df = pd.read_csv(path, encoding="utf-8")

    # Normalizar nombres de equipos
    df["equipo_local"] = df["equipo_local"].apply(normalize_team_name)
    df["equipo_visitante"] = df["equipo_visitante"].apply(normalize_team_name)

    # Extraer temporada como entero (año de inicio)
    df["temporada_inicio"] = df["temporada"].apply(
        lambda x: int(str(x).split("/")[0]) if "/" in str(x) else int(x)
    )

    # Columna de ganador del set
    df["ganador_set"] = df.apply(
        lambda r: r["equipo_local"] if r["ganador_set_local"] == 1 else r["equipo_visitante"],
        axis=1,
    )

    # Diferencia de puntos en el set
    df["diff_puntos_set"] = df["puntos_local"] - df["puntos_visitante"]

    print(
        f"  [sets_partidos] {len(df)} filas, "
        f"temporadas {df['temporada'].nunique()}, "
        f"equipos {df['equipo_local'].nunique()}"
    )

    return df


# ─────────────────────────────────────────────────────────────
# 2. MATCH FEATURES
# ─────────────────────────────────────────────────────────────


def load_match_features() -> pd.DataFrame:
    """
    Carga y limpia match_features.csv.
    Este archivo ya contiene features pre-calculadas a nivel de partido.
    """
    path = DB_DIR / "features" / "match_features.csv"
    df = pd.read_csv(path, encoding="utf-8")

    # Normalizar nombres de equipos
    df["local"] = df["local"].apply(normalize_team_name)
    df["visitante"] = df["visitante"].apply(normalize_team_name)

    # Extraer temporada como entero
    df["temporada_inicio"] = df["temporada"].apply(
        lambda x: int(str(x).split("/")[0]) if "/" in str(x) else int(x)
    )

    # Asegurar que la columna target es int
    df["gana_local"] = df["gana_local"].astype(int)

    print(f"  [match_features] {len(df)} partidos, " f"temporadas {df['temporada'].nunique()}")

    return df


# ─────────────────────────────────────────────────────────────
# 3. SET FEATURES
# ─────────────────────────────────────────────────────────────


def load_set_features() -> pd.DataFrame:
    """
    Carga y limpia set_features.csv.
    Este archivo tiene 2 filas por set (antes/después de jugar el set).
    Usaremos la fila "antes" (pre-set) para predecir y la "después"
    como verdad para entrenar.
    """
    path = DB_DIR / "features" / "set_features.csv"
    df = pd.read_csv(path, encoding="utf-8")

    # Extraer temporada desde partido_id
    df["temporada"] = df["partido_id"].apply(
        lambda x: str(x).split("_")[0] if "_" in str(x) else ""
    )
    df["temporada_inicio"] = df["temporada"].apply(
        lambda x: int(str(x).split("/")[0]) if "/" in str(x) else 0
    )

    # Asegurar que target es int
    df["ganador_set_local"] = df["ganador_set_local"].astype(int)

    # Las filas vienen en pares: la primera es "pre-set" y la segunda
    # es "post-set" (con sets_h_antes / sets_a_antes actualizados).
    # Para predicción usamos la PRIMERA fila de cada par (pre-set).
    # Marcamos cuáles son pre-set vs post-set.
    df["fila_idx"] = df.groupby(["partido_id", "set_num"]).cumcount()
    df_pre = df[df["fila_idx"] == 0].copy()
    df_pre.drop(columns=["fila_idx"], inplace=True)

    print(f"  [set_features] {len(df_pre)} filas pre-set " f"(de {len(df)} totales)")

    return df_pre


# ─────────────────────────────────────────────────────────────
# 4. ENFRENTAMIENTOS DIRECTOS (H2H)
# ─────────────────────────────────────────────────────────────


def load_enfrentamientos_directos() -> pd.DataFrame:
    """
    Carga todos los CSVs de enfrentamientos directos y los unifica.
    Limpia nombres duplicados (MonzaMonza → Monza).
    """
    pattern = str(DB_DIR / "enfrentamientos_directos" / "enfrentamientos_directos_*.csv")
    files = glob.glob(pattern)

    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f, encoding="utf-8")
        dfs.append(df)

    if not dfs:
        print("  [enfrentamientos] No se encontraron archivos")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Normalizar nombres (aquí están duplicados como MonzaMonza)
    df["home_club"] = df["home_club"].apply(normalize_team_name)
    df["away_club"] = df["away_club"].apply(normalize_team_name)

    # Resultado del partido
    df["gana_local"] = (df["home_sets"] > df["away_sets"]).astype(int)
    df["resultado"] = df["home_sets"].astype(str) + "-" + df["away_sets"].astype(str)

    print(f"  [enfrentamientos] {len(df)} partidos H2H, " f"temporadas: {df['season'].nunique()}")

    return df


# ─────────────────────────────────────────────────────────────
# 5. ESTADÍSTICAS DE EQUIPO POR TEMPORADA
# ─────────────────────────────────────────────────────────────


def load_team_season_stats() -> pd.DataFrame:
    """
    Carga Comparacion_equipos_10_años.csv — stats agregadas por equipo/temporada.
    """
    path = DB_DIR / "Comparacion_equipos_10_años.csv"
    df = pd.read_csv(path, encoding="utf-8")

    # Normalizar nombre del club
    df["Club_Club"] = df["Club_Club"].apply(normalize_team_name)

    # Renombrar columnas para claridad
    df.rename(
        columns={
            "Club_Club": "equipo",
            "Played Matches_Played Matches": "partidos_jugados",
            "Played Set_Played Set": "sets_jugados",
            "POINTS_Tot": "puntos_totales",
            "POINTS_BP": "break_points",
            "SERVE_Tot": "saques_totales",
            "SERVE_Ace": "aces",
            "SERVE_Err.": "errores_saque",
            "SERVE_Ace per Set": "aces_por_set",
            "RECEPTION_Tot": "recepciones_totales",
            "RECEPTION_Err.": "errores_recepcion",
            "RECEPTION_Exc.": "recepciones_excelentes",
            "RECEPTION_Exc. %": "pct_recepcion_exc",
            "ATTACK_Tot": "ataques_totales",
            "ATTACK_Err.": "errores_ataque",
            "ATTACK_Blocked": "ataques_bloqueados",
            "ATTACK_Exc.": "ataques_ganados",
            "ATTACK_Exc. %": "pct_ataque",
            "ATTACK_Effic.": "eficiencia_ataque",
            "BLOCK_Exc.": "bloqueos_ganados",
            "BLOCK_Points per Set": "bloqueos_por_set",
            "Temporada": "temporada_year",
        },
        inplace=True,
    )

    # Calcular métricas derivadas por set
    df["puntos_por_set"] = df["puntos_totales"] / df["sets_jugados"]
    df["aces_ratio"] = df["aces"] / df["saques_totales"]
    df["ataque_eficacia"] = df["ataques_ganados"] / df["ataques_totales"]
    df["recepcion_eficacia"] = df["recepciones_excelentes"] / df["recepciones_totales"]

    print(f"  [team_season_stats] {len(df)} filas equipo-temporada")

    return df


# ─────────────────────────────────────────────────────────────
# 6. ESTADÍSTICAS DE JUGADORES
# ─────────────────────────────────────────────────────────────


def load_player_stats() -> pd.DataFrame:
    """
    Carga los archivos de stats_por_equipo_completo (stats de jugadores
    por equipo y temporada). Excluye las filas 'Team Totals'.
    """
    pattern = str(DB_DIR / "stats_por_equipo_completo" / "*_historial_10_años.csv")
    files = glob.glob(pattern)

    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f, encoding="utf-8")
        dfs.append(df)

    if not dfs:
        print("  [player_stats] No se encontraron archivos")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Separar jugadores individuales de totales de equipo
    df["es_total_equipo"] = df["Player_Player"].str.contains("Team Totals", case=False, na=False)

    # Solo jugadores individuales
    df_players = df[~df["es_total_equipo"]].copy()

    # Renombrar columnas clave
    df_players.rename(
        columns={
            "Player_Player": "jugador",
            "Played Matches_Played Matches": "partidos",
            "Played Set_Played Set": "sets",
            "POINTS_Tot": "puntos",
            "POINTS_BP": "break_points",
            "SERVE_Ace": "aces",
            "SERVE_Err.": "errores_saque",
            "ATTACK_Tot": "ataques_totales",
            "ATTACK_Exc.": "ataques_ganados",
            "ATTACK_Exc. %": "pct_ataque",
            "BLOCK_Exc.": "bloqueos",
            "RECEPTION_Tot": "recepciones",
            "RECEPTION_Exc.": "recepciones_exc",
            "RECEPTION_Exc. %": "pct_recepcion",
            "Temporada": "temporada",
            "ID_Equipo": "equipo_id",
        },
        inplace=True,
    )

    # Convertir columnas numéricas
    num_cols = [
        "partidos",
        "sets",
        "puntos",
        "aces",
        "ataques_totales",
        "ataques_ganados",
        "bloqueos",
        "recepciones",
        "recepciones_exc",
    ]
    for col in num_cols:
        if col in df_players.columns:
            df_players[col] = pd.to_numeric(df_players[col], errors="coerce")

    # Calcular stats por set
    df_players["puntos_por_set"] = df_players["puntos"] / df_players["sets"].replace(0, np.nan)
    df_players["aces_por_set"] = df_players["aces"] / df_players["sets"].replace(0, np.nan)

    print(
        f"  [player_stats] {len(df_players)} filas de jugadores, "
        f"{df_players['jugador'].nunique()} jugadores unicos"
    )

    return df_players


# ─────────────────────────────────────────────────────────────
# 7. PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────


def run_pipeline() -> dict[str, pd.DataFrame]:
    """
    Ejecuta todo el pipeline de carga y limpieza.
    Devuelve un diccionario con todos los DataFrames limpios.
    """
    print("=" * 60)
    print("PIPELINE DE DATOS - SuperLega Volleyball Simulator")
    print("=" * 60)

    data = {}

    print("\n[1/6] Cargando sets_partidos...")
    data["sets"] = load_sets_partidos()

    print("\n[2/6] Cargando match_features...")
    data["match_features"] = load_match_features()

    print("\n[3/6] Cargando set_features...")
    data["set_features"] = load_set_features()

    print("\n[4/6] Cargando enfrentamientos directos...")
    data["h2h"] = load_enfrentamientos_directos()

    print("\n[5/6] Cargando stats de equipo por temporada...")
    data["team_stats"] = load_team_season_stats()

    print("\n[6/6] Cargando stats de jugadores...")
    data["player_stats"] = load_player_stats()

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DEL PIPELINE")
    print("=" * 60)
    for name, df in data.items():
        print(f"  {name:20s} -> {len(df):>6} filas, {len(df.columns):>3} columnas")

    return data


# ─────────────────────────────────────────────────────────────
# Ejecución directa
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = run_pipeline()

    # Validaciones básicas
    print("\n" + "=" * 60)
    print("VALIDACIONES")
    print("=" * 60)

    # Check: no hay equipos sin normalizar en match_features
    mf = data["match_features"]
    all_teams = set(mf["local"].unique()) | set(mf["visitante"].unique())
    print(f"\n  Equipos en match_features: {sorted(all_teams)}")

    # Check: distribución de target
    print(f"\n  Distribución gana_local (match): " f"{mf['gana_local'].value_counts().to_dict()}")

    sf = data["set_features"]
    print(
        f"  Distribución ganador_set_local (set): "
        f"{sf['ganador_set_local'].value_counts().to_dict()}"
    )

    print("\nPipeline completado correctamente.")
