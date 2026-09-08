"""Treino, validação LOGO e seleção do modelo final de ritmo."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "Dados" / "processados"
CONFIGURACAO = RAIZ / "configuracao" / "zonas_fc.json"
MODELOS_FOLDS = RAIZ / "artefactos" / "modelos" / "folds"
MODELO_FINAL = RAIZ / "artefactos" / "modelos" / "final"
METRICAS = RAIZ / "artefactos" / "metricas"
PREVISOES = RAIZ / "artefactos" / "previsoes_cv"
FEATURES = [
    "zona_ordem",
    "zona_limite_inferior_bpm",
    "zona_limite_superior_bpm",
    "zona_aberta",
    "declive_pct",
    "altitude_normalizada",
    "distancia_inicio_km",
    "subida_acumulada_m",
    "comprimento_troco_m",
]
ALVO = "ritmo_alvo_s_km"
NOMES_MODELOS = [
    "Ridge",
    "RandomForest",
    "GradientBoosting",
    "HistGradientBoosting",
    "SVR_RBF",
    "RedeNeuronal_MLP",
    "Dummy",
]


def criar_modelos() -> dict[str, Any]:
    """Cria candidatos com parâmetros fixos e pipelines autocontidos."""
    restricoes = [-1, -1, -1, 0, 1, 0, 0, 0, 0]
    if len(restricoes) != len(FEATURES):
        raise AssertionError("Vetor monotónico incompatível com as features")
    return {
        "Ridge": Pipeline(
            [("scaler", StandardScaler()), ("regressor", Ridge(alpha=10.0))]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=5,
            loss="huber",
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            learning_rate=0.05,
            l2_regularization=5,
            monotonic_cst=restricoes,
            random_state=42,
        ),
        "SVR_RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", SVR(kernel="rbf", C=10, epsilon=10, gamma="scale")),
            ]
        ),
        "RedeNeuronal_MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    MLPRegressor(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        solver="adam",
                        alpha=0.01,
                        batch_size=64,
                        learning_rate_init=0.001,
                        max_iter=1000,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=40,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Dummy": DummyRegressor(strategy="mean"),
    }


def calcular_pesos(df: pd.DataFrame) -> np.ndarray:
    """Atribui peso total unitário a cada atividade."""
    contagens = df.groupby("atividade_id")["atividade_id"].transform("size")
    return (1.0 / contagens).to_numpy(dtype=float)


def calcular_metricas(y_real: np.ndarray, y_previsto: np.ndarray) -> dict[str, float]:
    """Calcula métricas por atividade em segundos por quilómetro."""
    mae = mean_absolute_error(y_real, y_previsto)
    rmse = float(np.sqrt(mean_squared_error(y_real, y_previsto)))
    r2 = r2_score(y_real, y_previsto) if len(y_real) >= 2 else np.nan
    erro_assinado = float(np.mean(y_previsto - y_real))
    return {
        "mae_s_km": float(mae),
        "rmse_s_km": rmse,
        "r2": float(r2),
        "erro_assinado_s_km": erro_assinado,
    }


def _ajustar(modelo: Any, x: pd.DataFrame, y: pd.Series, pesos: np.ndarray) -> Any:
    """Ajusta um estimador ou pipeline com o parâmetro de peso correto."""
    if isinstance(modelo, Pipeline):
        modelo.fit(x, y, regressor__sample_weight=pesos)
    else:
        modelo.fit(x, y, sample_weight=pesos)
    return modelo


def avaliar_logo(
    df: pd.DataFrame, modelos: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avalia todos os modelos nos mesmos folds por atividade."""
    grupos = df["atividade_id"].astype(str)
    if grupos.nunique() < 2:
        raise ValueError("São necessárias pelo menos duas atividades para LeaveOneGroupOut")
    x = df[FEATURES]
    y = df[ALVO]
    logo = LeaveOneGroupOut()
    metricas: list[dict[str, Any]] = []
    previsoes: list[pd.DataFrame] = []
    for indice_fold, (indice_treino, indice_teste) in enumerate(logo.split(x, y, grupos), start=1):
        treino = df.iloc[indice_treino]
        teste = df.iloc[indice_teste]
        atividade_teste = str(teste["atividade_id"].iloc[0])
        if set(treino["atividade_id"].astype(str)) & set(teste["atividade_id"].astype(str)):
            raise AssertionError("Fuga de atividade entre treino e teste")
        pesos = calcular_pesos(treino)
        print(f"Fold {indice_fold}/{grupos.nunique()} — atividade {atividade_teste}")
        for nome, base in modelos.items():
            modelo = clone(base)
            inicio = time.perf_counter()
            modelo = _ajustar(modelo, treino[FEATURES], treino[ALVO], pesos)
            tempo_treino = time.perf_counter() - inicio
            inicio = time.perf_counter()
            previsto = modelo.predict(teste[FEATURES])
            tempo_previsao = time.perf_counter() - inicio
            valores = calcular_metricas(teste[ALVO].to_numpy(), previsto)
            metricas.append(
                {
                    "fold": indice_fold,
                    "atividade_id": atividade_teste,
                    "modelo": nome,
                    "n_treino": len(treino),
                    "n_teste": len(teste),
                    **valores,
                    "tempo_treino_s": tempo_treino,
                    "tempo_previsao_s": tempo_previsao,
                }
            )
            bloco = teste[["amostra_id", "atividade_id", "troco_id", "zona_ordem", "declive_pct", ALVO]].copy()
            bloco["modelo"] = nome
            bloco["ritmo_previsto_s_km"] = previsto
            bloco["residuo_s_km"] = bloco[ALVO] - bloco["ritmo_previsto_s_km"]
            previsoes.append(bloco)
            pasta = MODELOS_FOLDS / nome
            pasta.mkdir(parents=True, exist_ok=True)
            joblib.dump(modelo, pasta / f"fold_atividade_{atividade_teste}.joblib")
    return pd.DataFrame(metricas), pd.concat(previsoes, ignore_index=True)


def _segundos_para_mmss(valor: float) -> str:
    if not np.isfinite(valor):
        return "n/d"
    minutos = int(valor // 60)
    segundos = int(round(valor - 60 * minutos))
    if segundos == 60:
        minutos += 1
        segundos = 0
    return f"{minutos:02d}:{segundos:02d}"


def resumir_modelos(metricas: pd.DataFrame) -> pd.DataFrame:
    """Cria ranking macro, com uma atividade a valer um voto igual."""
    resumo = (
        metricas.groupby("modelo", as_index=False)
        .agg(
            mae_macro_s_km=("mae_s_km", "mean"),
            rmse_macro_s_km=("rmse_s_km", "mean"),
            r2_macro=("r2", "mean"),
            erro_assinado_macro_s_km=("erro_assinado_s_km", "mean"),
            tempo_treino_medio_s=("tempo_treino_s", "mean"),
            tempo_previsao_medio_s=("tempo_previsao_s", "mean"),
            n_atividades=("atividade_id", "nunique"),
        )
    )
    dummy_mae = float(resumo.loc[resumo["modelo"] == "Dummy", "mae_macro_s_km"].iloc[0])
    resumo["melhoria_vs_dummy_pct"] = 100 * (dummy_mae - resumo["mae_macro_s_km"]) / dummy_mae
    resumo["mae_macro_mmss_km"] = resumo["mae_macro_s_km"].map(_segundos_para_mmss)
    return resumo.sort_values(["mae_macro_s_km", "rmse_macro_s_km", "tempo_previsao_medio_s"]).reset_index(drop=True)


def selecionar_vencedor(resumo: pd.DataFrame) -> str:
    """Escolhe o melhor candidato real pelas métricas fora do treino."""
    candidatos = resumo[resumo["modelo"] != "Dummy"].sort_values(
        ["mae_macro_s_km", "rmse_macro_s_km", "tempo_previsao_medio_s"]
    )
    if candidatos.empty:
        raise RuntimeError("Não existem modelos candidatos para selecionar")
    return str(candidatos.iloc[0]["modelo"])


def treinar_final(
    nome_modelo: str, df: pd.DataFrame, modelos: dict[str, Any]
) -> Any:
    """Treina o vencedor com todas as atividades após a seleção OOF."""
    modelo = clone(modelos[nome_modelo])
    return _ajustar(modelo, df[FEATURES], df[ALVO], calcular_pesos(df))


def guardar_pacote_modelo(
    modelo: Any, nome: str, df: pd.DataFrame, resumo: pd.DataFrame
) -> dict[str, Any]:
    """Guarda pipeline, configuração, unidades e gama de treino."""
    with CONFIGURACAO.open(encoding="utf-8") as ficheiro:
        zonas = json.load(ficheiro)
    linha = resumo[resumo["modelo"] == nome].iloc[0]
    metadados = {
        "modelo_vencedor": nome,
        "features": FEATURES,
        "alvo": ALVO,
        "unidades": {"alvo": "s/km", "declive_pct": "%", "altitude_normalizada": "altitude_m/1000"},
        "perfil": {"passo_grelha_m": 10.0, "comprimento_troco_m": 50.0, "media_centrada_pontos": 5},
        "zonas": zonas,
        "metricas_oof": {
            "mae_macro_s_km": float(linha["mae_macro_s_km"]),
            "rmse_macro_s_km": float(linha["rmse_macro_s_km"]),
            "r2_macro": float(linha["r2_macro"]),
            "melhoria_vs_dummy_pct": float(linha["melhoria_vs_dummy_pct"]),
        },
        "n_atividades": int(df["atividade_id"].nunique()),
        "n_amostras": int(len(df)),
        "gamas_treino": {
            coluna: {"min": float(df[coluna].min()), "max": float(df[coluna].max())}
            for coluna in [*FEATURES, ALVO]
        },
        "versoes": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    pacote = {"modelo": modelo, "metadados": metadados}
    MODELO_FINAL.mkdir(parents=True, exist_ok=True)
    joblib.dump(pacote, MODELO_FINAL / "modelo_ritmo_final.joblib")
    (MODELO_FINAL / "metadados_modelo.json").write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadados


def main() -> None:
    """Executa a avaliação comparável e treina um único modelo final."""
    for pasta in [MODELOS_FOLDS, MODELO_FINAL, METRICAS, PREVISOES]:
        pasta.mkdir(parents=True, exist_ok=True)
    caminho = DADOS / "amostras_modelo.csv"
    if not caminho.exists():
        raise FileNotFoundError("Execute preparar_dados.py antes do treino")
    df = pd.read_csv(caminho, dtype={"atividade_id": str, "amostra_id": str})
    ausentes = [coluna for coluna in [*FEATURES, ALVO, "atividade_id"] if coluna not in df]
    if ausentes:
        raise ValueError(f"Colunas em falta: {ausentes}")
    modelos = criar_modelos()
    metricas, previsoes = avaliar_logo(df, modelos)
    resumo = resumir_modelos(metricas)
    vencedor = selecionar_vencedor(resumo)
    resumo["vencedor"] = resumo["modelo"].eq(vencedor)
    modelo_final = treinar_final(vencedor, df, modelos)
    metadados = guardar_pacote_modelo(modelo_final, vencedor, df, resumo)
    metricas.to_csv(METRICAS / "metricas_por_atividade.csv", index=False)
    resumo.to_csv(METRICAS / "resumo_modelos.csv", index=False)
    previsoes.to_csv(PREVISOES / "previsoes_fora_do_treino.csv", index=False)
    resultado = {
        "modelo_vencedor": vencedor,
        "metricas_oof": metadados["metricas_oof"],
        "ranking": resumo.to_dict(orient="records"),
        "nota": "Seleção baseada apenas em previsões LeaveOneGroupOut fora do treino.",
    }
    (METRICAS / "resultado.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    linha = resumo[resumo["modelo"] == vencedor].iloc[0]
    print("\nRanking dos modelos:")
    print(resumo[["modelo", "mae_macro_s_km", "rmse_macro_s_km", "r2_macro", "melhoria_vs_dummy_pct"]].to_string(index=False))
    print(f"\nModelo vencedor: {vencedor}")
    print(f"MAE macro OOF: {linha['mae_macro_s_km']:.2f} s/km ({linha['mae_macro_mmss_km']})")
    print(f"Melhoria face ao baseline: {linha['melhoria_vs_dummy_pct']:.1f}%")


if __name__ == "__main__":
    main()
