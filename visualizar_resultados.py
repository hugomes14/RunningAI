"""Visualizações e diagnósticos das previsões fora do treino."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

from treinar_modelos import FEATURES, ALVO, _ajustar, calcular_pesos, criar_modelos


RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "Dados" / "processados"
METRICAS = RAIZ / "artefactos" / "metricas"
PREVISOES = RAIZ / "artefactos" / "previsoes_cv"
MODELOS_FOLDS = RAIZ / "artefactos" / "modelos" / "folds"
GRAFICOS = RAIZ / "artefactos" / "graficos"


def grafico_comparacao_modelos(resumo: pd.DataFrame) -> None:
    """Compara MAE e RMSE macro de todos os modelos."""
    dados = resumo.melt(
        id_vars="modelo",
        value_vars=["mae_macro_s_km", "rmse_macro_s_km"],
        var_name="metrica",
        value_name="erro_s_km",
    )
    fig, eixo = plt.subplots(figsize=(11, 6))
    sns.barplot(data=dados, x="modelo", y="erro_s_km", hue="metrica", ax=eixo)
    eixo.set(xlabel="Modelo", ylabel="Erro macro (s/km)", title="Comparação dos modelos em validação LOGO")
    eixo.tick_params(axis="x", rotation=25)
    for contentor in eixo.containers:
        eixo.bar_label(contentor, fmt="%.1f", fontsize=8)
    fig.tight_layout()
    fig.savefig(GRAFICOS / "comparacao_modelos.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def grafico_por_atividade(metricas: pd.DataFrame) -> None:
    """Mostra o MAE de cada modelo em cada atividade excluída."""
    fig, eixo = plt.subplots(figsize=(15, 7))
    sns.barplot(data=metricas, x="atividade_id", y="mae_s_km", hue="modelo", ax=eixo)
    eixo.set(xlabel="Atividade de teste", ylabel="MAE (s/km)", title="MAE fora do treino por atividade")
    eixo.tick_params(axis="x", rotation=55)
    eixo.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(GRAFICOS / "metricas_por_atividade.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def grafico_tempos(resumo: pd.DataFrame) -> None:
    """Compara custos médios de treino e previsão em escala logarítmica."""
    dados = resumo.melt(
        id_vars="modelo",
        value_vars=["tempo_treino_medio_s", "tempo_previsao_medio_s"],
        var_name="fase",
        value_name="tempo_s",
    )
    fig, eixo = plt.subplots(figsize=(11, 6))
    sns.barplot(data=dados, x="modelo", y="tempo_s", hue="fase", ax=eixo)
    eixo.set_yscale("log")
    eixo.set(xlabel="Modelo", ylabel="Tempo médio (s, escala log)", title="Tempos de treino e previsão")
    eixo.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(GRAFICOS / "tempos_modelos.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def grafico_previsto_real(previsoes: pd.DataFrame, vencedor: str) -> None:
    """Mostra valores OOF previstos e reais do vencedor."""
    dados = previsoes[previsoes["modelo"] == vencedor]
    minimo = min(dados[ALVO].min(), dados["ritmo_previsto_s_km"].min())
    maximo = max(dados[ALVO].max(), dados["ritmo_previsto_s_km"].max())
    fig, eixo = plt.subplots(figsize=(8, 8))
    sns.scatterplot(
        data=dados,
        x=ALVO,
        y="ritmo_previsto_s_km",
        hue="zona_ordem",
        palette="viridis",
        alpha=0.55,
        s=25,
        ax=eixo,
    )
    eixo.plot([minimo, maximo], [minimo, maximo], "--", color="black", label="Identidade")
    eixo.set(
        xlabel="Ritmo real (s/km)",
        ylabel="Ritmo previsto (s/km)",
        title=f"Previsto vs. real — {vencedor}",
    )
    fig.tight_layout()
    fig.savefig(GRAFICOS / "previsto_vs_real.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def grafico_residuos(previsoes: pd.DataFrame, vencedor: str) -> None:
    """Guarda distribuição e padrão dos resíduos OOF."""
    dados = previsoes[previsoes["modelo"] == vencedor]
    fig, eixos = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=dados, x="residuo_s_km", bins=40, kde=True, ax=eixos[0], color="#2878B5")
    eixos[0].axvline(0, color="black", linestyle="--")
    eixos[0].set(title="Distribuição dos resíduos", xlabel="Real − previsto (s/km)")
    sns.scatterplot(
        data=dados,
        x="ritmo_previsto_s_km",
        y="residuo_s_km",
        hue="zona_ordem",
        palette="viridis",
        alpha=0.55,
        s=24,
        ax=eixos[1],
    )
    eixos[1].axhline(0, color="black", linestyle="--")
    eixos[1].set(title="Resíduos vs. previsto", xlabel="Previsto (s/km)", ylabel="Real − previsto (s/km)")
    fig.tight_layout()
    fig.savefig(GRAFICOS / "residuos_modelo.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def avaliar_por_zona_declive(previsoes: pd.DataFrame, vencedor: str) -> pd.DataFrame:
    """Calcula erros do vencedor por zona e classe de declive."""
    dados = previsoes[previsoes["modelo"] == vencedor].copy()
    dados["classe_declive"] = pd.cut(
        dados["declive_pct"],
        bins=[-np.inf, -6, -3, 0, 3, 6, np.inf],
        labels=["<-6", "-6 a -3", "-3 a 0", "0 a 3", "3 a 6", ">6"],
    )
    linhas: list[dict[str, Any]] = []
    for tipo, coluna in [("zona", "zona_ordem"), ("declive", "classe_declive")]:
        for grupo, bloco in dados.groupby(coluna, observed=True):
            erro = bloco["ritmo_previsto_s_km"] - bloco[ALVO]
            linhas.append(
                {
                    "tipo": tipo,
                    "grupo": str(grupo),
                    "n": len(bloco),
                    "mae_s_km": float(np.abs(erro).mean()),
                    "rmse_s_km": float(np.sqrt(np.mean(erro**2))),
                    "erro_assinado_s_km": float(erro.mean()),
                }
            )
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(METRICAS / "metricas_por_zona_declive.csv", index=False)
    fig, eixos = plt.subplots(1, 2, figsize=(14, 5))
    for eixo, tipo, titulo in zip(eixos, ["zona", "declive"], ["Por zona", "Por declive"]):
        bloco = tabela[tabela["tipo"] == tipo]
        sns.barplot(data=bloco, x="grupo", y="mae_s_km", ax=eixo, color="#72B7B2")
        eixo.set(xlabel=tipo.capitalize(), ylabel="MAE (s/km)", title=titulo)
        for contentor in eixo.containers:
            eixo.bar_label(contentor, fmt="%.1f", fontsize=8)
    fig.suptitle(f"Erro fora do treino do {vencedor}")
    fig.tight_layout()
    fig.savefig(GRAFICOS / "erro_por_zona_declive.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return tabela


def calcular_importancia_permutacao(df: pd.DataFrame, vencedor: str) -> pd.DataFrame:
    """Agrega importâncias calculadas só na atividade de teste de cada fold."""
    linhas: list[dict[str, Any]] = []
    for atividade, teste in df.groupby("atividade_id"):
        caminho = MODELOS_FOLDS / vencedor / f"fold_atividade_{atividade}.joblib"
        if not caminho.exists():
            raise FileNotFoundError(f"Modelo de fold em falta: {caminho}")
        modelo = joblib.load(caminho)
        resultado = permutation_importance(
            modelo,
            teste[FEATURES],
            teste[ALVO],
            scoring="neg_mean_absolute_error",
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        for feature, media, desvio in zip(FEATURES, resultado.importances_mean, resultado.importances_std):
            linhas.append(
                {
                    "atividade_id": str(atividade),
                    "feature": feature,
                    "importancia_media_s_km": float(media),
                    "importancia_desvio_s_km": float(desvio),
                }
            )
    detalhe = pd.DataFrame(linhas)
    tabela = (
        detalhe.groupby("feature", as_index=False)
        .agg(
            importancia_media_s_km=("importancia_media_s_km", "mean"),
            dispersao_atividades_s_km=("importancia_media_s_km", "std"),
            n_atividades=("atividade_id", "nunique"),
        )
        .sort_values("importancia_media_s_km", ascending=False)
    )
    detalhe.to_csv(METRICAS / "importancia_permutacao_por_atividade.csv", index=False)
    tabela.to_csv(METRICAS / "importancia_permutacao.csv", index=False)
    fig, eixo = plt.subplots(figsize=(10, 6))
    ordem = tabela.sort_values("importancia_media_s_km")
    eixo.barh(ordem["feature"], ordem["importancia_media_s_km"], xerr=ordem["dispersao_atividades_s_km"].fillna(0))
    eixo.axvline(0, color="black", linewidth=0.8)
    eixo.set(xlabel="Aumento de MAE após permutação (s/km)", title="Importância por permutação fora do treino")
    fig.tight_layout()
    fig.savefig(GRAFICOS / "importancia_permutacao.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return tabela


def curva_aprendizagem_por_atividades(df: pd.DataFrame, vencedor: str) -> pd.DataFrame:
    """Mede o vencedor com conjuntos crescentes de atividades completas."""
    atividades = np.array(sorted(df["atividade_id"].astype(str).unique()))
    if len(atividades) < 3:
        raise ValueError("A curva exige pelo menos três atividades")
    modelos = criar_modelos()
    rng = np.random.default_rng(42)
    linhas: list[dict[str, Any]] = []
    n_repeticoes = min(8, len(atividades))
    for repeticao in range(n_repeticoes):
        ordem = rng.permutation(atividades)
        for n_treino in range(2, len(atividades)):
            ids_treino = set(ordem[:n_treino])
            treino = df[df["atividade_id"].astype(str).isin(ids_treino)]
            teste = df[~df["atividade_id"].astype(str).isin(ids_treino)]
            modelo = clone(modelos[vencedor])
            modelo = _ajustar(modelo, treino[FEATURES], treino[ALVO], calcular_pesos(treino))
            previsto_treino = modelo.predict(treino[FEATURES])
            previsto_teste = modelo.predict(teste[FEATURES])
            linhas.append(
                {
                    "repeticao": repeticao,
                    "n_atividades_treino": n_treino,
                    "n_atividades_validacao": teste["atividade_id"].nunique(),
                    "mae_treino_s_km": mean_absolute_error(treino[ALVO], previsto_treino),
                    "mae_validacao_s_km": mean_absolute_error(teste[ALVO], previsto_teste),
                }
            )
    detalhe = pd.DataFrame(linhas)
    detalhe.to_csv(METRICAS / "curva_aprendizagem_repeticoes.csv", index=False)
    tabela = (
        detalhe.groupby("n_atividades_treino", as_index=False)
        .agg(
            mae_treino_media_s_km=("mae_treino_s_km", "mean"),
            mae_treino_dp_s_km=("mae_treino_s_km", "std"),
            mae_validacao_media_s_km=("mae_validacao_s_km", "mean"),
            mae_validacao_dp_s_km=("mae_validacao_s_km", "std"),
        )
    )
    tabela.to_csv(METRICAS / "curva_aprendizagem.csv", index=False)
    fig, eixo = plt.subplots(figsize=(10, 6))
    x = tabela["n_atividades_treino"].to_numpy()
    for prefixo, cor, nome in [("treino", "#2878B5", "Treino"), ("validacao", "#C44E52", "Validação")]:
        media = tabela[f"mae_{prefixo}_media_s_km"].to_numpy()
        desvio = tabela[f"mae_{prefixo}_dp_s_km"].fillna(0).to_numpy()
        eixo.plot(x, media, marker="o", color=cor, label=nome)
        eixo.fill_between(x, media - desvio, media + desvio, color=cor, alpha=0.15)
    eixo.set(
        xlabel="Número de atividades completas no treino",
        ylabel="MAE (s/km)",
        title=f"Curva de aprendizagem por atividades — {vencedor}",
    )
    eixo.legend()
    fig.tight_layout()
    fig.savefig(GRAFICOS / "curva_aprendizagem_atividades.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return tabela


def main() -> None:
    """Gera todos os gráficos e diagnósticos do vencedor."""
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    METRICAS.mkdir(parents=True, exist_ok=True)
    resumo = pd.read_csv(METRICAS / "resumo_modelos.csv")
    metricas = pd.read_csv(METRICAS / "metricas_por_atividade.csv", dtype={"atividade_id": str})
    previsoes = pd.read_csv(PREVISOES / "previsoes_fora_do_treino.csv", dtype={"atividade_id": str})
    df = pd.read_csv(DADOS / "amostras_modelo.csv", dtype={"atividade_id": str})
    vencedor = str(resumo.loc[resumo["vencedor"].astype(bool), "modelo"].iloc[0])
    grafico_comparacao_modelos(resumo)
    grafico_por_atividade(metricas)
    grafico_tempos(resumo)
    grafico_previsto_real(previsoes, vencedor)
    grafico_residuos(previsoes, vencedor)
    tabela_grupos = avaliar_por_zona_declive(previsoes, vencedor)
    importancias = calcular_importancia_permutacao(df, vencedor)
    curva = curva_aprendizagem_por_atividades(df, vencedor)
    zona_pior = tabela_grupos[tabela_grupos["tipo"] == "zona"].sort_values("mae_s_km", ascending=False).iloc[0]
    linha = resumo[resumo["modelo"] == vencedor].iloc[0]
    diagnostico = [
        "# Diagnóstico do modelo",
        "",
        f"- Modelo vencedor: **{vencedor}**.",
        f"- MAE macro fora do treino: **{linha['mae_macro_s_km']:.2f} s/km**.",
        f"- RMSE macro fora do treino: **{linha['rmse_macro_s_km']:.2f} s/km**.",
        f"- Melhoria face ao Dummy: **{linha['melhoria_vs_dummy_pct']:.1f}%**.",
        f"- Zona com maior MAE observado: **Z{zona_pior['grupo']}**, com {zona_pior['n']} amostras.",
        "",
        "## Features com maior importância por permutação",
        "",
        *[
            f"- `{row.feature}`: {row.importancia_media_s_km:.2f} s/km"
            for row in importancias.head(3).itertuples()
        ],
        "",
        "Estas associações são descritivas e não demonstram causalidade.",
    ]
    (METRICAS / "diagnostico_modelo.md").write_text("\n".join(diagnostico) + "\n", encoding="utf-8")
    print("Três features mais importantes:")
    print(importancias.head(3)[["feature", "importancia_media_s_km"]].to_string(index=False))
    print(f"Curva de aprendizagem: 2 a {int(curva['n_atividades_treino'].max())} atividades de treino")


if __name__ == "__main__":
    main()
