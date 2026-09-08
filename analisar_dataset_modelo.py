"""Análise exploratória do dataset final de modelação."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "Dados" / "processados"
ANALISE = RAIZ / "artefactos" / "analise_dataset"
GRAFICOS = RAIZ / "artefactos" / "graficos"
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


def validar_dataset(df: pd.DataFrame, manifesto: dict[str, Any]) -> None:
    """Valida colunas, valores e coerência básica do manifesto."""
    esperadas = ["amostra_id", "atividade_id", "troco_id", *FEATURES, ALVO, "peso_amostra"]
    ausentes = [coluna for coluna in esperadas if coluna not in df.columns]
    if ausentes:
        raise ValueError(f"Colunas em falta: {ausentes}")
    if manifesto.get("features") != FEATURES:
        raise ValueError("A ordem das features no manifesto não corresponde ao contrato")
    if int(manifesto.get("n_amostras", -1)) != len(df):
        raise ValueError("O número de amostras difere do manifesto; volte a processar")
    if df[FEATURES + [ALVO]].isna().any().any():
        raise ValueError("Existem valores em falta nas features ou no alvo")
    if not np.isfinite(df[FEATURES + [ALVO]].to_numpy(dtype=float)).all():
        raise ValueError("Existem valores não finitos nas features ou no alvo")
    if df["amostra_id"].duplicated().any():
        raise ValueError("Existem amostra_id duplicados")


def analisar_distribuicoes(df: pd.DataFrame, pasta_saida: Path) -> pd.DataFrame:
    """Guarda estatísticas e histogramas das variáveis principais."""
    colunas = [ALVO, "declive_pct", "altitude_normalizada", "distancia_inicio_km", "subida_acumulada_m"]
    descritivas = df[colunas].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    descritivas["assimetria"] = df[colunas].skew()
    descritivas.to_csv(ANALISE / "estatisticas_distribuicoes.csv")
    fig, eixos = plt.subplots(2, 3, figsize=(16, 9))
    titulos = {
        ALVO: "Ritmo alvo (s/km)",
        "declive_pct": "Declive (%)",
        "altitude_normalizada": "Altitude normalizada (m/1000)",
        "distancia_inicio_km": "Distância inicial (km)",
        "subida_acumulada_m": "Subida acumulada (m)",
    }
    for eixo, coluna in zip(eixos.flat, colunas):
        sns.histplot(df[coluna], bins=35, kde=True, ax=eixo, color="#2878B5")
        eixo.set_title(titulos[coluna])
    eixos.flat[-1].axis("off")
    fig.suptitle("Distribuições do dataset de modelação")
    fig.tight_layout()
    fig.savefig(pasta_saida / "distribuicoes_dataset.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return descritivas


def analisar_por_zona(df: pd.DataFrame, pasta_saida: Path) -> pd.DataFrame:
    """Analisa cobertura e ritmo por zona e atividade."""
    cobertura = pd.crosstab(df["atividade_id"], df["zona_ordem"])
    cobertura.to_csv(ANALISE / "amostras_por_atividade_zona.csv")
    fig, eixo = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="zona_ordem", y=ALVO, color="#72B7B2", ax=eixo, showfliers=False)
    sns.stripplot(data=df, x="zona_ordem", y=ALVO, color="black", alpha=0.18, size=2, ax=eixo)
    eixo.set(xlabel="Zona cardíaca", ylabel="Ritmo (s/km)", title="Ritmo por zona cardíaca")
    fig.tight_layout()
    fig.savefig(pasta_saida / "ritmo_por_zona.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return cobertura


def analisar_relacoes(df: pd.DataFrame, pasta_saida: Path) -> pd.DataFrame:
    """Guarda relação ritmo/declive e correlações numéricas."""
    fig, eixo = plt.subplots(figsize=(11, 7))
    sns.scatterplot(
        data=df,
        x="declive_pct",
        y=ALVO,
        hue="zona_ordem",
        palette="viridis",
        alpha=0.65,
        s=28,
        ax=eixo,
    )
    eixo.set(xlabel="Declive (%)", ylabel="Ritmo (s/km)", title="Ritmo, declive e zona cardíaca")
    fig.tight_layout()
    fig.savefig(pasta_saida / "ritmo_declive_zona.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    correlacoes = df[FEATURES + [ALVO]].corr(numeric_only=True)
    correlacoes.to_csv(ANALISE / "correlacoes.csv")
    fig, eixo = plt.subplots(figsize=(11, 9))
    sns.heatmap(correlacoes, cmap="vlag", center=0, ax=eixo)
    eixo.set_title("Correlações numéricas")
    fig.tight_layout()
    fig.savefig(pasta_saida / "correlacoes_dataset.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return correlacoes


def detetar_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Marca candidatos a outlier de ritmo ou declive pelo critério IQR."""
    marcas: list[pd.DataFrame] = []
    for coluna in [ALVO, "declive_pct"]:
        q1, q3 = df[coluna].quantile([0.25, 0.75])
        iqr = q3 - q1
        inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        candidatos = df[(df[coluna] < inferior) | (df[coluna] > superior)][
            ["amostra_id", "atividade_id", "troco_id", "zona_ordem", coluna]
        ].copy()
        candidatos["variavel"] = coluna
        candidatos["limite_inferior"] = inferior
        candidatos["limite_superior"] = superior
        candidatos["razao"] = "fora de 1,5 × IQR"
        candidatos = candidatos.rename(columns={coluna: "valor"})
        marcas.append(candidatos)
    return pd.concat(marcas, ignore_index=True)


def criar_resumo_eda(
    df: pd.DataFrame,
    descritivas: pd.DataFrame,
    cobertura: pd.DataFrame,
    correlacoes: pd.DataFrame,
    outliers: pd.DataFrame,
) -> None:
    """Escreve um resumo Markdown calculado a partir dos dados."""
    atividades = df["atividade_id"].nunique()
    zonas = sorted(df["zona_ordem"].unique().astype(int).tolist())
    cobertura_zonas = df.groupby("zona_ordem").size().to_dict()
    pares = []
    for i, coluna_a in enumerate(FEATURES):
        for coluna_b in FEATURES[i + 1 :]:
            valor = correlacoes.loc[coluna_a, coluna_b]
            if abs(valor) >= 0.9:
                pares.append(f"- `{coluna_a}` / `{coluna_b}`: {valor:.3f}")
    texto = [
        "# Resumo da análise exploratória do dataset",
        "",
        f"- Amostras: **{len(df)}**",
        f"- Atividades: **{atividades}**",
        f"- Zonas representadas: **{', '.join(f'Z{z}' for z in zonas)}**",
        f"- Candidatos a outlier por IQR: **{len(outliers)}**",
        f"- Ritmo mediano: **{df[ALVO].median():.1f} s/km**",
        f"- Declive entre p05 e p95: **{df['declive_pct'].quantile(.05):.1f}% a {df['declive_pct'].quantile(.95):.1f}%**",
        "",
        "## Cobertura por zona",
        "",
        *[f"- Z{int(zona)}: {int(n)} amostras" for zona, n in cobertura_zonas.items()],
        "",
        "## Colinearidade elevada",
        "",
        *(pares or ["- Não foram encontrados pares com |r| ≥ 0,90."]),
        "",
        "Os outliers foram identificados e preservados no dataset. A cobertura desigual entre zonas e atividades deve ser considerada na interpretação da validação.",
    ]
    (ANALISE / "resumo_eda.md").write_text("\n".join(texto), encoding="utf-8")
    descritivas.to_csv(ANALISE / "estatisticas_distribuicoes.csv")
    cobertura.to_csv(ANALISE / "cobertura_atividade_zona.csv")


def main() -> None:
    """Executa a análise exploratória do dataset de modelação."""
    ANALISE.mkdir(parents=True, exist_ok=True)
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    caminho_dataset = DADOS / "amostras_modelo.csv"
    caminho_manifesto = DADOS / "manifesto_processamento.json"
    if not caminho_dataset.exists() or not caminho_manifesto.exists():
        raise FileNotFoundError("Execute preparar_dados.py antes desta etapa")
    df = pd.read_csv(caminho_dataset, dtype={"atividade_id": str})
    manifesto = json.loads(caminho_manifesto.read_text(encoding="utf-8"))
    validar_dataset(df, manifesto)
    descritivas = analisar_distribuicoes(df, GRAFICOS)
    cobertura = analisar_por_zona(df, GRAFICOS)
    correlacoes = analisar_relacoes(df, GRAFICOS)
    outliers = detetar_outliers(df)
    outliers.to_csv(ANALISE / "outliers_iqr.csv", index=False)
    criar_resumo_eda(df, descritivas, cobertura, correlacoes, outliers)
    print(f"Amostras: {len(df)}")
    print(f"Atividades: {df['atividade_id'].nunique()}")
    print(f"Zonas cobertas: {sorted(df['zona_ordem'].unique().astype(int).tolist())}")
    print(f"Outliers IQR detetados: {len(outliers)} (preservados)")


if __name__ == "__main__":
    main()
