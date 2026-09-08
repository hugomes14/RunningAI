"""Compara janelas cardíacas sem alterar os artefactos de produção."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut

try:
    from . import preparar_dados as preparacao
    from .treinar_modelos import (
        ALVO,
        FEATURES,
        _ajustar,
        calcular_metricas,
        calcular_pesos,
        criar_modelos,
    )
except ImportError:  # Permite executar diretamente: python analisar_sensibilidade_estabilidade.py
    import preparar_dados as preparacao
    from treinar_modelos import (
        ALVO,
        FEATURES,
        _ajustar,
        calcular_metricas,
        calcular_pesos,
        criar_modelos,
    )


RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "Dados" / "processados"
METRICAS = RAIZ / "artefactos" / "metricas"
GRAFICOS = RAIZ / "artefactos" / "graficos"
CONFIGURACOES = [
    {"janela_s": 60.0, "cobertura_s": 45.0, "amostras_min": 6},
    {"janela_s": 40.0, "cobertura_s": 30.0, "amostras_min": 4},
    {"janela_s": 30.0, "cobertura_s": 20.0, "amostras_min": 3},
]


def construir_amostras(registos: pd.DataFrame, zonas: dict, configuracao: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstrói amostras em memória para uma configuração de janela."""
    preparacao.JANELA_FC_S = configuracao["janela_s"]
    preparacao.COBERTURA_FC_MIN_S = configuracao["cobertura_s"]
    preparacao.AMOSTRAS_FC_MIN = configuracao["amostras_min"]
    trocos: list[pd.DataFrame] = []
    intervalos: list[pd.DataFrame] = []
    for atividade_id, atividade in registos.groupby("atividade_id", sort=False):
        perfil = preparacao.perfil_percurso(atividade)
        perfil.insert(0, "atividade_id", str(atividade_id))
        bloco_intervalos = preparacao.criar_intervalos(atividade)
        bloco_intervalos["atividade_id"] = str(atividade_id)
        trocos.append(perfil)
        intervalos.append(bloco_intervalos)
    intervalos_df = pd.concat(intervalos, ignore_index=True)
    amostras, grupos = preparacao.agregar_amostras(
        intervalos_df, pd.concat(trocos, ignore_index=True), zonas
    )
    return preparacao.calcular_pesos_atividade(amostras), intervalos_df


def avaliar_gradient_boosting(amostras: pd.DataFrame) -> dict[str, float]:
    """Calcula métricas macro LOGO sem guardar modelos de sensibilidade."""
    grupos = amostras["atividade_id"].astype(str)
    logo = LeaveOneGroupOut()
    modelo_base = criar_modelos()["GradientBoosting"]
    resultados: list[dict[str, float]] = []
    for treino_idx, teste_idx in logo.split(amostras[FEATURES], amostras[ALVO], grupos):
        treino = amostras.iloc[treino_idx]
        teste = amostras.iloc[teste_idx]
        modelo = _ajustar(
            clone(modelo_base), treino[FEATURES], treino[ALVO], calcular_pesos(treino)
        )
        resultados.append(calcular_metricas(teste[ALVO].to_numpy(), modelo.predict(teste[FEATURES])))
    tabela = pd.DataFrame(resultados)
    return {
        "mae_macro_s_km": float(tabela["mae_s_km"].mean()),
        "rmse_macro_s_km": float(tabela["rmse_s_km"].mean()),
        "r2_macro": float(tabela["r2"].mean()),
    }


def main() -> None:
    """Executa a análise e guarda tabela e gráfico comparativos."""
    METRICAS.mkdir(parents=True, exist_ok=True)
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    registos = pd.read_csv(DADOS / "registos_normalizados.csv", parse_dates=["timestamp"])
    zonas = json.loads((RAIZ / "configuracao" / "zonas_fc.json").read_text(encoding="utf-8"))
    resultados: list[dict[str, float | int]] = []
    for configuracao in CONFIGURACOES:
        amostras, intervalos = construir_amostras(registos, zonas, configuracao)
        metricas = avaliar_gradient_boosting(amostras)
        resultados.append(
            {
                **configuracao,
                "n_atividades": int(amostras["atividade_id"].nunique()),
                "n_amostras": int(len(amostras)),
                "intervalos_validos_pct": float(100 * intervalos["valido"].mean()),
                **metricas,
            }
        )
        print(resultados[-1])
    tabela = pd.DataFrame(resultados)
    tabela.to_csv(METRICAS / "sensibilidade_janela_fc.csv", index=False)
    fig, eixo_erro = plt.subplots(figsize=(9, 5.5))
    eixo_amostras = eixo_erro.twinx()
    eixo_erro.plot(tabela["janela_s"], tabela["mae_macro_s_km"], "o-", color="#C44E52", label="MAE")
    eixo_amostras.plot(tabela["janela_s"], tabela["n_amostras"], "s-", color="#2878B5", label="Amostras")
    eixo_erro.set(xlabel="Janela cardíaca (s)", ylabel="MAE macro LOGO (s/km)", title="Sensibilidade à duração da janela cardíaca")
    eixo_amostras.set_ylabel("Número de amostras")
    eixo_erro.invert_xaxis()
    linhas = eixo_erro.lines + eixo_amostras.lines
    eixo_erro.legend(linhas, [linha.get_label() for linha in linhas], loc="best")
    fig.tight_layout()
    fig.savefig(GRAFICOS / "sensibilidade_janela_fc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
