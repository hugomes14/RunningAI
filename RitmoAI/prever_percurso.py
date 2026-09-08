"""Previsão de ritmo por troço para uma ou todas as zonas cardíacas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .analisar_fits import ler_fit, normalizar_registos
    from .preparar_dados import FEATURES, perfil_percurso
except ImportError:  # Permite executar diretamente: python prever_percurso.py
    from analisar_fits import ler_fit, normalizar_registos
    from preparar_dados import FEATURES, perfil_percurso


RAIZ = Path(__file__).resolve().parent
MODELO_PADRAO = RAIZ / "artefactos" / "modelos" / "final" / "modelo_ritmo_final.joblib"
DADOS_TREINO = RAIZ / "Dados" / "processados" / "amostras_modelo.csv"
SAIDA_PADRAO = RAIZ / "artefactos" / "previsoes_percurso"
GRAFICOS = RAIZ / "artefactos" / "graficos"


def carregar_percurso(caminho: Path) -> pd.DataFrame:
    """Lê apenas distância e altitude de FIT ou CSV."""
    if not caminho.exists():
        raise FileNotFoundError(f"Percurso inexistente: {caminho}")
    if caminho.suffix.lower() == ".fit":
        mensagens = ler_fit(caminho)
        df = normalizar_registos(mensagens, caminho.stem)
    elif caminho.suffix.lower() == ".csv":
        df = pd.read_csv(caminho)
    else:
        raise ValueError("O percurso deve ser FIT ou CSV")
    ausentes = [coluna for coluna in ["distancia_m", "altitude_m"] if coluna not in df]
    if ausentes:
        raise ValueError(f"Colunas do percurso em falta: {ausentes}")
    return df[["distancia_m", "altitude_m"]].copy()


def reconstruir_features(
    perfil: pd.DataFrame, zonas: dict[str, Any], zonas_pedidas: list[int]
) -> pd.DataFrame:
    """Replica as nove features conhecidas antes da corrida."""
    linhas: list[pd.DataFrame] = []
    mapa = {int(z["zona"]): z for z in zonas["zonas"]}
    for zona_id in zonas_pedidas:
        if zona_id not in mapa:
            raise ValueError(f"Zona inválida: Z{zona_id}")
        zona = mapa[zona_id]
        bloco = perfil.copy()
        bloco["zona_ordem"] = zona_id
        bloco["zona_limite_inferior_bpm"] = zona["limite_inferior_bpm"]
        bloco["zona_limite_superior_bpm"] = zona["limite_superior_bpm"]
        bloco["zona_aberta"] = zona.get("zona_aberta", 0)
        linhas.append(bloco)
    resultado = pd.concat(linhas, ignore_index=True)
    faltam = [feature for feature in FEATURES if feature not in resultado]
    if faltam:
        raise ValueError(f"Não foi possível reconstruir as features: {faltam}")
    return resultado


def detetar_pouco_suporte(
    linha: pd.Series, treino: pd.DataFrame
) -> tuple[bool, int, int]:
    """Sinaliza regiões pouco cobertas por atividades históricas."""
    candidatos = treino[
        (treino["zona_ordem"] == linha["zona_ordem"])
        & ((treino["declive_pct"] - linha["declive_pct"]).abs() <= 2.0)
        & ((treino["altitude_normalizada"] - linha["altitude_normalizada"]).abs() <= 0.15)
        & ((treino["distancia_inicio_km"] - linha["distancia_inicio_km"]).abs() <= 3.0)
    ]
    n = len(candidatos)
    atividades = candidatos["atividade_id"].astype(str).nunique()
    return n < 5 or atividades < 2, n, atividades


def prever(
    pacote: dict[str, Any], features: pd.DataFrame, treino: pd.DataFrame
) -> pd.DataFrame:
    """Prevê, valida a gama e acrescenta indicadores de suporte."""
    ordem = pacote["metadados"]["features"]
    if ordem != FEATURES:
        raise ValueError("A ordem de features do pacote difere da implementação")
    resultado = features.copy()
    resultado["ritmo_previsto_s_km"] = pacote["modelo"].predict(resultado[ordem])
    gama = pacote["metadados"]["gamas_treino"]["ritmo_alvo_s_km"]
    limite_inferior = 0.5 * float(gama["min"])
    limite_superior = 1.5 * float(gama["max"])
    resultado["previsao_valida"] = (
        np.isfinite(resultado["ritmo_previsto_s_km"])
        & resultado["ritmo_previsto_s_km"].between(limite_inferior, limite_superior)
        & (resultado["ritmo_previsto_s_km"] > 0)
    )
    suporte = resultado.apply(lambda linha: detetar_pouco_suporte(linha, treino), axis=1)
    resultado[["pouco_suporte", "n_amostras_suporte", "n_atividades_suporte"]] = pd.DataFrame(
        suporte.tolist(), index=resultado.index
    )
    resultado["ritmo_previsto_mmss_km"] = resultado["ritmo_previsto_s_km"].map(formatar_ritmo)
    resultado["tempo_troco_previsto_s"] = (
        resultado["ritmo_previsto_s_km"] * resultado["comprimento_troco_m"] / 1000.0
    ).where(resultado["previsao_valida"])
    return resultado


def formatar_ritmo(valor: float) -> str:
    """Formata segundos por quilómetro como mm:ss/km."""
    if not np.isfinite(valor) or valor <= 0:
        return "inválido"
    minutos = int(valor // 60)
    segundos = int(round(valor - 60 * minutos))
    if segundos == 60:
        minutos += 1
        segundos = 0
    return f"{minutos:02d}:{segundos:02d}/km"


def criar_resumo(resultado: pd.DataFrame, percurso: Path, modelo: Path) -> dict[str, Any]:
    """Resume distância, tempo e suporte por zona."""
    zonas: dict[str, Any] = {}
    for zona, bloco in resultado.groupby("zona_ordem"):
        validas = bloco[bloco["previsao_valida"]]
        distancia = float(validas["comprimento_troco_m"].sum())
        tempo = float(validas["tempo_troco_previsto_s"].sum())
        ritmo = 1000 * tempo / distancia if distancia > 0 else np.nan
        zonas[f"Z{int(zona)}"] = {
            "distancia_valida_m": distancia,
            "tempo_estimado_sem_paragens_s": tempo,
            "ritmo_medio_s_km": ritmo if np.isfinite(ritmo) else None,
            "ritmo_medio_mmss_km": formatar_ritmo(ritmo),
            "distancia_pouco_suporte_m": float(bloco.loc[bloco["pouco_suporte"], "comprimento_troco_m"].sum()),
            "trocos_invalidos": int((~bloco["previsao_valida"]).sum()),
        }
    return {
        "percurso": str(percurso),
        "modelo": str(modelo),
        "zonas": zonas,
        "aviso": "A previsão assume permanência na zona escolhida e não simula a resposta cardíaca real durante o esforço.",
    }


def criar_grafico(resultado: pd.DataFrame, caminho: Path) -> None:
    """Mostra altitude, declive e ritmo ao longo do percurso."""
    fig, eixos = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    base = resultado.drop_duplicates("troco_id").sort_values("troco_id")
    x_base = base["distancia_inicio_km"]
    eixos[0].plot(x_base, base["altitude_media_m"], color="#5B8C5A")
    eixos[0].set_ylabel("Altitude (m)")
    eixos[1].plot(x_base, base["declive_pct"], color="#8C564B")
    eixos[1].axhline(0, color="black", linewidth=0.7)
    eixos[1].set_ylabel("Declive (%)")
    for zona, bloco in resultado.groupby("zona_ordem"):
        bloco = bloco.sort_values("troco_id")
        eixos[2].plot(bloco["distancia_inicio_km"], bloco["ritmo_previsto_s_km"] / 60, label=f"Z{int(zona)}")
        pouco = bloco[bloco["pouco_suporte"]]
        eixos[2].scatter(pouco["distancia_inicio_km"], pouco["ritmo_previsto_s_km"] / 60, marker="x", s=20)
    eixos[2].invert_yaxis()
    eixos[2].set(xlabel="Distância (km)", ylabel="Ritmo (min/km)")
    eixos[2].legend(ncol=5)
    fig.suptitle("Previsão de ritmo por percurso e zona")
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Interpreta argumentos e grava previsão, resumo e gráfico."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", type=Path, default=MODELO_PADRAO)
    parser.add_argument("--percurso", type=Path, required=True)
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--zona", type=int, choices=range(1, 6))
    grupo.add_argument("--todas-zonas", action="store_true")
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    args = parser.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    pacote = joblib.load(args.modelo)
    percurso = carregar_percurso(args.percurso)
    perfil = perfil_percurso(percurso)
    zonas_pedidas = list(range(1, 6)) if args.todas_zonas else [args.zona]
    features = reconstruir_features(perfil, pacote["metadados"]["zonas"], zonas_pedidas)
    treino = pd.read_csv(DADOS_TREINO, dtype={"atividade_id": str})
    resultado = prever(pacote, features, treino)
    nome = args.percurso.stem.replace("_ACTIVITY", "")
    csv_path = args.saida / f"previsao_{nome}.csv"
    json_path = args.saida / f"previsao_{nome}.json"
    png_path = GRAFICOS / "previsao_percurso.png"
    resultado.to_csv(csv_path, index=False)
    resumo = criar_resumo(resultado, args.percurso, args.modelo)
    json_path.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    criar_grafico(resultado, png_path)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print("AVISO:", resumo["aviso"])


if __name__ == "__main__":
    main()
