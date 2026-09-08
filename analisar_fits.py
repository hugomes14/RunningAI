"""Análise exploratória inicial dos ficheiros FIT de corrida."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from garmin_fit_sdk import Decoder, Stream


RAIZ = Path(__file__).resolve().parent
DADOS_BRUTOS = RAIZ / "Dados" / "brutos"
DADOS_PROCESSADOS = RAIZ / "Dados" / "processados"
ANALISE = RAIZ / "artefactos" / "analise_inicial"
GRAFICOS = RAIZ / "artefactos" / "graficos"
LOGS = RAIZ / "artefactos" / "logs"


def configurar_logging() -> None:
    """Configura mensagens para consola e ficheiro."""
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "analise_fits.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def ler_fit(caminho: Path) -> dict[str, list[dict[str, Any]]]:
    """Valida e descodifica um FIT com streams independentes."""
    if not caminho.is_file():
        raise FileNotFoundError(f"Ficheiro FIT inexistente: {caminho}")
    if not Decoder(Stream.from_file(str(caminho))).check_integrity():
        raise ValueError(f"Falha na integridade FIT: {caminho.name}")
    mensagens, erros = Decoder(Stream.from_file(str(caminho))).read()
    if erros:
        raise ValueError(f"Erros ao descodificar {caminho.name}: {erros}")
    return mensagens


def normalizar_registos(
    mensagens: dict[str, list[dict[str, Any]]], atividade_id: str
) -> pd.DataFrame:
    """Converte mensagens record para o esquema canónico do projeto."""
    registos = mensagens.get("record_mesgs", [])
    linhas: list[dict[str, Any]] = []
    usou_altitude_normal = False
    usou_velocidade_normal = False
    for registo in registos:
        altitude = registo.get("enhanced_altitude")
        if altitude is None:
            altitude = registo.get("altitude")
            usou_altitude_normal = usou_altitude_normal or altitude is not None
        velocidade = registo.get("enhanced_speed")
        if velocidade is None:
            velocidade = registo.get("speed")
            usou_velocidade_normal = usou_velocidade_normal or velocidade is not None
        linhas.append(
            {
                "atividade_id": atividade_id,
                "timestamp": registo.get("timestamp"),
                "distancia_m": registo.get("distance"),
                "altitude_m": altitude,
                "fc_bpm": registo.get("heart_rate"),
                "potencia_w": registo.get("power"),
                "velocidade_fit_m_s": velocidade,
            }
        )
    if usou_altitude_normal:
        logging.info("%s: fallback para altitude normal", atividade_id)
    if usou_velocidade_normal:
        logging.info("%s: fallback para velocidade normal", atividade_id)
    df = pd.DataFrame(linhas)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "atividade_id",
                "timestamp",
                "distancia_m",
                "altitude_m",
                "fc_bpm",
                "potencia_w",
                "velocidade_fit_m_s",
            ]
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for coluna in ["distancia_m", "altitude_m", "fc_bpm", "potencia_w", "velocidade_fit_m_s"]:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
    return df


def validar_atividade(mensagens: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Valida desporto e número de sessões."""
    sessoes = mensagens.get("session_mesgs", [])
    desportos = {
        str(s.get("sport", "")).lower() for s in sessoes
    } | {str(s.get("sport", "")).lower() for s in mensagens.get("sport_mesgs", [])}
    corrida = any("running" in desporto or desporto == "run" for desporto in desportos)
    motivos: list[str] = []
    if not corrida:
        motivos.append(f"desporto não reconhecido como corrida: {sorted(desportos)}")
    if len(sessoes) != 1:
        motivos.append(f"número de sessões diferente de 1: {len(sessoes)}")
    if not mensagens.get("record_mesgs"):
        motivos.append("sem mensagens record")
    return {"valida": not motivos, "motivos": motivos, "n_sessoes": len(sessoes)}


def _ritmo_intervalos(df: pd.DataFrame) -> pd.Series:
    """Calcula ritmo instantâneo auditável a partir de distância e tempo."""
    ordenado = df.sort_values("timestamp")
    dt = ordenado["timestamp"].diff().dt.total_seconds()
    dx = ordenado["distancia_m"].diff()
    velocidade = dx / dt
    return (1000.0 / velocidade).where((dt > 0) & (dx > 0) & np.isfinite(velocidade))


def resumir_atividade(df: pd.DataFrame) -> dict[str, Any]:
    """Produz estatísticas e indicadores de qualidade de uma atividade."""
    ordenado = df.sort_values("timestamp").copy()
    dt = ordenado["timestamp"].diff().dt.total_seconds()
    dx = ordenado["distancia_m"].diff()
    velocidade = dx / dt
    movimento = (dt > 0) & (dt <= 3) & (dx > 0) & (velocidade >= 0.5)
    altitude = ordenado["altitude_m"].interpolate(limit_direction="both")
    subida = altitude.diff().clip(lower=0).sum()
    distancia = float(ordenado["distancia_m"].max() - ordenado["distancia_m"].min())
    duracao = (ordenado["timestamp"].max() - ordenado["timestamp"].min()).total_seconds()
    tempo_movimento = float(dt.where(movimento).sum())
    ritmo_medio = 1000.0 * tempo_movimento / float(dx.where(movimento).sum()) if dx.where(movimento).sum() > 0 else np.nan
    return {
        "atividade_id": str(ordenado["atividade_id"].iloc[0]),
        "data_inicio": ordenado["timestamp"].min().isoformat(),
        "n_registos": len(ordenado),
        "distancia_m": distancia,
        "duracao_s": duracao,
        "tempo_movimento_s": tempo_movimento,
        "subida_aproximada_m": float(subida),
        "ritmo_medio_s_km": ritmo_medio,
        "fc_media_bpm": float(ordenado["fc_bpm"].mean()),
        "intervalo_mediano_s": float(dt.dropna().median()),
        "intervalo_max_s": float(dt.dropna().max()),
        "pct_altitude_ausente": float(100 * ordenado["altitude_m"].isna().mean()),
        "pct_fc_ausente": float(100 * ordenado["fc_bpm"].isna().mean()),
        "pct_velocidade_ausente": float(100 * ordenado["velocidade_fit_m_s"].isna().mean()),
        "timestamps_duplicados": int(ordenado["timestamp"].duplicated().sum()),
        "recuos_distancia": int((dx < 0).sum()),
        "saltos_temporais_gt3": int((dt > 3).sum()),
        "saltos_distancia": int((dx > 10 * dt).sum()),
        "fc_fora_30_240": int(((ordenado["fc_bpm"] < 30) | (ordenado["fc_bpm"] > 240)).sum()),
    }


def visualizar_atividade(df: pd.DataFrame, pasta_saida: Path) -> None:
    """Guarda altitude, ritmo e FC ao longo da distância."""
    pasta_saida.mkdir(parents=True, exist_ok=True)
    ordenado = df.sort_values("timestamp").copy()
    ordenado["distancia_km"] = ordenado["distancia_m"] / 1000
    ordenado["ritmo_s_km"] = _ritmo_intervalos(ordenado)
    atividade = str(ordenado["atividade_id"].iloc[0])
    fig, eixos = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    eixos[0].plot(ordenado["distancia_km"], ordenado["altitude_m"], color="#5B8C5A")
    eixos[0].set_ylabel("Altitude (m)")
    ritmo = ordenado["ritmo_s_km"].where(ordenado["ritmo_s_km"].between(120, 1800))
    eixos[1].plot(ordenado["distancia_km"], ritmo, color="#2878B5", alpha=0.8)
    eixos[1].set_ylabel("Ritmo (s/km)")
    eixos[2].plot(ordenado["distancia_km"], ordenado["fc_bpm"], color="#C44E52")
    eixos[2].set_ylabel("FC (bpm)")
    eixos[2].set_xlabel("Distância (km)")
    fig.suptitle(f"Perfil da atividade {atividade}")
    fig.tight_layout()
    fig.savefig(pasta_saida / f"atividade_{atividade}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def visualizar_qualidade(resumo: pd.DataFrame, pasta_saida: Path) -> None:
    """Guarda um painel com ausências e anomalias por atividade."""
    pasta_saida.mkdir(parents=True, exist_ok=True)
    fig, eixos = plt.subplots(1, 2, figsize=(15, 5))
    resumo.set_index("atividade_id")[[
        "pct_altitude_ausente", "pct_fc_ausente", "pct_velocidade_ausente"
    ]].plot(kind="bar", ax=eixos[0])
    eixos[0].set_ylabel("Ausência (%)")
    eixos[0].set_title("Cobertura dos campos")
    resumo.set_index("atividade_id")[[
        "timestamps_duplicados", "recuos_distancia", "saltos_temporais_gt3", "saltos_distancia"
    ]].plot(kind="bar", ax=eixos[1])
    eixos[1].set_ylabel("Contagem")
    eixos[1].set_title("Problemas de sequência")
    for eixo in eixos:
        eixo.tick_params(axis="x", rotation=45)
        eixo.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pasta_saida / "qualidade_dados.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def visualizar_conjunto(registos: pd.DataFrame, pasta_saida: Path) -> None:
    """Guarda perfis conjuntos e histogramas globais."""
    fig, eixos = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    for atividade, grupo in registos.groupby("atividade_id", sort=False):
        g = grupo.sort_values("timestamp").copy()
        x = g["distancia_m"] / 1000
        eixos[0].plot(x, g["altitude_m"], alpha=0.7, label=atividade)
        eixos[1].plot(x, _ritmo_intervalos(g).where(lambda s: s.between(120, 1800)), alpha=0.6)
        eixos[2].plot(x, g["fc_bpm"], alpha=0.6)
    eixos[0].legend(ncol=2, fontsize=7)
    eixos[0].set_ylabel("Altitude (m)")
    eixos[1].set_ylabel("Ritmo (s/km)")
    eixos[2].set_ylabel("FC (bpm)")
    eixos[2].set_xlabel("Distância (km)")
    fig.suptitle("Perfis das atividades")
    fig.tight_layout()
    fig.savefig(pasta_saida / "perfil_atividades.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    ritmos = pd.concat([_ritmo_intervalos(g) for _, g in registos.groupby("atividade_id")])
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4))
    eixos[0].hist(registos["fc_bpm"].dropna(), bins=40, color="#C44E52", alpha=0.8)
    eixos[0].set(xlabel="FC (bpm)", ylabel="Registos", title="Distribuição global da FC")
    eixos[1].hist(ritmos[ritmos.between(120, 1800)].dropna(), bins=50, color="#2878B5", alpha=0.8)
    eixos[1].set(xlabel="Ritmo (s/km)", ylabel="Intervalos", title="Distribuição global do ritmo")
    fig.tight_layout()
    fig.savefig(pasta_saida / "histogramas_globais.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Executa a análise exploratória inicial."""
    configurar_logging()
    DADOS_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    ANALISE.mkdir(parents=True, exist_ok=True)
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    caminhos = sorted(DADOS_BRUTOS.glob("*.fit"))
    if not caminhos:
        raise FileNotFoundError(f"Nenhum FIT encontrado em {DADOS_BRUTOS}")
    registos_validos: list[pd.DataFrame] = []
    resumos: list[dict[str, Any]] = []
    excluidas: list[dict[str, str]] = []
    for caminho in caminhos:
        try:
            mensagens = ler_fit(caminho)
            validacao = validar_atividade(mensagens)
            if not validacao["valida"]:
                excluidas.append({"ficheiro": caminho.name, "motivo": "; ".join(validacao["motivos"])})
                continue
            atividade_id = caminho.stem.replace("_ACTIVITY", "")
            df = normalizar_registos(mensagens, atividade_id)
            if df.empty:
                excluidas.append({"ficheiro": caminho.name, "motivo": "sem registos normalizados"})
                continue
            registos_validos.append(df)
            resumos.append(resumir_atividade(df))
            visualizar_atividade(df, GRAFICOS)
            logging.info("Atividade %s: %d registos", atividade_id, len(df))
        except Exception as erro:  # mantém o lote e documenta a falha individual
            logging.exception("Falha em %s", caminho.name)
            excluidas.append({"ficheiro": caminho.name, "motivo": str(erro)})
    if not registos_validos:
        raise RuntimeError("Nenhuma atividade válida após a auditoria FIT")
    registos = pd.concat(registos_validos, ignore_index=True)
    resumo = pd.DataFrame(resumos).sort_values("data_inicio")
    registos.to_csv(DADOS_PROCESSADOS / "registos_normalizados.csv", index=False)
    resumo.to_csv(ANALISE / "resumo_atividades.csv", index=False)
    pd.DataFrame(excluidas, columns=["ficheiro", "motivo"]).to_csv(
        ANALISE / "atividades_excluidas.csv", index=False
    )
    visualizar_qualidade(resumo, GRAFICOS)
    visualizar_conjunto(registos, GRAFICOS)
    print(f"Atividades processadas: {len(resumo)}")
    print(f"Atividades excluídas: {len(excluidas)}")
    print(resumo[["atividade_id", "distancia_m", "ritmo_medio_s_km", "fc_media_bpm"]].to_string(index=False))
    if excluidas:
        print(pd.DataFrame(excluidas).to_string(index=False))


if __name__ == "__main__":
    main()
