"""Criação do dataset de ritmo por percurso e zona cardíaca."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parent
DADOS_BRUTOS = RAIZ / "Dados" / "brutos"
DADOS_PROCESSADOS = RAIZ / "Dados" / "processados"
CONFIGURACAO = RAIZ / "configuracao" / "zonas_fc.json"
PASSO_GRELHA_M = 10.0
COMPRIMENTO_TROCO_M = 50.0
JANELA_FC_S = 40.0
COBERTURA_FC_MIN_S = 30.0
AMOSTRAS_FC_MIN = 4
SALTO_TEMPORAL_MAX_S = 15.0
AMPLITUDE_ROBUSTA_FC_MAX_BPM = 15.0
DIFERENCA_MEDIA_FC_MAX_BPM = 8.0
TEMPO_INICIAL_EXCLUIDO_S = 120.0
INTERVALOS_MIN_POR_TROCO = 2
COBERTURA_TROCO_MIN = 0.60
COBERTURA_TROCO_MAX = 1.20
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


def carregar_ou_criar_zonas(
    caminho_json: Path, idade: int | None, limites_fc: list[int] | None
) -> dict[str, Any]:
    """Carrega zonas, ou cria configuração explícita a partir de argumentos."""
    caminho_json.parent.mkdir(parents=True, exist_ok=True)
    if limites_fc is not None:
        limites = [int(v) for v in limites_fc]
        origem = "Limites pessoais fornecidos por --limites-fc"
        idade_guardada = idade
    elif caminho_json.exists():
        with caminho_json.open(encoding="utf-8") as ficheiro:
            return json.load(ficheiro)
    elif idade is not None:
        fc_max = 220 - idade
        limites = [round(fc_max * percentagem) for percentagem in np.linspace(0.5, 1.0, 6)]
        origem = "Estimativa 220 - idade; aproximação sem teste fisiológico"
        idade_guardada = idade
    else:
        modelo = {
            "idade": None,
            "fc_max_estimada": None,
            "limites_fc_bpm": None,
            "zonas": None,
            "origem": None,
            "data_criacao": datetime.now(timezone.utc).date().isoformat(),
        }
        caminho_json.write_text(json.dumps(modelo, ensure_ascii=False, indent=2), encoding="utf-8")
        raise ValueError(
            "Configuração de zonas incompleta. Preencha configuracao/zonas_fc.json "
            "ou use --idade/--limites-fc (seis fronteiras, por exemplo 103 124 144 165 185 206)."
        )
    if len(limites) != 6:
        raise ValueError("--limites-fc exige seis fronteiras para Z1–Z5")
    zonas = []
    for indice in range(5):
        zonas.append(
            {
                "zona": indice + 1,
                "nome": f"Z{indice + 1}",
                "limite_inferior_bpm": limites[indice],
                "limite_superior_bpm": limites[indice + 1],
                "zona_aberta": 0,
            }
        )
    configuracao = {
        "idade": idade_guardada,
        "fc_max_estimada": limites[-1],
        "limites_fc_bpm": limites,
        "zonas": zonas,
        "origem": origem,
        "data_criacao": datetime.now(timezone.utc).date().isoformat(),
        "notas": "Os limites pessoais têm prioridade sobre a fórmula 220 - idade.",
    }
    caminho_json.write_text(json.dumps(configuracao, ensure_ascii=False, indent=2), encoding="utf-8")
    return configuracao


def validar_zonas(zonas: dict[str, Any]) -> None:
    """Valida cinco zonas contíguas, ordenadas e numericamente coerentes."""
    lista = zonas.get("zonas")
    if not isinstance(lista, list) or len(lista) != 5:
        raise ValueError("A configuração deve conter exatamente cinco zonas")
    anterior: float | None = None
    for indice, zona in enumerate(lista, start=1):
        if int(zona.get("zona", -1)) != indice:
            raise ValueError(f"Ordem inválida na zona {indice}")
        inferior = zona.get("limite_inferior_bpm")
        superior = zona.get("limite_superior_bpm")
        if not isinstance(inferior, (int, float)) or not isinstance(superior, (int, float)):
            raise ValueError(f"Limites não numéricos na zona {indice}")
        if inferior >= superior:
            raise ValueError(f"Limites não crescentes na zona {indice}")
        if anterior is not None and inferior != anterior:
            raise ValueError(f"Fronteira descontínua antes da zona {indice}")
        anterior = float(superior)


def classificar_zona(fc_bpm: float, zonas: dict[str, Any]) -> int | None:
    """Classifica uma FC nas zonas Z1–Z5; o limite final é inclusivo."""
    if not np.isfinite(fc_bpm):
        return None
    for indice, zona in enumerate(zonas["zonas"]):
        inferior = float(zona["limite_inferior_bpm"])
        superior = float(zona["limite_superior_bpm"])
        if inferior <= fc_bpm < superior or (indice == 4 and fc_bpm == superior):
            return int(zona["zona"])
    return None


def perfil_percurso(df_atividade: pd.DataFrame) -> pd.DataFrame:
    """Cria features altimétricas a partir do perfil completo conhecido."""
    perfil = df_atividade[["distancia_m", "altitude_m"]].dropna().copy()
    perfil = perfil.sort_values("distancia_m")
    perfil = perfil.groupby("distancia_m", as_index=False)["altitude_m"].median()
    if len(perfil) < 2 or perfil["distancia_m"].max() < COMPRIMENTO_TROCO_M:
        raise ValueError("Percurso insuficiente para construir troços de 50 m")
    distancia_max = float(perfil["distancia_m"].max())
    grelha = np.arange(0.0, distancia_max + PASSO_GRELHA_M, PASSO_GRELHA_M)
    grelha[-1] = max(grelha[-1], distancia_max)
    altitude_interp = np.interp(grelha, perfil["distancia_m"], perfil["altitude_m"])
    altitude_suave = pd.Series(altitude_interp).rolling(5, center=True, min_periods=1).mean().to_numpy()
    fronteiras = np.arange(0.0, distancia_max, COMPRIMENTO_TROCO_M)
    linhas: list[dict[str, float | int]] = []
    subida_acumulada = 0.0
    for troco_id, inicio in enumerate(fronteiras):
        fim = min(inicio + COMPRIMENTO_TROCO_M, distancia_max)
        comprimento = fim - inicio
        alt_inicio = float(np.interp(inicio, grelha, altitude_suave))
        alt_fim = float(np.interp(fim, grelha, altitude_suave))
        diferenca = alt_fim - alt_inicio
        if diferenca > 0:
            subida_acumulada += diferenca
        mascara = (grelha >= inicio) & (grelha <= fim)
        altitude_media = float(np.mean(altitude_suave[mascara])) if mascara.any() else (alt_inicio + alt_fim) / 2
        altitude_normalizada = altitude_media / 1000.0
        linhas.append(
            {
                "troco_id": troco_id,
                "distancia_inicio_m": inicio,
                "distancia_fim_m": fim,
                "comprimento_troco_m": comprimento,
                "altitude_inicio_m": alt_inicio,
                "altitude_fim_m": alt_fim,
                "altitude_media_m": altitude_media,
                "altitude_normalizada": altitude_normalizada,
                "declive_pct": 100.0 * diferenca / comprimento,
                "distancia_inicio_km": inicio / 1000.0,
                "subida_acumulada_m": subida_acumulada,
            }
        )
    return pd.DataFrame(linhas)


def estabilidade_fc(df_atividade: pd.DataFrame) -> pd.DataFrame:
    """Calcula estabilidade causal, independente da frequência de gravação."""
    df = df_atividade.sort_values("timestamp").reset_index(drop=True).copy()
    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    fc = pd.to_numeric(df["fc_bpm"], errors="coerce")
    inicio = timestamps.iloc[0]
    janela: deque[tuple[pd.Timestamp, float]] = deque()
    estavel: list[bool] = []
    motivos: list[str] = []
    anterior: pd.Timestamp | None = None
    for timestamp, valor_fc in zip(timestamps, fc):
        if pd.isna(timestamp) or not np.isfinite(valor_fc) or valor_fc <= 0:
            janela.clear()
            estavel.append(False)
            motivos.append("fc_ou_timestamp_invalido")
            anterior = timestamp if not pd.isna(timestamp) else None
            continue
        if anterior is not None:
            salto = (timestamp - anterior).total_seconds()
            if salto <= 0 or salto > SALTO_TEMPORAL_MAX_S:
                janela.clear()
        anterior = timestamp
        janela.append((timestamp, float(valor_fc)))
        while janela and (timestamp - janela[0][0]).total_seconds() > JANELA_FC_S:
            janela.popleft()
        tempo_inicio = (timestamp - inicio).total_seconds()
        valores = np.array([valor for _, valor in janela], dtype=float)
        cobertura_s = (
            (janela[-1][0] - janela[0][0]).total_seconds() if len(janela) >= 2 else 0.0
        )
        segundos = np.array(
            [(instante - janela[0][0]).total_seconds() for instante, _ in janela],
            dtype=float,
        )
        media_temporal = (
            float(np.trapezoid(valores, segundos) / cobertura_s)
            if cobertura_s > 0
            else float(valor_fc)
        )
        amplitude_robusta = (
            float(np.percentile(valores, 90) - np.percentile(valores, 10))
            if len(valores) >= 2
            else 0.0
        )
        if tempo_inicio < TEMPO_INICIAL_EXCLUIDO_S:
            estavel.append(False)
            motivos.append("primeiros_120s")
        elif cobertura_s < COBERTURA_FC_MIN_S:
            estavel.append(False)
            motivos.append("cobertura_fc_insuficiente")
        elif len(valores) < AMOSTRAS_FC_MIN:
            estavel.append(False)
            motivos.append("amostras_fc_insuficientes")
        elif amplitude_robusta > AMPLITUDE_ROBUSTA_FC_MAX_BPM:
            estavel.append(False)
            motivos.append("amplitude_robusta_fc_excessiva")
        elif abs(float(valor_fc) - media_temporal) > DIFERENCA_MEDIA_FC_MAX_BPM:
            estavel.append(False)
            motivos.append("diferenca_media_temporal_excessiva")
        else:
            estavel.append(True)
            motivos.append("estavel")
    df["fc_estavel"] = estavel
    df["motivo_estabilidade"] = motivos
    return df


def criar_intervalos(df_atividade: pd.DataFrame) -> pd.DataFrame:
    """Cria intervalos consecutivos e audita as regras básicas."""
    df = estabilidade_fc(df_atividade)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["dt_s"] = df["timestamp"].diff().dt.total_seconds()
    df["dx_m"] = df["distancia_m"].diff()
    df["distancia_meio_m"] = (df["distancia_m"] + df["distancia_m"].shift(1)) / 2
    df["troco_id"] = np.floor(df["distancia_meio_m"] / COMPRIMENTO_TROCO_M).astype("Int64")
    motivos: list[str] = []
    validos: list[bool] = []
    for linha in df.itertuples(index=False):
        motivo = "valido"
        if pd.isna(linha.timestamp) or pd.isna(linha.dt_s):
            motivo = "timestamp_invalido"
        elif linha.dt_s <= 0:
            motivo = "timestamp_repetido_ou_recuado"
        elif linha.dt_s > SALTO_TEMPORAL_MAX_S:
            motivo = "salto_temporal"
        elif pd.isna(linha.dx_m) or linha.dx_m == 0:
            motivo = "pausa"
        elif linha.dx_m < 0:
            motivo = "recuo_distancia"
        elif linha.dx_m > 10 * linha.dt_s:
            motivo = "salto_distancia"
        elif pd.isna(linha.fc_bpm) or linha.fc_bpm <= 0:
            motivo = "fc_ausente_ou_nao_positiva"
        elif linha.dx_m / linha.dt_s < 0.5:
            motivo = "velocidade_inferior_limiar"
        elif not linha.fc_estavel:
            motivo = str(linha.motivo_estabilidade)
        validos.append(motivo == "valido")
        motivos.append(motivo)
    df["valido"] = validos
    df["motivo_exclusao"] = motivos
    return df


def agregar_amostras(
    intervalos: pd.DataFrame, trocos: pd.DataFrame, zonas: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classifica antes de agregar e devolve amostras e auditoria de grupos."""
    intervalos = intervalos.copy()
    intervalos["zona_ordem"] = intervalos["fc_bpm"].map(lambda valor: classificar_zona(valor, zonas))
    fora_zona = intervalos["valido"] & intervalos["zona_ordem"].isna()
    intervalos.loc[fora_zona, "valido"] = False
    intervalos.loc[fora_zona, "motivo_exclusao"] = "fc_fora_das_zonas"
    elegiveis = intervalos[intervalos["valido"]].copy()
    if elegiveis.empty:
        return pd.DataFrame(), pd.DataFrame()
    agregadas = (
        elegiveis.groupby(["atividade_id", "troco_id", "zona_ordem"], as_index=False)
        .agg(n_intervalos=("dx_m", "size"), distancia_valida_m=("dx_m", "sum"), tempo_valido_s=("dt_s", "sum"))
    )
    agregadas = agregadas.merge(trocos, on=["atividade_id", "troco_id"], how="left", validate="many_to_one")
    agregadas["cobertura"] = agregadas["distancia_valida_m"] / agregadas["comprimento_troco_m"]
    agregadas["motivo_grupo"] = np.select(
        [
            agregadas["n_intervalos"] < INTERVALOS_MIN_POR_TROCO,
            agregadas["cobertura"] < COBERTURA_TROCO_MIN,
            agregadas["cobertura"] > COBERTURA_TROCO_MAX,
        ],
        ["menos_2_intervalos", "cobertura_troco_insuficiente", "cobertura_troco_excessiva"],
        default="valido",
    )
    validas = agregadas[agregadas["motivo_grupo"] == "valido"].copy()
    validas["ritmo_alvo_s_km"] = 1000.0 * validas["tempo_valido_s"] / validas["distancia_valida_m"]
    mapa_zonas = {int(z["zona"]): z for z in zonas["zonas"]}
    validas["zona_ordem"] = validas["zona_ordem"].astype(int)
    validas["zona_limite_inferior_bpm"] = validas["zona_ordem"].map(
        lambda z: mapa_zonas[z]["limite_inferior_bpm"]
    )
    validas["zona_limite_superior_bpm"] = validas["zona_ordem"].map(
        lambda z: mapa_zonas[z]["limite_superior_bpm"]
    )
    validas["zona_aberta"] = validas["zona_ordem"].map(lambda z: mapa_zonas[z].get("zona_aberta", 0))
    validas["amostra_id"] = (
        validas["atividade_id"].astype(str)
        + "_"
        + validas["troco_id"].astype(int).astype(str)
        + "_Z"
        + validas["zona_ordem"].astype(str)
    )
    colunas = ["amostra_id", "atividade_id", "troco_id", *FEATURES, "ritmo_alvo_s_km"]
    return validas[colunas].sort_values(["atividade_id", "troco_id", "zona_ordem"]), agregadas


def calcular_pesos_atividade(amostras: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta pesos cuja soma é um em cada atividade."""
    resultado = amostras.copy()
    contagens = resultado.groupby("atividade_id")["atividade_id"].transform("size")
    resultado["peso_amostra"] = 1.0 / contagens
    return resultado


def _sha256(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as ficheiro:
        for bloco in iter(lambda: ficheiro.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _limites_argumento(valores: list[str] | None) -> list[int] | None:
    if valores is None:
        return None
    return [int(valor) for valor in valores]


def main() -> None:
    """Cria perfis, intervalos auditados e amostras finais."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idade", type=int)
    parser.add_argument("--limites-fc", nargs=6)
    parser.add_argument("--reprocessar", action="store_true")
    args = parser.parse_args()
    DADOS_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    zonas = carregar_ou_criar_zonas(CONFIGURACAO, args.idade, _limites_argumento(args.limites_fc))
    validar_zonas(zonas)
    registos_path = DADOS_PROCESSADOS / "registos_normalizados.csv"
    if not registos_path.exists():
        raise FileNotFoundError("Execute analisar_fits.py antes de preparar os dados")
    registos = pd.read_csv(registos_path, parse_dates=["timestamp"])
    todos_trocos: list[pd.DataFrame] = []
    todos_intervalos: list[pd.DataFrame] = []
    qualidade: list[dict[str, Any]] = []
    for atividade, grupo in registos.groupby("atividade_id", sort=False):
        trocos = perfil_percurso(grupo)
        trocos.insert(0, "atividade_id", str(atividade))
        intervalos = criar_intervalos(grupo)
        intervalos["atividade_id"] = str(atividade)
        todos_trocos.append(trocos)
        todos_intervalos.append(intervalos)
        motivos = intervalos["motivo_exclusao"].value_counts()
        for motivo, contagem in motivos.items():
            qualidade.append(
                {
                    "atividade_id": str(atividade),
                    "nivel": "intervalo",
                    "motivo": motivo,
                    "n": int(contagem),
                    "percentagem": float(100 * contagem / len(intervalos)),
                }
            )
    trocos_df = pd.concat(todos_trocos, ignore_index=True)
    intervalos_df = pd.concat(todos_intervalos, ignore_index=True)
    amostras, auditoria_grupos = agregar_amostras(intervalos_df, trocos_df, zonas)
    if amostras.empty:
        raise RuntimeError("Nenhuma amostra passou os filtros de estabilidade e cobertura")
    amostras = calcular_pesos_atividade(amostras)
    for (atividade, motivo), grupo in auditoria_grupos.groupby(["atividade_id", "motivo_grupo"]):
        qualidade.append(
            {
                "atividade_id": str(atividade),
                "nivel": "grupo",
                "motivo": motivo,
                "n": len(grupo),
                "percentagem": float(100 * len(grupo) / len(auditoria_grupos[auditoria_grupos["atividade_id"] == atividade])),
            }
        )
    trocos_df.to_csv(DADOS_PROCESSADOS / "trocos_percurso.csv", index=False)
    intervalos_df.to_csv(DADOS_PROCESSADOS / "intervalos_auditados.csv", index=False)
    amostras.to_csv(DADOS_PROCESSADOS / "amostras_modelo.csv", index=False)
    pd.DataFrame(qualidade).to_csv(DADOS_PROCESSADOS / "qualidade_dados.csv", index=False)
    fontes = [
        {"caminho": str(path.relative_to(RAIZ)), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(DADOS_BRUTOS.glob("*.fit"))
    ]
    manifesto = {
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "fontes": fontes,
        "parametros": {
            "passo_grelha_m": PASSO_GRELHA_M,
            "comprimento_troco_m": COMPRIMENTO_TROCO_M,
            "dt_max_s": SALTO_TEMPORAL_MAX_S,
            "velocidade_min_m_s": 0.5,
            "janela_fc_s": JANELA_FC_S,
            "cobertura_fc_min_s": COBERTURA_FC_MIN_S,
            "amostras_fc_min": AMOSTRAS_FC_MIN,
            "amplitude_robusta_fc_max_bpm": AMPLITUDE_ROBUSTA_FC_MAX_BPM,
            "diferenca_media_temporal_fc_max_bpm": DIFERENCA_MEDIA_FC_MAX_BPM,
            "tempo_inicial_excluido_s": TEMPO_INICIAL_EXCLUIDO_S,
            "intervalos_min_por_troco": INTERVALOS_MIN_POR_TROCO,
            "cobertura_min": COBERTURA_TROCO_MIN,
            "cobertura_max": COBERTURA_TROCO_MAX,
            "normalizacao_altitude": "escala fixa: altitude_media_m / 1000",
        },
        "zonas": zonas,
        "features": FEATURES,
        "datasets": [
            "registos_normalizados.csv",
            "trocos_percurso.csv",
            "intervalos_auditados.csv",
            "amostras_modelo.csv",
            "qualidade_dados.csv",
        ],
        "n_atividades": int(amostras["atividade_id"].nunique()),
        "n_amostras": int(len(amostras)),
    }
    (DADOS_PROCESSADOS / "manifesto_processamento.json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    colunas_features = [coluna for coluna in amostras.columns if coluna in FEATURES]
    assert colunas_features == FEATURES
    proibidas = {"fc_bpm", "velocidade_fit_m_s", "potencia_w"} & set(amostras.columns)
    if proibidas:
        raise AssertionError(f"Features proibidas no dataset: {sorted(proibidas)}")
    print(f"Atividades com amostras: {amostras['atividade_id'].nunique()}")
    print(f"Amostras válidas: {len(amostras)}")
    print("Amostras por zona:")
    print(amostras["zona_ordem"].value_counts().sort_index().to_string())
    print("Confirmação: nove features permitidas, sem FC exata, velocidade ou potência.")


if __name__ == "__main__":
    main()
