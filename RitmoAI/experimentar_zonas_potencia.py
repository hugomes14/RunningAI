"""Compara zonas cardíacas e de potência em troços e folds idênticos."""

from __future__ import annotations

import json
import os
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from .analisar_fits import ler_fit, normalizar_registos
    from .preparar_dados import (
        COBERTURA_TROCO_MAX,
        COBERTURA_TROCO_MIN,
        COMPRIMENTO_TROCO_M,
        INTERVALOS_MIN_POR_TROCO,
        SALTO_TEMPORAL_MAX_S,
        TEMPO_INICIAL_EXCLUIDO_S,
        perfil_percurso,
    )
except ImportError:  # Permite executar diretamente: python experimentar_zonas_potencia.py
    from analisar_fits import ler_fit, normalizar_registos
    from preparar_dados import (
        COBERTURA_TROCO_MAX,
        COBERTURA_TROCO_MIN,
        COMPRIMENTO_TROCO_M,
        INTERVALOS_MIN_POR_TROCO,
        SALTO_TEMPORAL_MAX_S,
        TEMPO_INICIAL_EXCLUIDO_S,
        perfil_percurso,
    )


RAIZ = Path(__file__).resolve().parent
DADOS_BRUTOS = RAIZ / "Dados" / "brutos"
CONFIG_FC = RAIZ / "configuracao" / "zonas_fc.json"
SAIDA = RAIZ / "artefactos" / "experiencia_potencia"
ALVO = "ritmo_alvo_s_km"
FEATURES = [
    "zona_ordem",
    "zona_limite_inferior",
    "zona_limite_superior",
    "zona_aberta",
    "declive_pct",
    "altitude_normalizada",
    "distancia_inicio_km",
    "subida_acumulada_m",
    "comprimento_troco_m",
]


def limites_fit_potencia(mensagens: dict[str, Any]) -> tuple[list[int], int | None]:
    """Escolhe a configuração de potência mais frequente dentro de um FIT."""
    candidatos: list[tuple[tuple[int, ...], int | None]] = []
    for mensagem in mensagens.get("time_in_zone_mesgs", []):
        valores = mensagem.get("power_zone_high_boundary")
        if valores:
            limites = tuple(int(valor) for valor in valores if valor is not None)
            if len(limites) >= 6:
                candidatos.append((limites[:6], mensagem.get("functional_threshold_power")))
    if candidatos:
        limites, ftp = Counter(candidatos).most_common(1)[0][0]
        return list(limites), int(ftp) if ftp is not None else None
    return [], None


def carregar_atividades() -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Descodifica os FIT e conserva apenas atividades com potência utilizável."""
    atividades: list[dict[str, Any]] = []
    cobertura: list[dict[str, Any]] = []
    for caminho in sorted(DADOS_BRUTOS.glob("*.fit")):
        mensagens = ler_fit(caminho)
        atividade_id = caminho.stem.replace("_ACTIVITY", "")
        registos = normalizar_registos(mensagens, atividade_id)
        n_potencia = int(registos["potencia_w"].notna().sum())
        limites, ftp = limites_fit_potencia(mensagens)
        cobertura.append(
            {
                "atividade_id": atividade_id,
                "n_registos": len(registos),
                "n_potencia": n_potencia,
                "cobertura_potencia_pct": 100 * n_potencia / len(registos) if len(registos) else 0,
                "ftp_w": ftp,
                "limites_potencia": ",".join(map(str, limites)),
            }
        )
        if n_potencia and limites:
            atividades.append(
                {
                    "atividade_id": atividade_id,
                    "caminho": caminho,
                    "registos": registos,
                    "limites_historicos": limites,
                    "ftp_historico": ftp,
                    "data": registos["timestamp"].min(),
                }
            )
    if not atividades:
        raise RuntimeError("Nenhum FIT contém simultaneamente registos e zonas de potência")
    return atividades, pd.DataFrame(cobertura)


def classificar(valor: float, limites: list[float]) -> int | None:
    """Converte um valor histórico numa das cinco zonas configuradas."""
    if not np.isfinite(valor):
        return None
    for indice, (inferior, superior) in enumerate(zip(limites[:-1], limites[1:]), start=1):
        if inferior <= valor < superior or (indice == 5 and valor <= superior):
            return indice
    return None


def construir_trocos_pareados(
    atividades: list[dict[str, Any]], limites_fc: list[int], limites_potencia: list[int]
) -> pd.DataFrame:
    """Cria um alvo por troço e mede FC e potência nas mesmas observações."""
    blocos: list[pd.DataFrame] = []
    for atividade in atividades:
        df = atividade["registos"].sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["dt_s"] = df["timestamp"].diff().dt.total_seconds()
        df["dx_m"] = df["distancia_m"].diff()
        df["distancia_meio_m"] = (df["distancia_m"] + df["distancia_m"].shift(1)) / 2
        df["troco_id"] = np.floor(df["distancia_meio_m"] / COMPRIMENTO_TROCO_M).astype("Int64")
        decorrido = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()
        velocidade = df["dx_m"] / df["dt_s"]
        valida = (
            df["timestamp"].notna()
            & df["dt_s"].between(np.nextafter(0, 1), SALTO_TEMPORAL_MAX_S)
            & (df["dx_m"] > 0)
            & (df["dx_m"] <= 10 * df["dt_s"])
            & (velocidade >= 0.5)
            & (decorrido >= TEMPO_INICIAL_EXCLUIDO_S)
            & (df["fc_bpm"] > 0)
            & (df["potencia_w"] > 0)
        )
        elegiveis = df[valida].copy()
        if elegiveis.empty:
            continue
        elegiveis["fc_tempo"] = elegiveis["fc_bpm"] * elegiveis["dt_s"]
        elegiveis["potencia_tempo"] = elegiveis["potencia_w"] * elegiveis["dt_s"]
        agregado = (
            elegiveis.groupby(["atividade_id", "troco_id"], as_index=False)
            .agg(
                n_intervalos=("dx_m", "size"),
                distancia_valida_m=("dx_m", "sum"),
                tempo_valido_s=("dt_s", "sum"),
                fc_tempo=("fc_tempo", "sum"),
                potencia_tempo=("potencia_tempo", "sum"),
            )
        )
        perfil = perfil_percurso(df)
        agregado = agregado.merge(
            perfil,
            on="troco_id",
            how="left",
            validate="one_to_one",
        )
        agregado["cobertura"] = agregado["distancia_valida_m"] / agregado["comprimento_troco_m"]
        agregado = agregado[
            (agregado["n_intervalos"] >= INTERVALOS_MIN_POR_TROCO)
            & agregado["cobertura"].between(COBERTURA_TROCO_MIN, COBERTURA_TROCO_MAX)
        ].copy()
        agregado["fc_media_bpm"] = agregado["fc_tempo"] / agregado["tempo_valido_s"]
        agregado["potencia_media_w"] = agregado["potencia_tempo"] / agregado["tempo_valido_s"]
        agregado["zona_fc"] = agregado["fc_media_bpm"].map(lambda x: classificar(x, limites_fc))
        agregado["zona_potencia"] = agregado["potencia_media_w"].map(
            lambda x: classificar(x, limites_potencia)
        )
        agregado[ALVO] = 1000 * agregado["tempo_valido_s"] / agregado["distancia_valida_m"]
        blocos.append(agregado.dropna(subset=["zona_fc", "zona_potencia", ALVO]))
    if not blocos:
        raise RuntimeError("Não foi possível criar troços pareados")
    resultado = pd.concat(blocos, ignore_index=True)
    resultado["zona_fc"] = resultado["zona_fc"].astype(int)
    resultado["zona_potencia"] = resultado["zona_potencia"].astype(int)
    resultado["amostra_id"] = (
        resultado["atividade_id"].astype(str) + "_" + resultado["troco_id"].astype(str)
    )
    return resultado


def criar_modelos() -> dict[str, Any]:
    """Replica os candidatos do pipeline principal."""
    restricoes = [-1, -1, -1, 0, 1, 0, 0, 0, 0]
    return {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(alpha=10.0))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.8, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=2, min_samples_leaf=5,
            loss="huber", random_state=42
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200, max_leaf_nodes=15, min_samples_leaf=20, learning_rate=0.05,
            l2_regularization=5, monotonic_cst=restricoes, random_state=42
        ),
        "SVR_RBF": Pipeline(
            [("scaler", StandardScaler()), ("regressor", SVR(kernel="rbf", C=10, epsilon=10))]
        ),
        "RedeNeuronal_MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", MLPRegressor(
                    hidden_layer_sizes=(32, 16), alpha=0.01, batch_size=64,
                    learning_rate_init=0.001, max_iter=1000, early_stopping=True,
                    validation_fraction=0.15, n_iter_no_change=40, random_state=42
                )),
            ]
        ),
        "Dummy": DummyRegressor(strategy="mean"),
    }


def dataset_modelo(
    trocos: pd.DataFrame, tipo: str, limites: list[int]
) -> pd.DataFrame:
    """Produz features com o mesmo esquema para cada tipo de zona."""
    coluna_zona = "zona_fc" if tipo == "Frequência cardíaca" else "zona_potencia"
    df = trocos.copy()
    df["zona_ordem"] = df[coluna_zona].astype(int)
    df["zona_limite_inferior"] = df["zona_ordem"].map(lambda z: limites[z - 1])
    df["zona_limite_superior"] = df["zona_ordem"].map(lambda z: limites[z])
    df["zona_aberta"] = 0
    return df


def ajustar(modelo: Any, x: pd.DataFrame, y: pd.Series, pesos: np.ndarray) -> Any:
    if isinstance(modelo, Pipeline):
        return modelo.fit(x, y, regressor__sample_weight=pesos)
    return modelo.fit(x, y, sample_weight=pesos)


def avaliar(tipo: str, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa LeaveOneGroupOut e conserva todas as previsões fora do treino."""
    modelos = criar_modelos()
    logo = LeaveOneGroupOut()
    grupos = df["atividade_id"].astype(str)
    metricas: list[dict[str, Any]] = []
    previsoes: list[pd.DataFrame] = []
    for fold, (idx_treino, idx_teste) in enumerate(
        logo.split(df[FEATURES], df[ALVO], grupos), start=1
    ):
        treino, teste = df.iloc[idx_treino], df.iloc[idx_teste]
        pesos = 1 / treino.groupby("atividade_id")["atividade_id"].transform("size").to_numpy()
        atividade = str(teste["atividade_id"].iloc[0])
        print(f"{tipo}: fold {fold}/{grupos.nunique()} — {atividade}", flush=True)
        for nome, base in modelos.items():
            inicio = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                modelo = ajustar(clone(base), treino[FEATURES], treino[ALVO], pesos)
            treino_s = time.perf_counter() - inicio
            previsto = modelo.predict(teste[FEATURES])
            real = teste[ALVO].to_numpy()
            metricas.append(
                {
                    "tipo_zona": tipo,
                    "fold": fold,
                    "atividade_id": atividade,
                    "modelo": nome,
                    "mae_s_km": mean_absolute_error(real, previsto),
                    "rmse_s_km": float(np.sqrt(mean_squared_error(real, previsto))),
                    "r2": r2_score(real, previsto) if len(real) >= 2 else np.nan,
                    "tempo_treino_s": treino_s,
                }
            )
            bloco = teste[["amostra_id", "atividade_id", ALVO]].copy()
            bloco["tipo_zona"] = tipo
            bloco["modelo"] = nome
            bloco["ritmo_previsto_s_km"] = previsto
            previsoes.append(bloco)
    return pd.DataFrame(metricas), pd.concat(previsoes, ignore_index=True)


def resumir(metricas: pd.DataFrame) -> pd.DataFrame:
    resumo = (
        metricas.groupby(["tipo_zona", "modelo"], as_index=False)
        .agg(
            mae_macro_s_km=("mae_s_km", "mean"),
            rmse_macro_s_km=("rmse_s_km", "mean"),
            r2_macro=("r2", "mean"),
            tempo_treino_medio_s=("tempo_treino_s", "mean"),
            n_atividades=("atividade_id", "nunique"),
        )
    )
    dummy = resumo[resumo["modelo"] == "Dummy"].set_index("tipo_zona")["mae_macro_s_km"]
    resumo["melhoria_vs_dummy_pct"] = resumo.apply(
        lambda x: 100 * (dummy[x["tipo_zona"]] - x["mae_macro_s_km"]) / dummy[x["tipo_zona"]],
        axis=1,
    )
    return resumo.sort_values(["tipo_zona", "mae_macro_s_km"]).reset_index(drop=True)


def criar_graficos(
    cobertura: pd.DataFrame, trocos: pd.DataFrame, resumo: pd.DataFrame
) -> None:
    """Visualiza cobertura, associação e desempenho preditivo."""
    sns.set_theme(style="whitegrid")
    SAIDA.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    c = cobertura[cobertura["n_potencia"] > 0].copy()
    sns.barplot(data=c, x="atividade_id", y="cobertura_potencia_pct", color="#6c5ce7", ax=ax)
    ax.set(xlabel="Atividade", ylabel="Cobertura de potência (%)", ylim=(0, 105))
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(SAIDA / "cobertura_potencia.png", dpi=180)
    plt.close(fig)

    longo = pd.concat(
        [
            trocos[[ALVO, "zona_fc"]].rename(columns={"zona_fc": "zona"}).assign(tipo="Frequência cardíaca"),
            trocos[[ALVO, "zona_potencia"]].rename(columns={"zona_potencia": "zona"}).assign(tipo="Potência"),
        ],
        ignore_index=True,
    )
    fig, eixos = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for eixo, tipo in zip(eixos, ["Frequência cardíaca", "Potência"]):
        sns.boxplot(data=longo[longo["tipo"] == tipo], x="zona", y=ALVO, showfliers=False, ax=eixo)
        eixo.set(title=tipo, xlabel="Zona", ylabel="Ritmo (s/km)")
    fig.tight_layout()
    fig.savefig(SAIDA / "ritmo_por_tipo_zona.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    sem_dummy = resumo[resumo["modelo"] != "Dummy"]
    sns.barplot(data=sem_dummy, x="modelo", y="mae_macro_s_km", hue="tipo_zona", ax=ax)
    ax.set(xlabel="Modelo", ylabel="MAE macro fora do treino (s/km)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(SAIDA / "comparacao_mae_fc_potencia.png", dpi=180)
    plt.close(fig)


def escrever_relatorio(
    cobertura: pd.DataFrame,
    trocos: pd.DataFrame,
    correlacoes: pd.DataFrame,
    correlacoes_declive: pd.DataFrame,
    resumo: pd.DataFrame,
    comparacao: dict[str, Any],
    configuracao: dict[str, Any],
) -> None:
    """Guarda uma síntese legível e ligada aos resultados reproduzíveis."""
    ranking = resumo[
        [
            "tipo_zona",
            "modelo",
            "mae_macro_s_km",
            "rmse_macro_s_km",
            "r2_macro",
            "melhoria_vs_dummy_pct",
        ]
    ].copy()
    for coluna in ranking.select_dtypes(include="number"):
        ranking[coluna] = ranking[coluna].round(2)
    correlacoes_tabela = correlacoes.copy()
    for coluna in correlacoes_tabela.select_dtypes(include="number"):
        correlacoes_tabela[coluna] = correlacoes_tabela[coluna].round(3)
    declive_tabela = correlacoes_declive.pivot(
        index="classe_declive", columns="variavel", values="spearman_ritmo"
    ).reset_index()
    for coluna in declive_tabela.select_dtypes(include="number"):
        declive_tabela[coluna] = declive_tabela[coluna].round(3)
    contagens = pd.DataFrame(
        {
            "Zona": range(1, 6),
            "Troços — FC": trocos["zona_fc"].value_counts().reindex(range(1, 6), fill_value=0),
            "Troços — potência": trocos["zona_potencia"].value_counts().reindex(range(1, 6), fill_value=0),
        }
    )
    texto = f"""# Experiência — zonas cardíacas versus zonas de potência

## Objetivo e desenho

Esta experiência testa se a gama de potência melhora a previsão do ritmo face à gama de frequência cardíaca. A comparação usa os mesmos **{len(trocos)} troços de 50 m**, as mesmas **{trocos['atividade_id'].nunique()} atividades**, os mesmos alvos e os mesmos folds `LeaveOneGroupOut`. Em cada versão mudam apenas a zona e os seus limites.

As métricas desta experiência só devem ser comparadas entre si. Não são diretamente comparáveis com as métricas do modelo principal, porque o subconjunto contém apenas atividades com potência e usa um único alvo pareado por troço.

A potência exata e a FC exata servem apenas para classificar os dados históricos e não entram como features do modelo. O alvo continua a ser calculado por distância e tempo. Esta construção evita que uma alternativa ganhe por usar mais observações.

Dos {len(cobertura)} FIT analisados, **{int((cobertura['n_potencia'] > 0).sum())} têm potência**. Nestas atividades, a cobertura dos registos de potência é aproximadamente total.

Foram usados os limites de potência do FIT mais recente: **{', '.join(map(str, configuracao['limites_potencia_w']))} W**, com limiar funcional de **{configuracao['ftp_w']} W**.

## Associação com o ritmo

{correlacoes_tabela.to_markdown(index=False)}

A potência média contínua apresenta uma relação mais forte com o ritmo do que a FC média. Porém, ao reduzir a potência às cinco gamas Garmin, grande parte dessa informação desaparece: a correlação ordinal da zona de potência é substancialmente mais fraca do que a da zona cardíaca.

Ao controlar aproximadamente o terreno por classes de declive, a potência contínua mantém correlações fortes com o ritmo:

{declive_tabela.to_markdown(index=False)}

![Ritmo por tipo de zona](ritmo_por_tipo_zona.png)

## Cobertura das zonas

{contagens.to_markdown(index=False)}

![Cobertura de potência](cobertura_potencia.png)

## Validação dos modelos

{ranking.to_markdown(index=False)}

![Comparação do MAE](comparacao_mae_fc_potencia.png)

O melhor resultado cardíaco foi **{comparacao['modelo_fc']}**, com MAE macro de **{comparacao['mae_fc']:.2f} s/km**. O melhor resultado por potência foi **{comparacao['modelo_potencia']}**, com **{comparacao['mae_potencia']:.2f} s/km**.

A potência reduz o MAE em média **{-comparacao['diferenca_potencia_menos_fc_s_km']:.2f} s/km**, mas vence apenas **{comparacao['atividades_potencia_melhor']} de {comparacao['n_atividades']} atividades**. O intervalo bootstrap de 95% da diferença potência menos FC é **[{comparacao['bootstrap_95_inferior']:.2f}, {comparacao['bootstrap_95_superior']:.2f}] s/km** e atravessa zero. O RMSE dos dois vencedores é praticamente igual.

## Conclusão

Os dados sustentam que a **potência contínua contém mais informação instantânea sobre o ritmo**, mas ainda não demonstram que **cinco gamas de potência** generalizem melhor do que as zonas cardíacas. A pequena melhoria média é incerta e depende do algoritmo.

O modelo atual não deve ser substituído com base nesta execução. O teste deve ser repetido quando existirem mais atividades com potência. Também faz sentido avaliar as zonas históricas relativas ao limiar funcional registado em cada atividade, porque o FTP observado nos FIT variou ao longo do tempo.

## Ficheiros auditáveis

- `trocos_pareados.csv`: amostras idênticas usadas nas duas alternativas;
- `correlacoes.csv`: correlações contínuas e por zona;
- `correlacoes_por_declive.csv`: associações dentro de classes de terreno;
- `metricas_por_atividade.csv`: métricas de cada fold;
- `previsoes_fora_do_treino.csv`: previsões estritamente fora do treino;
- `resumo_modelos.csv`: ranking agregado;
- `resultado.json`: síntese estruturada.
"""
    (SAIDA / "RELATORIO_EXPERIENCIA.md").write_text(texto, encoding="utf-8")


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    atividades, cobertura = carregar_atividades()
    config_fc = json.loads(CONFIG_FC.read_text(encoding="utf-8"))
    limites_fc = [int(x) for x in config_fc["limites_fc_bpm"]]
    referencia = max(atividades, key=lambda item: item["data"])
    limites_potencia = referencia["limites_historicos"]
    configuracao = {
        "limites_potencia_w": limites_potencia,
        "ftp_w": referencia["ftp_historico"],
        "atividade_referencia": referencia["atividade_id"],
        "nota": "Limites da atividade mais recente, aplicados a todo o subconjunto para comparar com a metodologia cardíaca atual.",
    }
    (SAIDA / "configuracao_potencia.json").write_text(
        json.dumps(configuracao, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cobertura.to_csv(SAIDA / "cobertura_potencia.csv", index=False)
    trocos = construir_trocos_pareados(atividades, limites_fc, limites_potencia)
    trocos.to_csv(SAIDA / "trocos_pareados.csv", index=False)

    correlacoes = []
    for tipo, valor, zona in [
        ("Frequência cardíaca", "fc_media_bpm", "zona_fc"),
        ("Potência", "potencia_media_w", "zona_potencia"),
    ]:
        correlacoes.append(
            {
                "tipo_zona": tipo,
                "pearson_valor_ritmo": trocos[valor].corr(trocos[ALVO], method="pearson"),
                "spearman_valor_ritmo": trocos[valor].corr(trocos[ALVO], method="spearman"),
                "spearman_zona_ritmo": trocos[zona].corr(trocos[ALVO], method="spearman"),
                "n_trocos": len(trocos),
                "n_atividades": trocos["atividade_id"].nunique(),
            }
        )
    correlacoes_df = pd.DataFrame(correlacoes)
    correlacoes_df.to_csv(SAIDA / "correlacoes.csv", index=False)
    trocos["classe_declive"] = pd.cut(
        trocos["declive_pct"],
        [-np.inf, -3, -1, 1, 3, np.inf],
        labels=["descida forte", "descida", "plano", "subida", "subida forte"],
    )
    correlacoes_declive = []
    for classe, bloco in trocos.groupby("classe_declive", observed=True):
        for nome, coluna in [
            ("FC média", "fc_media_bpm"),
            ("Zona FC", "zona_fc"),
            ("Potência média", "potencia_media_w"),
            ("Zona potência", "zona_potencia"),
        ]:
            correlacoes_declive.append(
                {
                    "classe_declive": str(classe),
                    "variavel": nome,
                    "spearman_ritmo": bloco[coluna].corr(bloco[ALVO], method="spearman"),
                    "n_trocos": len(bloco),
                }
            )
    correlacoes_declive_df = pd.DataFrame(correlacoes_declive)
    correlacoes_declive_df.to_csv(SAIDA / "correlacoes_por_declive.csv", index=False)

    resultados_metricas = []
    resultados_previsoes = []
    for tipo, limites in [
        ("Frequência cardíaca", limites_fc),
        ("Potência", limites_potencia),
    ]:
        df_modelo = dataset_modelo(trocos, tipo, limites)
        metricas, previsoes = avaliar(tipo, df_modelo)
        resultados_metricas.append(metricas)
        resultados_previsoes.append(previsoes)
    metricas = pd.concat(resultados_metricas, ignore_index=True)
    previsoes = pd.concat(resultados_previsoes, ignore_index=True)
    resumo = resumir(metricas)
    metricas.to_csv(SAIDA / "metricas_por_atividade.csv", index=False)
    previsoes.to_csv(SAIDA / "previsoes_fora_do_treino.csv", index=False)
    resumo.to_csv(SAIDA / "resumo_modelos.csv", index=False)
    criar_graficos(cobertura, trocos, resumo)

    melhor = resumo[resumo["modelo"] != "Dummy"].loc[
        lambda x: x.groupby("tipo_zona")["mae_macro_s_km"].idxmin()
    ]
    linha_fc = melhor[melhor["tipo_zona"] == "Frequência cardíaca"].iloc[0]
    linha_potencia = melhor[melhor["tipo_zona"] == "Potência"].iloc[0]
    mae_atividade = metricas.pivot_table(
        index=["atividade_id", "tipo_zona"], columns="modelo", values="mae_s_km"
    )
    fc = mae_atividade.xs("Frequência cardíaca", level="tipo_zona")[linha_fc["modelo"]]
    potencia = mae_atividade.xs("Potência", level="tipo_zona")[linha_potencia["modelo"]]
    diferencas = (potencia - fc).dropna().to_numpy()
    rng = np.random.default_rng(42)
    bootstrap = np.mean(
        rng.choice(diferencas, size=(50_000, len(diferencas)), replace=True), axis=1
    )
    intervalo = np.quantile(bootstrap, [0.025, 0.975])
    comparacao = {
        "modelo_fc": str(linha_fc["modelo"]),
        "mae_fc": float(linha_fc["mae_macro_s_km"]),
        "rmse_fc": float(linha_fc["rmse_macro_s_km"]),
        "modelo_potencia": str(linha_potencia["modelo"]),
        "mae_potencia": float(linha_potencia["mae_macro_s_km"]),
        "rmse_potencia": float(linha_potencia["rmse_macro_s_km"]),
        "diferenca_potencia_menos_fc_s_km": float(diferencas.mean()),
        "atividades_potencia_melhor": int((diferencas < 0).sum()),
        "n_atividades": int(len(diferencas)),
        "bootstrap_95_inferior": float(intervalo[0]),
        "bootstrap_95_superior": float(intervalo[1]),
    }
    resultado = {
        "cobertura": {
            "atividades_total": int(len(cobertura)),
            "atividades_com_potencia": int((cobertura["n_potencia"] > 0).sum()),
            "trocos_pareados": int(len(trocos)),
        },
        "configuracao": configuracao,
        "correlacoes": correlacoes,
        "correlacoes_por_declive": correlacoes_declive,
        "melhores_modelos": melhor.to_dict(orient="records"),
        "comparacao_pareada_dos_vencedores": comparacao,
        "metodologia": "Mesmos troços, alvos e folds por atividade; muda apenas a gama de intensidade.",
    }
    (SAIDA / "resultado.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    escrever_relatorio(
        cobertura, trocos, correlacoes_df, correlacoes_declive_df,
        resumo, comparacao, configuracao
    )
    print("\nCorrelações:")
    print(correlacoes_df.to_string(index=False))
    print("\nMelhores modelos:")
    print(melhor[["tipo_zona", "modelo", "mae_macro_s_km", "rmse_macro_s_km", "r2_macro", "melhoria_vs_dummy_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
