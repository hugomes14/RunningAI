"""Geração do relatório final a partir dos artefactos reais."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "Dados" / "processados"
ARTEFACTOS = RAIZ / "artefactos"
METRICAS = ARTEFACTOS / "metricas"
GRAFICOS = ARTEFACTOS / "graficos"
RELATORIOS = ARTEFACTOS / "relatorios"


def _tabela(df: pd.DataFrame, casas: int = 2) -> str:
    """Converte um DataFrame pequeno numa tabela Markdown."""
    try:
        return df.round(casas).to_markdown(index=False)
    except ImportError:
        cabecalho = "| " + " | ".join(map(str, df.columns)) + " |"
        separador = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        linhas = ["| " + " | ".join(map(str, linha)) + " |" for linha in df.round(casas).to_numpy()]
        return "\n".join([cabecalho, separador, *linhas])


def _imagem(nome: str, legenda: str) -> str:
    caminho = Path("artefactos") / "graficos" / nome
    if (RAIZ / caminho).exists():
        return f"![{legenda}]({caminho.as_posix()})"
    return f"*Visualização não disponível: `{caminho.as_posix()}`.*"


def secao_objetivo() -> str:
    return """## 1. Objetivo

O projeto estima o ritmo pessoal de corrida num percurso conhecido a partir da posição no percurso, do perfil altimétrico e da zona cardíaca escolhida. A frequência cardíaca exata classifica os dados históricos em Z1–Z5, mas não entra como feature. Foram comparados cinco algoritmos de regressão e um baseline constante; um único vencedor foi treinado no final.
"""


def secao_dados() -> str:
    manifesto = json.loads((DADOS / "manifesto_processamento.json").read_text(encoding="utf-8"))
    resumo = pd.read_csv(
        ARTEFACTOS / "analise_inicial" / "resumo_atividades.csv",
        dtype={"atividade_id": str},
    )
    return f"""## 2. Dados

- FIT crus auditados: **{len(manifesto['fontes'])}**.
- Atividades válidas na análise inicial: **{len(resumo)}**.
- Distância histórica total: **{resumo['distancia_m'].sum() / 1000:.1f} km**.
- Período: **{resumo['data_inicio'].min()}** a **{resumo['data_inicio'].max()}**.
- Os FIT originais permanecem em `Dados/brutos/`; todos os datasets derivados ficam em `Dados/processados/`.
- O manifesto guarda o hash SHA-256 de cada origem e os parâmetros de processamento.
"""


def secao_analise_inicial() -> str:
    resumo = pd.read_csv(
        ARTEFACTOS / "analise_inicial" / "resumo_atividades.csv",
        dtype={"atividade_id": str},
    )
    tabela = resumo[["atividade_id", "distancia_m", "ritmo_medio_s_km", "fc_media_bpm", "subida_aproximada_m"]].copy()
    tabela["distancia_km"] = tabela.pop("distancia_m") / 1000
    return f"""## 3. Análise exploratória inicial

O resumo abaixo resulta diretamente dos registos normalizados.

{_tabela(tabela)}

{_imagem('perfil_atividades.png', 'Perfis das atividades')}

{_imagem('histogramas_globais.png', 'Distribuições globais')}
"""


def secao_qualidade() -> str:
    qualidade = pd.read_csv(DADOS / "qualidade_dados.csv")
    intervalos = pd.read_csv(DADOS / "intervalos_auditados.csv", low_memory=False)
    manifesto = json.loads((DADOS / "manifesto_processamento.json").read_text(encoding="utf-8"))
    parametros = manifesto["parametros"]
    contagens = intervalos["motivo_exclusao"].value_counts().rename_axis("motivo").reset_index(name="n")
    contagens["percentagem"] = 100 * contagens["n"] / len(intervalos)
    return f"""## 4. Qualidade dos dados

Foram auditados **{len(intervalos)}** intervalos. Nenhuma exclusão foi silenciosa; cada linha conserva `valido` e `motivo_exclusao`.

O filtro causal usa os últimos **{parametros['janela_fc_s']:.0f} s**, exige pelo menos **{parametros['amostras_fc_min']} medições** que cubram **{parametros['cobertura_fc_min_s']:.0f} s**, amplitude robusta P90−P10 até **{parametros['amplitude_robusta_fc_max_bpm']:.0f} bpm** e diferença para a média temporal até **{parametros['diferenca_media_temporal_fc_max_bpm']:.0f} bpm**. São aceites intervalos até **{parametros['dt_max_s']:.0f} s**, o que inclui gravação inteligente do Garmin; cada troço precisa de pelo menos **{parametros['intervalos_min_por_troco']} intervalos** e cobertura entre **{100 * parametros['cobertura_min']:.0f}%** e **{100 * parametros['cobertura_max']:.0f}%**.

{_tabela(contagens.head(12))}

{_imagem('qualidade_dados.png', 'Qualidade dos dados por atividade')}
"""


def secao_perfil_alvo() -> str:
    manifesto = json.loads((DADOS / "manifesto_processamento.json").read_text(encoding="utf-8"))
    parametros = manifesto["parametros"]
    return f"""## 5. Perfil do percurso e variável-alvo

O perfil usa apenas distância e altitude, variáveis disponíveis antes da corrida. A altitude foi interpolada a cada **{parametros['passo_grelha_m']:.0f} m**, suavizada por média centrada de cinco pontos e dividida em troços de **{parametros['comprimento_troco_m']:.0f} m**. A suavização centrada é válida neste contexto porque o perfil completo é conhecido antes da partida.

A altitude usada pelo modelo foi colocada numa escala fixa através de `altitude_media_m / 1000` após a interpolação e suavização. Esta transformação preserva a diferença de altitude absoluta entre percursos, não depende dos dados de outras atividades e evita fuga na validação. A altitude em metros foi preservada nos dados do perfil e nos gráficos.

O alvo foi calculado por:

`ritmo_alvo_s_km = 1000 × soma(tempo válido) / soma(distância válida)`

O campo de velocidade do FIT não foi usado para construir o alvo.
"""


def secao_zonas() -> str:
    zonas = json.loads((RAIZ / "configuracao" / "zonas_fc.json").read_text(encoding="utf-8"))
    tabela = pd.DataFrame(zonas["zonas"])[["nome", "limite_inferior_bpm", "limite_superior_bpm"]]
    return f"""## 6. Zonas cardíacas

{_tabela(tabela, 0)}

Origem: {zonas['origem']}

Nota: {zonas['notas']}
"""


def secao_dataset_final() -> str:
    df = pd.read_csv(DADOS / "amostras_modelo.csv")
    por_zona = df.groupby("zona_ordem", as_index=False).agg(amostras=("amostra_id", "size"), atividades=("atividade_id", "nunique"), ritmo_mediano_s_km=("ritmo_alvo_s_km", "median"))
    return f"""## 7. Dataset final

- Amostras: **{len(df)}**.
- Atividades com amostras elegíveis: **{df['atividade_id'].nunique()}**.
- Features: **9**, todas conhecidas antes da corrida.
- O dataset não contém FC exata, potência nem velocidade como entrada do modelo.

{_tabela(por_zona)}

{_imagem('distribuicoes_dataset.png', 'Distribuições do dataset')}

{_imagem('ritmo_por_zona.png', 'Ritmo por zona')}

{_imagem('ritmo_declive_zona.png', 'Ritmo, declive e zona')}
"""


def secao_validacao() -> str:
    resumo = pd.read_csv(METRICAS / "resumo_modelos.csv")
    n_atividades = int(resumo["n_atividades"].max())
    return f"""## 8. Validação por atividades

Foi aplicado `LeaveOneGroupOut` com **{n_atividades} folds**. Em cada fold, uma atividade completa ficou no teste e as restantes no treino. Os pesos deram o mesmo peso total a cada atividade de treino. Scalers e modelos foram ajustados apenas dentro do treino de cada fold.

As métricas apresentadas resultam apenas de previsões fora do treino. O modelo final treinado com todos os dados não foi usado para estimar desempenho.
"""


def secao_modelos() -> str:
    return """## 9. Modelos comparados

- Ridge com padronização;
- Random Forest;
- Gradient Boosting com perda Huber;
- HistGradientBoosting com restrições monotónicas;
- SVR RBF com padronização;
- rede neuronal MLP (32 e 16 neurónios), com padronização, regularização L2 e paragem antecipada;
- DummyRegressor como baseline constante.

Todos receberam as mesmas linhas, folds e pesos. Os parâmetros foram fixados antes da avaliação; não houve tuning na atividade de teste.

A MLP foi mantida pequena devido ao número reduzido de atividades. Em vários folds, o otimizador atingiu o limite de 1 000 iterações enquanto ainda melhorava, pelo que o resultado representa uma primeira experiência e não uma otimização exaustiva da arquitetura.
"""


def secao_resultados() -> str:
    resumo = pd.read_csv(METRICAS / "resumo_modelos.csv")
    colunas = ["modelo", "mae_macro_s_km", "rmse_macro_s_km", "r2_macro", "melhoria_vs_dummy_pct", "tempo_treino_medio_s", "vencedor"]
    return f"""## 10. Resultados

{_tabela(resumo[colunas])}

{_imagem('comparacao_modelos.png', 'Comparação dos modelos')}

{_imagem('metricas_por_atividade.png', 'Métricas por atividade')}

{_imagem('tempos_modelos.png', 'Tempos dos modelos')}
"""


def secao_diagnostico() -> str:
    resultado = json.loads((METRICAS / "resultado.json").read_text(encoding="utf-8"))
    metricas = resultado["metricas_oof"]
    intervalos = pd.read_csv(DADOS / "intervalos_auditados.csv", dtype={"atividade_id": str}, low_memory=False)
    amostras = pd.read_csv(DADOS / "amostras_modelo.csv", dtype={"atividade_id": str})
    por_atividade = pd.read_csv(METRICAS / "metricas_por_atividade.csv", dtype={"atividade_id": str})
    medianas = intervalos.groupby("atividade_id")["dt_s"].median().rename("dt_mediano_s")
    quantidades = amostras.groupby("atividade_id").size().rename("n_amostras")
    comparacao_modos = por_atividade[por_atividade["modelo"] == resultado["modelo_vencedor"]].merge(
        medianas, on="atividade_id"
    ).merge(quantidades, on="atividade_id")
    comparacao_modos["modo_gravacao"] = np.where(
        comparacao_modos["dt_mediano_s"] > 3, "inteligente", "1 segundo"
    )
    resumo_modos = comparacao_modos.groupby("modo_gravacao", as_index=False).agg(
        atividades=("atividade_id", "nunique"),
        amostras=("n_amostras", "sum"),
        mae_medio_s_km=("mae_s_km", "mean"),
        rmse_medio_s_km=("rmse_s_km", "mean"),
    )
    return f"""## 11. Diagnóstico do vencedor

O modelo vencedor foi **{resultado['modelo_vencedor']}**, com MAE macro de **{metricas['mae_macro_s_km']:.2f} s/km**, RMSE macro de **{metricas['rmse_macro_s_km']:.2f} s/km**, R² macro de **{metricas['r2_macro']:.3f}** e melhoria de **{metricas['melhoria_vs_dummy_pct']:.1f}%** face ao baseline.

O R² macro negativo mostra que a generalização entre algumas atividades é fraca apesar da redução do erro absoluto face ao baseline. As atividades representam percursos e tipos de treino diferentes; por isso, MAE por atividade e cobertura do espaço de features são essenciais para interpretar o resultado.

Desempenho por frequência de gravação:

{_tabela(resumo_modos)}

{_imagem('previsto_vs_real.png', 'Previsto versus real')}

{_imagem('residuos_modelo.png', 'Resíduos do vencedor')}

{_imagem('erro_por_zona_declive.png', 'Erro por zona e declive')}

{_imagem('importancia_permutacao.png', 'Importância por permutação')}

{_imagem('curva_aprendizagem_atividades.png', 'Curva de aprendizagem por atividades')}
"""


def secao_previsao() -> str:
    ficheiros = sorted((ARTEFACTOS / "previsoes_percurso").glob("*.json"))
    if not ficheiros:
        return """## 12. Previsão de percurso

A etapa de previsão de percurso não foi executada.
"""
    resumo = json.loads(ficheiros[-1].read_text(encoding="utf-8"))
    tabela = pd.DataFrame(
        [{"zona": zona, **valores} for zona, valores in resumo["zonas"].items()]
    )[["zona", "distancia_valida_m", "tempo_estimado_sem_paragens_s", "ritmo_medio_mmss_km", "distancia_pouco_suporte_m", "trocos_invalidos"]]
    return f"""## 12. Previsão de percurso

Percurso de demonstração: `{resumo['percurso']}`.

{_tabela(tabela)}

{_imagem('previsao_percurso.png', 'Previsão ao longo do percurso')}

{resumo['aviso']}
"""


def secao_limitacoes() -> str:
    df = pd.read_csv(DADOS / "amostras_modelo.csv")
    outliers = pd.read_csv(ARTEFACTOS / "analise_dataset" / "outliers_iqr.csv")
    contagens = df["zona_ordem"].value_counts()
    return f"""## 13. Limitações

- O estudo contém **{df['atividade_id'].nunique()} atividades elegíveis** de uma única pessoa; a validade externa para outras pessoas é reduzida.
- Foram identificados **{len(outliers)} candidatos a outlier** por IQR e nenhum foi removido automaticamente.
- A cobertura das zonas é desigual. A zona com menos dados tem **{int(contagens.min())} amostras**, o que torna as suas previsões especialmente incertas.
- As zonas atuais do Garmin foram aplicadas a todo o histórico. Alguns FIT antigos registam limites diferentes, pelo que a evolução da configuração pessoal pode introduzir inconsistência.
- A validação por atividade evita a fuga entre troços correlacionados, mas novos percursos podem ter altitudes, declives ou distâncias fora da cobertura histórica.
- O filtro de FC é causal, mas seleciona apenas períodos estáveis e pode favorecer troços com comportamento fisiológico mais regular.
- A previsão por zona assume que o atleta permanece nessa zona; o modelo não simula atraso ou deriva da resposta cardíaca.
- Se fosse usada a fórmula `220 - idade`, ela seria apenas uma aproximação. Nesta execução, os limites vieram diretamente da configuração Garmin mais recente.
"""


def secao_conclusoes() -> str:
    resultado = json.loads((METRICAS / "resultado.json").read_text(encoding="utf-8"))
    m = resultado["metricas_oof"]
    return f"""## 14. Conclusões

O **{resultado['modelo_vencedor']}** apresentou o menor MAE macro fora do treino, com **{m['mae_macro_s_km']:.2f} s/km**, e superou o baseline em **{m['melhoria_vs_dummy_pct']:.1f}%**. O desempenho varia bastante entre atividades e o R² macro permaneceu negativo, pelo que as previsões devem ser lidas em conjunto com os indicadores de pouco suporte e com os gráficos por zona, declive e atividade.

O passo seguinte com maior valor é recolher mais atividades que preencham as zonas e perfis altimétricos pouco representados, sobretudo Z1, e repetir a validação sem alterar o conjunto de teste de cada fold.
"""


def main() -> None:
    """Compõe e grava as catorze secções do relatório."""
    RELATORIOS.mkdir(parents=True, exist_ok=True)
    secoes = [
        secao_objetivo(),
        secao_dados(),
        secao_analise_inicial(),
        secao_qualidade(),
        secao_perfil_alvo(),
        secao_zonas(),
        secao_dataset_final(),
        secao_validacao(),
        secao_modelos(),
        secao_resultados(),
        secao_diagnostico(),
        secao_previsao(),
        secao_limitacoes(),
        secao_conclusoes(),
    ]
    conteudo = "# Relatório Final — Previsão de Ritmo por Percurso e Zona Cardíaca\n\n" + "\n".join(secoes)
    destino = RAIZ / "RELATORIO_FINAL.md"
    destino.write_text(conteudo, encoding="utf-8")
    shutil.copyfile(destino, RELATORIOS / "RELATORIO_FINAL.md")
    print("Relatório criado com 14 secções:")
    print("objetivo, dados, EDA inicial, qualidade, perfil/alvo, zonas, dataset, validação, modelos, resultados, diagnóstico, previsão, limitações e conclusões")


if __name__ == "__main__":
    main()
