"""Aplicação Flask para previsão de ritmo e consulta dos gráficos de treino."""

from __future__ import annotations

import base64
import io
import os
import tempfile
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from preparar_dados import perfil_percurso
from prever_percurso import (
    DADOS_TREINO,
    MODELO_PADRAO,
    carregar_percurso,
    criar_resumo,
    prever,
    reconstruir_features,
)


RAIZ = Path(__file__).resolve().parent
DADOS_BRUTOS = RAIZ / "Dados" / "brutos"
GRAFICOS = RAIZ / "artefactos" / "graficos"
RESUMO_ATIVIDADES = RAIZ / "artefactos" / "analise_inicial" / "resumo_atividades.csv"
EXTENSOES_PERMITIDAS = {".fit", ".csv"}
LIMITE_UPLOAD_MB = 32
LOCK_GRAFICO = Lock()

GRAFICOS_PRINCIPAIS = [
    ("comparacao_modelos.png", "Comparação dos modelos", "MAE e RMSE em validação por atividade."),
    ("metricas_por_atividade.png", "Erro por atividade", "Variação do MAE nas atividades deixadas fora do treino."),
    ("previsto_vs_real.png", "Previsto vs. real", "Dispersão das previsões fora do treino do modelo vencedor."),
    ("residuos_modelo.png", "Resíduos", "Distribuição e padrão dos erros fora do treino."),
    ("erro_por_zona_declive.png", "Erro por zona e declive", "Desempenho nas diferentes intensidades e inclinações."),
    ("importancia_permutacao.png", "Importância das variáveis", "Perda de desempenho ao permutar cada variável no teste."),
    ("curva_aprendizagem_atividades.png", "Curva de aprendizagem", "Evolução do erro com mais atividades completas."),
    ("tempos_modelos.png", "Tempos dos modelos", "Custo médio de treino e previsão."),
    ("correlacoes_dataset.png", "Correlações", "Relações lineares no dataset de modelação."),
    ("distribuicoes_dataset.png", "Distribuições", "Distribuição das variáveis e do ritmo-alvo."),
    ("ritmo_por_zona.png", "Ritmo por zona", "Distribuição do ritmo observado em cada zona cardíaca."),
    ("ritmo_declive_zona.png", "Ritmo, zona e declive", "Interação entre intensidade, inclinação e ritmo."),
    ("qualidade_dados.png", "Qualidade dos dados", "Intervalos aceites e rejeitados por atividade."),
]


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = LIMITE_UPLOAD_MB * 1024 * 1024

_cache_modelo: dict[str, Any] = {"mtime": None, "pacote": None}
_cache_treino: dict[str, Any] = {"mtime": None, "dados": None}


def carregar_modelo_atual() -> dict[str, Any]:
    """Carrega novamente o modelo quando o ficheiro é atualizado."""
    if not MODELO_PADRAO.exists():
        raise FileNotFoundError("Modelo final inexistente. Execute primeiro o treino.")
    mtime = MODELO_PADRAO.stat().st_mtime_ns
    if _cache_modelo["mtime"] != mtime:
        _cache_modelo.update(mtime=mtime, pacote=joblib.load(MODELO_PADRAO))
    return _cache_modelo["pacote"]


def carregar_treino_atual() -> pd.DataFrame:
    """Atualiza o dataset usado para calcular o indicador de suporte."""
    mtime = DADOS_TREINO.stat().st_mtime_ns
    if _cache_treino["mtime"] != mtime:
        _cache_treino.update(
            mtime=mtime,
            dados=pd.read_csv(DADOS_TREINO, dtype={"atividade_id": str}),
        )
    return _cache_treino["dados"]


def listar_percursos() -> list[dict[str, str]]:
    """Lista percursos existentes com data e distância quando disponíveis."""
    metadados: dict[str, dict[str, Any]] = {}
    if RESUMO_ATIVIDADES.exists():
        resumo = pd.read_csv(RESUMO_ATIVIDADES, dtype={"atividade_id": str})
        metadados = {str(linha["atividade_id"]): linha.to_dict() for _, linha in resumo.iterrows()}
    percursos: list[dict[str, str]] = []
    for caminho in sorted(DADOS_BRUTOS.iterdir(), reverse=True):
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_PERMITIDAS:
            atividade_id = caminho.stem.replace("_ACTIVITY", "")
            dados = metadados.get(atividade_id, {})
            data = str(dados.get("data_inicio", ""))[:10]
            distancia = dados.get("distancia_m")
            distancia_texto = f"{float(distancia) / 1000:.1f} km" if distancia is not None else ""
            detalhes = " · ".join(parte for parte in [data, distancia_texto] if parte)
            percursos.append(
                {"nome": caminho.name, "atividade_id": atividade_id, "detalhes": detalhes}
            )
    return percursos


def resolver_percurso_existente(nome: str) -> Path:
    """Resolve um nome apenas dentro da pasta de dados brutos."""
    permitidos = {item["nome"] for item in listar_percursos()}
    if nome not in permitidos:
        raise ValueError("Selecione um percurso existente válido.")
    return DADOS_BRUTOS / nome


def formatar_duracao(segundos: float) -> str:
    """Formata uma duração em horas, minutos e segundos."""
    total = max(0, int(round(segundos)))
    horas, resto = divmod(total, 3600)
    minutos, segundos = divmod(resto, 60)
    return f"{horas:d}:{minutos:02d}:{segundos:02d}"


def criar_grafico_previsao(resultado: pd.DataFrame) -> str:
    """Devolve um gráfico PNG codificado em base64 para a resposta HTML."""
    base = resultado.drop_duplicates("troco_id").sort_values("troco_id")
    previsao = resultado.sort_values("troco_id")
    with LOCK_GRAFICO:
        fig, eixos = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
        eixos[0].fill_between(
            base["distancia_inicio_km"], base["altitude_media_m"],
            color="#66c2a5", alpha=0.32,
        )
        eixos[0].plot(base["distancia_inicio_km"], base["altitude_media_m"], color="#147d64", linewidth=2)
        eixos[0].set_ylabel("Altitude (m)")
        eixos[0].grid(alpha=0.18)
        eixos[1].axhline(0, color="#8a94a6", linewidth=0.8)
        eixos[1].plot(base["distancia_inicio_km"], base["declive_pct"], color="#d97706", linewidth=1.8)
        eixos[1].set_ylabel("Declive (%)")
        eixos[1].grid(alpha=0.18)
        eixos[2].plot(
            previsao["distancia_inicio_km"], previsao["ritmo_previsto_s_km"] / 60,
            color="#4f46e5", linewidth=2.2,
        )
        pouco = previsao[previsao["pouco_suporte"]]
        eixos[2].scatter(
            pouco["distancia_inicio_km"], pouco["ritmo_previsto_s_km"] / 60,
            color="#dc2626", marker="x", s=25, label="Pouco suporte",
        )
        eixos[2].invert_yaxis()
        eixos[2].set(xlabel="Distância (km)", ylabel="Ritmo (min/km)")
        eixos[2].grid(alpha=0.18)
        if not pouco.empty:
            eixos[2].legend(frameon=False)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def executar_previsao(caminho: Path, nome_apresentacao: str, zona: int) -> dict[str, Any]:
    """Executa o pipeline completo de previsão para uma única zona."""
    pacote = carregar_modelo_atual()
    percurso = carregar_percurso(caminho)
    perfil = perfil_percurso(percurso)
    features = reconstruir_features(perfil, pacote["metadados"]["zonas"], [zona])
    resultado = prever(pacote, features, carregar_treino_atual())
    resumo = criar_resumo(resultado, caminho, MODELO_PADRAO)["zonas"][f"Z{zona}"]
    distancia = float(resumo["distancia_valida_m"])
    pouco_suporte = float(resumo["distancia_pouco_suporte_m"])
    linhas = []
    for linha in resultado.sort_values("troco_id").itertuples():
        linhas.append(
            {
                "inicio_km": f"{linha.distancia_inicio_km:.2f}",
                "altitude_m": f"{linha.altitude_media_m:.0f}",
                "declive_pct": f"{linha.declive_pct:+.1f}%",
                "ritmo": linha.ritmo_previsto_mmss_km,
                "suporte": "Baixo" if linha.pouco_suporte else "Adequado",
            }
        )
    return {
        "nome": nome_apresentacao,
        "zona": zona,
        "distancia_km": distancia / 1000,
        "subida_m": float(perfil["subida_acumulada_m"].max()),
        "ritmo": resumo["ritmo_medio_mmss_km"],
        "tempo": formatar_duracao(float(resumo["tempo_estimado_sem_paragens_s"])),
        "suporte_pct": 100 * max(0.0, distancia - pouco_suporte) / distancia if distancia else 0.0,
        "trocos_invalidos": int(resumo["trocos_invalidos"]),
        "grafico": criar_grafico_previsao(resultado),
        "trocos": linhas,
    }


def contexto_modelo() -> dict[str, Any]:
    """Resume o modelo atual para o cabeçalho da interface."""
    metadados = carregar_modelo_atual()["metadados"]
    metricas = metadados["metricas_oof"]
    return {
        "nome": metadados["modelo_vencedor"],
        "atividades": metadados["n_atividades"],
        "amostras": metadados["n_amostras"],
        "mae": metricas["mae_macro_s_km"],
    }


def processar_pedido_previsao() -> dict[str, Any]:
    """Valida o formulário atual e elimina qualquer upload temporário."""
    temporario: Path | None = None
    try:
        zona = int(request.form.get("zona", "0"))
        if zona not in range(1, 6):
            raise ValueError("A zona cardíaca deve estar entre Z1 e Z5.")
        origem = request.form.get("origem", "existente")
        if origem == "upload":
            ficheiro = request.files.get("percurso_upload")
            if ficheiro is None or not ficheiro.filename:
                raise ValueError("Escolha um ficheiro FIT ou CSV para enviar.")
            nome_seguro = secure_filename(ficheiro.filename)
            extensao = Path(nome_seguro).suffix.lower()
            if extensao not in EXTENSOES_PERMITIDAS:
                raise ValueError("Formato inválido. Utilize um ficheiro FIT ou CSV.")
            with tempfile.NamedTemporaryFile(prefix="percurso_", suffix=extensao, delete=False) as destino:
                ficheiro.save(destino)
                temporario = Path(destino.name)
            caminho = temporario
            nome = nome_seguro
        elif origem == "existente":
            caminho = resolver_percurso_existente(request.form.get("percurso_existente", ""))
            nome = caminho.name
        else:
            raise ValueError("Origem do percurso inválida.")
        return executar_previsao(caminho, nome, zona)
    finally:
        if temporario is not None:
            temporario.unlink(missing_ok=True)


@app.route("/", methods=["GET", "POST"])
def inicio():
    percursos = listar_percursos()
    previsao = None
    erro = None
    zona_selecionada = request.form.get("zona", "3")
    percurso_selecionado = request.form.get("percurso_existente", percursos[0]["nome"] if percursos else "")
    if request.method == "POST":
        try:
            previsao = processar_pedido_previsao()
        except (ValueError, FileNotFoundError, KeyError) as exc:
            erro = str(exc)
        except Exception as exc:  # Mantém a interface utilizável e regista o tipo técnico.
            app.logger.exception("Falha durante a previsão")
            erro = f"Não foi possível processar o percurso ({type(exc).__name__})."
    return render_template(
        "inicio.html",
        pagina="previsao",
        percursos=percursos,
        previsao=previsao,
        erro=erro,
        zona_selecionada=zona_selecionada,
        percurso_selecionado=percurso_selecionado,
        modelo=contexto_modelo(),
    )


@app.post("/api/prever")
def api_prever():
    """Calcula uma previsão sem recarregar a página completa."""
    try:
        previsao = processar_pedido_previsao()
        html = render_template("_resultado.html", previsao=previsao)
        return jsonify(ok=True, html=html)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        return jsonify(ok=False, erro=str(exc)), 400
    except Exception as exc:
        app.logger.exception("Falha durante a previsão assíncrona")
        return jsonify(ok=False, erro=f"Não foi possível processar o percurso ({type(exc).__name__})."), 500


@app.get("/graficos")
def graficos():
    principais = []
    for nome, titulo, descricao in GRAFICOS_PRINCIPAIS:
        caminho = GRAFICOS / nome
        if caminho.exists():
            principais.append(
                {"nome": nome, "titulo": titulo, "descricao": descricao, "versao": caminho.stat().st_mtime_ns}
            )
    atividades = []
    for caminho in sorted(GRAFICOS.glob("atividade_*.png"), reverse=True):
        atividades.append(
            {"nome": caminho.name, "titulo": caminho.stem.replace("atividade_", "Atividade "), "versao": caminho.stat().st_mtime_ns}
        )
    return render_template(
        "graficos.html", pagina="graficos", graficos=principais,
        atividades=atividades, modelo=contexto_modelo(),
    )


@app.get("/ficheiros/graficos/<path:nome>")
def ficheiro_grafico(nome: str):
    permitidos = {item[0] for item in GRAFICOS_PRINCIPAIS}
    permitidos.update(caminho.name for caminho in GRAFICOS.glob("atividade_*.png"))
    if nome not in permitidos:
        abort(404)
    return send_from_directory(GRAFICOS, nome, max_age=0)


@app.errorhandler(413)
def upload_demasiado_grande(_erro):
    if request.path == "/api/prever":
        return jsonify(
            ok=False, erro=f"O ficheiro excede o limite de {LIMITE_UPLOAD_MB} MB."
        ), 413
    return render_template(
        "erro.html", pagina="previsao", codigo=413,
        mensagem=f"O ficheiro excede o limite de {LIMITE_UPLOAD_MB} MB.",
        modelo=contexto_modelo(),
    ), 413


@app.context_processor
def contexto_global():
    return {"ano": datetime.now().year}


if __name__ == "__main__":
    app.run(
        host=os.environ.get("APP_HOST", "127.0.0.1"),
        port=int(os.environ.get("APP_PORT", "5000")),
        debug=False,
    )
