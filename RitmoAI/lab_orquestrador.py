"""Orquestrador retomável do Projeto Prático 2."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


RAIZ = Path(__file__).resolve().parent
LOG = RAIZ / "artefactos" / "logs" / "execucao.log"
ETAPAS = [
    ("analise_fits", "analisar_fits.py"),
    ("preparacao", "preparar_dados.py"),
    ("eda_modelo", "analisar_dataset_modelo.py"),
    ("treino", "treinar_modelos.py"),
    ("visualizacao", "visualizar_resultados.py"),
    ("relatorio", "gerar_relatorio.py"),
]
DEPENDENCIAS = {
    "garmin_fit_sdk": "garmin-fit-sdk>=21.214.0",
    "pandas": "pandas>=2.2",
    "numpy": "numpy>=1.26",
    "sklearn": "scikit-learn>=1.4",
    "matplotlib": "matplotlib>=3.8",
    "seaborn": "seaborn>=0.13",
    "joblib": "joblib>=1.3",
    "threadpoolctl": "threadpoolctl>=3.2",
    "tabulate": "tabulate>=0.9",
}


def escrever_requisitos() -> None:
    """Cria requirements.txt sem instalar dependências."""
    conteudo = "\n".join(DEPENDENCIAS.values()) + "\n"
    (RAIZ / "requirements.txt").write_text(conteudo, encoding="utf-8")


def dependencias_em_falta() -> list[str]:
    """Devolve pacotes necessários ainda indisponíveis."""
    return [pacote for modulo, pacote in DEPENDENCIAS.items() if importlib.util.find_spec(modulo) is None]


def validar_pre_condicoes(etapa: str, args: argparse.Namespace) -> None:
    """Valida entradas mínimas antes de cada etapa."""
    if etapa == "analise_fits" and not list((RAIZ / "Dados" / "brutos").glob("*.fit")):
        raise FileNotFoundError("Não existem FIT em Dados/brutos/")
    if etapa == "preparacao":
        if not (RAIZ / "Dados" / "processados" / "registos_normalizados.csv").exists():
            raise FileNotFoundError("Falta registos_normalizados.csv; execute a análise FIT")
        config = RAIZ / "configuracao" / "zonas_fc.json"
        if not config.exists() and args.idade is None and args.limites_fc is None:
            raise FileNotFoundError("Falta configuração de zonas ou argumentos --idade/--limites-fc")
    if etapa == "eda_modelo" and not (RAIZ / "Dados" / "processados" / "amostras_modelo.csv").exists():
        raise FileNotFoundError("Falta amostras_modelo.csv")
    if etapa == "treino" and not (RAIZ / "Dados" / "processados" / "manifesto_processamento.json").exists():
        raise FileNotFoundError("Falta manifesto_processamento.json")
    if etapa == "visualizacao" and not (RAIZ / "artefactos" / "metricas" / "resumo_modelos.csv").exists():
        raise FileNotFoundError("Faltam métricas do treino")


def comando_etapa(etapa: str, script: str, args: argparse.Namespace) -> list[str]:
    """Monta o comando da etapa e repassa apenas argumentos aplicáveis."""
    comando = [sys.executable, str(RAIZ / script)]
    if etapa == "preparacao":
        if args.idade is not None:
            comando.extend(["--idade", str(args.idade)])
        if args.limites_fc:
            comando.extend(["--limites-fc", *map(str, args.limites_fc)])
        if args.reprocessar:
            comando.append("--reprocessar")
    return comando


def executar_etapa(
    nome: str, comando: list[str], ambiente: dict[str, str]
) -> tuple[bool, float, str]:
    """Executa uma etapa e devolve sucesso, duração e output."""
    inicio = time.perf_counter()
    processo = subprocess.run(
        comando,
        cwd=RAIZ,
        text=True,
        capture_output=True,
        encoding="utf-8",
        env=ambiente,
        check=False,
    )
    duracao = time.perf_counter() - inicio
    output = processo.stdout
    if processo.stderr:
        output += "\n[stderr]\n" + processo.stderr
    return processo.returncode == 0, duracao, output


def executar_previsao(args: argparse.Namespace, ambiente: dict[str, str]) -> tuple[bool, float, str]:
    """Executa a previsão opcional antes do relatório."""
    comando = [sys.executable, str(RAIZ / "prever_percurso.py"), "--percurso", str(args.percurso)]
    if args.todas_zonas:
        comando.append("--todas-zonas")
    elif args.zona is not None:
        comando.extend(["--zona", str(args.zona)])
    else:
        raise ValueError("A previsão exige --zona ou --todas-zonas")
    return executar_etapa("previsao", comando, ambiente)


def registar(texto: str) -> None:
    """Acrescenta uma mensagem ao log e mostra-a na consola."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as ficheiro:
        ficheiro.write(texto.rstrip() + "\n")
    print(texto)


def selecionar_etapas(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Aplica filtros de seleção ou retoma."""
    etapas = ETAPAS
    if args.a_partir:
        nomes = [nome for nome, _ in etapas]
        if args.a_partir not in nomes:
            raise ValueError(f"Etapa desconhecida: {args.a_partir}")
        etapas = etapas[nomes.index(args.a_partir) :]
    if args.etapas:
        pedidas = set(args.etapas.split(","))
        desconhecidas = pedidas - {nome for nome, _ in ETAPAS}
        if desconhecidas:
            raise ValueError(f"Etapas desconhecidas: {sorted(desconhecidas)}")
        etapas = [item for item in etapas if item[0] in pedidas]
    return etapas


def main() -> None:
    """Valida dependências e executa o pipeline pela ordem correta."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etapas", help="Nomes separados por vírgulas")
    parser.add_argument("--a-partir", choices=[nome for nome, _ in ETAPAS])
    parser.add_argument("--idade", type=int)
    parser.add_argument("--limites-fc", nargs=6, type=int)
    parser.add_argument("--reprocessar", action="store_true")
    parser.add_argument("--percurso", type=Path)
    parser.add_argument("--zona", type=int, choices=range(1, 6))
    parser.add_argument("--todas-zonas", action="store_true")
    parser.add_argument("--continuar-apos-erro", action="store_true")
    args = parser.parse_args()
    escrever_requisitos()
    falta = dependencias_em_falta()
    if falta:
        print("Dependências em falta:", ", ".join(falta))
        print(f"Instale com: {sys.executable} -m pip install -r requirements.txt")
        raise SystemExit(2)
    ambiente = os.environ.copy()
    mpl = RAIZ / "artefactos" / ".mplconfig"
    mpl.mkdir(parents=True, exist_ok=True)
    ambiente["MPLCONFIGDIR"] = str(mpl)
    etapas = selecionar_etapas(args)
    inicio_execucao = datetime.now().isoformat(timespec="seconds")
    registar(f"\n=== Execução iniciada em {inicio_execucao} ===")
    resultados: list[dict[str, Any]] = []
    previsao_executada = False
    for nome, script in etapas:
        if nome == "relatorio" and args.percurso is not None and not previsao_executada:
            sucesso, duracao, output = executar_previsao(args, ambiente)
            registar(f"\n--- previsão ({duracao:.2f}s) | {'OK' if sucesso else 'FALHOU'} ---\n{output}")
            resultados.append({"etapa": "previsao", "sucesso": sucesso, "duracao_s": duracao})
            previsao_executada = True
            if not sucesso and not args.continuar_apos_erro:
                raise SystemExit(1)
        try:
            validar_pre_condicoes(nome, args)
            sucesso, duracao, output = executar_etapa(nome, comando_etapa(nome, script, args), ambiente)
        except Exception as erro:
            sucesso, duracao, output = False, 0.0, str(erro)
        registar(f"\n--- {nome} ({duracao:.2f}s) | {'OK' if sucesso else 'FALHOU'} ---\n{output}")
        resultados.append({"etapa": nome, "sucesso": sucesso, "duracao_s": duracao})
        if not sucesso and not args.continuar_apos_erro:
            registar("Execução interrompida após falha.")
            raise SystemExit(1)
    if args.percurso is not None and not previsao_executada:
        sucesso, duracao, output = executar_previsao(args, ambiente)
        registar(f"\n--- previsão ({duracao:.2f}s) | {'OK' if sucesso else 'FALHOU'} ---\n{output}")
        resultados.append({"etapa": "previsao", "sucesso": sucesso, "duracao_s": duracao})
    total = sum(item["duracao_s"] for item in resultados)
    falhas = [item["etapa"] for item in resultados if not item["sucesso"]]
    registar(f"\n=== Resumo: {len(resultados) - len(falhas)} OK, {len(falhas)} falhas, {total:.2f}s ===")
    if falhas:
        registar(f"Falhas: {', '.join(falhas)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
