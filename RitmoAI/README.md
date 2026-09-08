# RitmoAI

RitmoAI é o primeiro serviço da plataforma RunningAI. Estima o ritmo de um atleta ao longo de um percurso com base na distância, altimetria normalizada, declive, subida acumulada e zona cardíaca pretendida.

## Conteúdo do projeto

```text
RitmoAI/
├── Dados/
│   ├── brutos/                 # exportações FIT do Garmin Connect
│   └── processados/            # datasets derivados pelo pipeline
├── artefactos/
│   ├── graficos/               # visualizações da análise e do treino
│   ├── metricas/               # resultados da validação
│   ├── modelos/                # modelos treinados, excluídos do Git
│   └── relatorios/             # relatórios gerados
├── configuracao/zonas_fc.json  # limites pessoais das zonas cardíacas
├── lab_orquestrador.py         # execução completa e retomável
├── preparar_dados.py           # construção das amostras
├── treinar_modelos.py          # comparação e treino dos modelos
├── prever_percurso.py          # previsão por linha de comandos
└── RELATORIO_FINAL.md           # metodologia, resultados e limitações
```

`Dados/` contém informação pessoal e ficheiros volumosos, por isso está excluída do Git. Os FIT exportados diretamente do Garmin Connect ficam em `Dados/brutos/`; os CSV e manifestos criados pelo projeto ficam em `Dados/processados/`. Os modelos em `artefactos/modelos/` também são locais.

## Pipeline

O projeto realiza a análise exploratória dos ficheiros FIT, filtra períodos com frequência cardíaca estável numa janela de 40 segundos, constrói troços do percurso e compara vários regressores: Ridge, Random Forest, Gradient Boosting, HistGradientBoosting, SVR e uma rede neuronal MLP. A avaliação usa `LeaveOneGroupOut`, mantendo cada atividade de teste fora do treino.

Foi também testada a substituição das zonas cardíacas por zonas de potência. A melhoria foi pequena e incerta, pelo que o modelo final continua a usar zonas cardíacas. Os resultados dessa experiência estão preservados no relatório e em `artefactos/experiencia_potencia/`.

## Instalação isolada

A partir da raiz do repositório e com um ambiente virtual ativo:

```bash
python -m pip install -r RitmoAI/requirements.txt
mkdir -p RitmoAI/Dados/brutos RitmoAI/Dados/processados
```

Copie os ficheiros `.fit` para `RitmoAI/Dados/brutos/`. Os limites cardíacos podem ser definidos em `configuracao/zonas_fc.json` ou fornecidos ao pipeline.

## Treino

```bash
python RitmoAI/lab_orquestrador.py --reprocessar
```

Para retomar a partir do treino:

```bash
python RitmoAI/lab_orquestrador.py --a-partir treino
```

Para substituir os limites das cinco zonas cardíacas, forneça os seis extremos em ordem crescente:

```bash
python RitmoAI/lab_orquestrador.py \
  --limites-fc 103 124 144 165 185 206 \
  --reprocessar
```

## Previsão por linha de comandos

```bash
python RitmoAI/prever_percurso.py \
  --percurso RitmoAI/Dados/brutos/ATIVIDADE.fit \
  --zona 3
```

Para prever todas as zonas use `--todas-zonas`. Na aplicação RunningAI, o mesmo serviço está disponível na rota `/ritmo-ai` e aceita percursos existentes ou uploads FIT/CSV.

## Resultados atuais

A execução documentada em `RELATORIO_FINAL.md` usa 34 atividades e 5 376 amostras. O Gradient Boosting obteve o menor MAE médio por atividade: 68,58 s/km, uma melhoria de 30,6% face ao baseline.

Este é um modelo pessoal e experimental. A cobertura das zonas é desigual, os dados pertencem a um único atleta e o desempenho varia entre atividades e percursos.

## Documentação

- `RELATORIO_FINAL.md`: metodologia, métricas, gráficos e limitações.
- `Guião de Prompts para Projeto Prático 2 — Previsão de Ritmo por Percurso e Zona Cardíaca.md`: especificação executável do projeto.
- `Proj 2 - Meta Prompt Completo.md`: meta prompt desenvolvido para o módulo.
