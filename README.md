# RunningAI — Serviços inteligentes para corrida

Plataforma Flask preparada para reunir vários serviços de inteligência artificial aplicados à corrida. O primeiro serviço, **RitmoAI**, estima o ritmo de um atleta a partir do perfil do percurso e da zona cardíaca pretendida.

## Funcionalidades

- análise exploratória e auditoria dos ficheiros FIT;
- separação entre dados Garmin brutos e dados processados;
- filtragem de períodos com frequência cardíaca estável numa janela de 40 segundos;
- extração de distância, altitude normalizada, declive e posição no percurso;
- classificação das observações nas zonas cardíacas Z1–Z5;
- comparação de Ridge, Random Forest, Gradient Boosting, HistGradientBoosting, SVR e rede neuronal MLP;
- validação `LeaveOneGroupOut`, mantendo cada atividade de teste fora do treino;
- gráficos de métricas, resíduos, qualidade dos dados e importância das variáveis;
- previsão por linha de comandos ou através de uma interface web responsiva;
- catálogo RunningAI preparado para receber novos serviços.

## Estrutura

```text
.
├── Dados/                         # dados locais, excluídos do Git
│   ├── brutos/                    # ficheiros FIT exportados do Garmin Connect
│   └── processados/               # CSV e manifestos criados pelo pipeline
├── artefactos/
│   ├── graficos/                  # visualizações da análise e do treino
│   ├── metricas/                  # resultados da validação
│   ├── modelos/                   # modelos treinados, excluídos do Git
│   └── relatorios/                # relatório gerado
├── configuracao/zonas_fc.json     # limites pessoais das zonas cardíacas
├── templates/ e static/           # interface Flask
├── lab_orquestrador.py            # execução completa do pipeline
├── prever_percurso.py             # previsão por linha de comandos
└── app.py                         # aplicação web
```

`Dados/` e todo o seu conteúdo estão no `.gitignore` porque podem conter informação pessoal e ficheiros volumosos. Os modelos em `artefactos/modelos/` também não são versionados. Para executar o projeto numa instalação nova é necessário fornecer os próprios ficheiros FIT e treinar o modelo localmente.

## Instalação

Requer Python 3.11 ou superior.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Crie as pastas de dados e copie para `Dados/brutos/` os ficheiros `.fit` exportados do Garmin Connect:

```bash
mkdir -p Dados/brutos Dados/processados
```

## Configuração das zonas cardíacas

Os limites atuais encontram-se em `configuracao/zonas_fc.json`. Podem ser usados diretamente ou substituídos durante a preparação dos dados:

```bash
python lab_orquestrador.py --limites-fc 103 124 144 165 185 206 --reprocessar
```

Os seis valores definem os limites das cinco zonas, por ordem crescente.

## Executar o pipeline

Para realizar a análise inicial, preparar os dados, treinar os modelos, gerar os gráficos e atualizar o relatório:

```bash
python lab_orquestrador.py --reprocessar
```

Também é possível retomar numa etapa específica:

```bash
python lab_orquestrador.py --a-partir treino
```

As etapas disponíveis são `analise_fits`, `preparacao`, `eda_modelo`, `treino`, `visualizacao` e `relatorio`.

## Aplicação web

Depois de treinar o modelo:

```bash
python app.py
```

Abra <http://127.0.0.1:5000> para aceder ao catálogo RunningAI. Selecione **RitmoAI** ou abra diretamente <http://127.0.0.1:5000/ritmo-ai>. No serviço pode:

- escolher um percurso existente em `Dados/brutos/`;
- enviar um ficheiro FIT ou um CSV com as colunas `distancia_m` e `altitude_m`;
- selecionar uma zona cardíaca;
- consultar ritmo médio, tempo estimado, subida acumulada, perfil previsto e suporte dos dados.

Para usar outra porta:

```bash
APP_PORT=8000 python app.py
```

## Previsão pela linha de comandos

```bash
python prever_percurso.py --percurso Dados/brutos/ATIVIDADE.fit --zona 3
```

Para calcular todas as zonas:

```bash
python prever_percurso.py --percurso Dados/brutos/ATIVIDADE.fit --todas-zonas
```

## Resultados atuais

Na execução documentada em `RELATORIO_FINAL.md` foram usadas 34 atividades e 5 376 amostras. O Gradient Boosting obteve o menor MAE médio por atividade, com 68,58 s/km, e uma melhoria de 30,6% face ao modelo baseline.

O resultado deve ser interpretado como uma previsão pessoal e experimental. Os dados pertencem a um único atleta, a cobertura das zonas é desigual e o desempenho varia entre atividades e percursos.

Foi também testada a substituição das zonas cardíacas por zonas de potência. A melhoria observada foi pequena e incerta, pelo que o modelo e a aplicação continuam a usar exclusivamente zonas cardíacas. A experiência e as métricas estão documentadas em `RELATORIO_FINAL.md`.

## Documentação

- `RELATORIO_FINAL.md`: metodologia, métricas, gráficos e limitações;
- `INSTRUCOES_APP.md`: instruções rápidas da aplicação;
- `Guião de Prompts para Projeto Prático 2 — Previsão de Ritmo por Percurso e Zona Cardíaca.md`: especificação usada para desenvolver o projeto.
