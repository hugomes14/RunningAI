# Guião de Prompts para Projeto Prático 2 — Previsão de Ritmo por Percurso e Zona Cardíaca

## 📚 Introdução ao Prompt Engineering

Este guião aplica cinco princípios em cada um dos oito prompts:

- **Sê específico.** Cada prompt indica o nome exato do ficheiro, das funções, das colunas e das fórmulas. Um LLM que recebe "cria um modelo de previsão de ritmo" produz código genérico e incoerente com as etapas anteriores; um LLM que recebe a fórmula exata do alvo, a ordem exata das features e os nomes exatos dos ficheiros produz código que encaixa no resto do pipeline.
- **Dá contexto.** Cada prompt refere de onde vêm os dados, que etapa os produziu e para que etapa seguem. Isto evita que o LLM invente formatos de entrada ou reprocesse dados já tratados.
- **Pede exemplos.** Cada prompt pede prints informativos, mensagens de erro claras e pelo menos um exemplo de execução na consola, para que o resultado seja verificável sem abrir o código.
- **Itera.** O guião está dividido em oito etapas encadeadas em vez de um único pedido monolítico. Cada etapa pode ser corrigida e repetida antes de avançar para a seguinte, sem repetir trabalho já validado.
- **Estrutura a tarefa.** Cada prompt separa entradas, funções, regras técnicas, saídas e critérios de aceitação, para que o LLM não misture responsabilidades num único bloco de código difícil de rever.

## Assunções e Inferências

- **Dados disponíveis.** Só existem ficheiros `.fit` em `Dados/brutos/`, exportados do Garmin Connect, possivelmente acompanhados de ficheiros `:Zone.Identifier` do Windows, que devem ser ignorados. Não se assume nenhum outro formato de origem (não há GPX, TCX ou exportações de outros relógios).
- **Valores pessoais em falta.** Idade e limites pessoais de frequência cardíaca não estão disponíveis por defeito. O guião nunca inventa estes valores: pede-os como argumentos opcionais (`--idade`, `--limites-fc`) e, na ausência de configuração válida, interrompe a preparação de dados com um diagnóstico claro em vez de continuar com valores fictícios.
- **Unidades.** Distâncias em metros nos cálculos internos e em quilómetros nalgumas features (`distancia_inicio_km`); altitude em metros; frequência cardíaca em bpm; declive em percentagem; ritmo em segundos por quilómetro internamente (`s/km`) e em `mm:ss/km` na apresentação ao utilizador; tempos de treino e previsão em segundos.
- **Algoritmos.** Assume-se `scikit-learn` já disponível no ambiente, com suporte para `Ridge`, `RandomForestRegressor`, `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `SVR`, `MLPRegressor` e `DummyRegressor`. Se a versão instalada não aceitar `sample_weight` nalgum destes estimadores, o guião pede que essa limitação seja registada em vez de ocultada, sem impedir a comparação dos restantes modelos.
- **Validação.** Assume-se pelo menos duas atividades de corrida válidas nos dados crus, condição mínima para `LeaveOneGroupOut`. Se essa condição não se verificar, os prompts pedem um diagnóstico sem métricas inventadas em vez de uma validação forçada.
- **Artefactos.** Assume-se que cada script cria as pastas de que necessita antes de escrever ficheiros e que a estrutura de pastas indicada abaixo é criada de forma incremental, prompt a prompt, sem exigir que o utilizador a prepare manualmente.

```text
Dados/
  brutos/
    *.fit
    *.fit:Zone.Identifier
  processados/
    registos_normalizados.csv
    intervalos_auditados.csv
    trocos_percurso.csv
    amostras_modelo.csv
    qualidade_dados.csv
    manifesto_processamento.json
configuracao/
  zonas_fc.json
artefactos/
  analise_inicial/
  analise_dataset/
  modelos/folds/
  modelos/final/
  metricas/
  previsoes_cv/
  previsoes_percurso/
  graficos/
  relatorios/
  logs/
RELATORIO_FINAL.md
```

---

## 🔍 PROMPT 1 — Análise exploratória inicial dos ficheiros FIT

### O que vais aprender

- Descodificar ficheiros `.fit` do Garmin com `garmin-fit-sdk` e validar a sua integridade.
- Normalizar registos de atividades heterogéneas para um esquema comum.
- Distinguir problemas de qualidade de dados (pausas, recuos, saltos, valores em falta) antes de qualquer modelação.
- Produzir uma análise exploratória visual e estatística que sustenta decisões nas etapas seguintes.

### Prompt pronto a copiar

```text
Cria o ficheiro Python `analisar_fits.py` para a análise exploratória inicial de
atividades de corrida em ficheiros Garmin FIT.

CONTEXTO
Os dados crus estão em `Dados/brutos/*.fit`, exportações diretas do Garmin
Connect. Podem existir ficheiros auxiliares com sufixo `:Zone.Identifier`
(metadados de proveniência do Windows) que devem ser ignorados. Este é o
primeiro script do pipeline: não recebe nenhum artefacto de etapas
anteriores.

FUNÇÕES OBRIGATÓRIAS
- `ler_fit(caminho: Path) -> dict`: usa `garmin_fit_sdk.Decoder` e
  `garmin_fit_sdk.Stream` para ler o ficheiro, chama `check_integrity()` e
  trata os erros devolvidos por `read()`. Devolve as mensagens FIT
  descodificadas ou lança uma exceção clara e informativa.
- `normalizar_registos(mensagens: dict, atividade_id: str) -> pandas.DataFrame`:
  produz um DataFrame com as colunas `atividade_id`, `timestamp`,
  `distancia_m`, `altitude_m`, `fc_bpm`, `velocidade_fit_m_s`. Prefere campos
  `enhanced_altitude`/`enhanced_speed` quando existirem e regista no log
  quando recorre aos campos normais como fallback.
- `validar_atividade(mensagens: dict) -> dict`: confirma que a atividade é
  corrida ou uma variante de corrida reconhecida pelo FIT; separa sessões
  quando existir mais de uma no mesmo ficheiro; se não for possível separar,
  marca o ficheiro para exclusão com o motivo. Regista e ignora atividades de
  outros desportos, com justificação explícita no log.
- `resumir_atividade(df: pandas.DataFrame) -> dict`: calcula data, distância
  total, duração, tempo em movimento, subida acumulada aproximada, ritmo
  médio, FC média, frequência de amostragem dos registos e percentagem de
  cobertura de cada campo (`altitude_m`, `fc_bpm`, `velocidade_fit_m_s`).
- `visualizar_atividade(df: pandas.DataFrame, pasta_saida: Path) -> None`:
  gráfico de altitude, ritmo e FC ao longo do tempo e da distância, para cada
  atividade.
- `visualizar_qualidade(resumo: pandas.DataFrame, pasta_saida: Path) -> None`:
  gráfico de ausências e problemas de qualidade por atividade.
- `main() -> None`: orquestra a leitura de todos os `Dados/brutos/*.fit`
  (ignorando `:Zone.Identifier`), sem nomes fixos no código, e grava todos os
  artefactos.

REGRAS TÉCNICAS
- Usa apenas `Dados/brutos/*.fit` como entrada; nunca escreve, renomeia ou
  apaga ficheiros dessa pasta.
- Deteta e regista separadamente: valores em falta, timestamps duplicados,
  pausas, recuos de distância, saltos temporais e saltos de distância
  anómalos, e valores fisiologicamente impossíveis de FC ou altitude.
- Gera histogramas globais de FC e de ritmo, além dos gráficos por
  atividade.
- Usa `pathlib` para todos os caminhos, cria as pastas de saída antes de
  escrever, usa seed 42 sempre que houver aleatoriedade, guarda as figuras
  com pelo menos 200 dpi e fecha todas as figuras depois de gravar.
- Usa type hints, docstrings e comentários úteis em português europeu, sem
  gerúndios.
- Termina com `if __name__ == "__main__": main()`; não cries classes,
  notebooks ou ficheiros adicionais.
- As conclusões impressas na consola devem derivar exclusivamente dos dados
  processados nesta execução, nunca de valores inventados.

SAÍDAS
- `Dados/processados/registos_normalizados.csv` com os registos normalizados
  de todas as atividades válidas.
- `artefactos/analise_inicial/resumo_atividades.csv` com o resumo por
  atividade.
- Gráficos por atividade e histogramas globais em `artefactos/graficos/`.
- Prints informativos no final com o número de atividades processadas,
  excluídas e a razão de cada exclusão.

Mostra no fim um exemplo do resumo impresso na consola para uma atividade.
```

### Ficheiro a criar

`analisar_fits.py`

### Entradas e artefactos necessários

- `Dados/brutos/*.fit` (dados crus, nenhum artefacto anterior é necessário nesta etapa).

### Funções obrigatórias e responsabilidades

`ler_fit`, `normalizar_registos`, `validar_atividade`, `resumir_atividade`, `visualizar_atividade`, `visualizar_qualidade`, `main`.

### Bibliotecas, regras técnicas e validações

`garmin-fit-sdk` (`Decoder`, `Stream`), `pandas`, `matplotlib`, `pathlib`; validação de integridade, sessões, timestamps, unidades e cobertura de campos; nenhuma escrita em `Dados/brutos/`.

### Saídas e critérios de aceitação verificáveis

- `Dados/processados/registos_normalizados.csv` existe e contém todas as atividades válidas com as seis colunas normalizadas.
- `artefactos/analise_inicial/resumo_atividades.csv` tem uma linha por atividade válida.
- Existem gráficos por atividade e histogramas globais em `artefactos/graficos/`.
- A consola mostra o número de atividades processadas, excluídas e o motivo de cada exclusão.

### Após receber o código

1. Guarda o ficheiro como `analisar_fits.py` na raiz do projeto.
2. Confirma que `Dados/brutos/` contém pelo menos um `.fit` real.
3. Executa `python analisar_fits.py` e confirma que a pasta `Dados/processados/` e `artefactos/analise_inicial/` são criadas.
4. Abre `registos_normalizados.csv` e verifica as seis colunas esperadas.
5. Verifica visualmente pelo menos um gráfico por atividade e os histogramas globais.

---

## 🧭 PROMPT 2 — Perfil, zonas cardíacas e preparação dos dados

### O que vais aprender

- Construir o perfil altimétrico de um percurso a partir apenas de distância e altitude.
- Classificar intervalos históricos em zonas cardíacas sem usar a FC como feature do modelo.
- Aplicar um filtro causal de estabilidade cardíaca sem consultar dados futuros.
- Agregar intervalos válidos em amostras de modelação com pesos por atividade.

### Prompt pronto a copiar

```text
Cria o ficheiro Python `preparar_dados.py`, que prepara o dataset de
modelação a partir dos registos normalizados da etapa anterior.

CONTEXTO
Importa `ler_fit` e `normalizar_registos` de `analisar_fits.py`. Este script
lê `Dados/brutos/*.fit` outra vez apenas se for necessário reconstruir os
dados; caso `Dados/processados/registos_normalizados.csv` já exista e
corresponda aos hashes atuais dos FIT, pode reaproveitá-lo. O objetivo final
é `Dados/processados/amostras_modelo.csv`, usado por todas as etapas
seguintes.

FUNÇÕES OBRIGATÓRIAS
- `carregar_ou_criar_zonas(caminho_json: Path, idade: int | None,
  limites_fc: dict | None) -> dict`: lê `configuracao/zonas_fc.json`. Se
  existirem limites pessoais fornecidos por argumento, usa-os com
  prioridade. Se só existir idade, estima `fc_max = 220 - idade` e cria
  Z1–Z5 entre 50% e 100% de `fc_max`, com seis fronteiras arredondadas,
  documentando que é uma aproximação. Se o ficheiro não existir ou estiver
  incompleto e não houver argumentos suficientes, cria um modelo com valores
  `null`, mostra um exemplo válido no ecrã e termina com um código de saída
  diferente de zero, sem inventar idade nem limites.
- `validar_zonas(zonas: dict) -> None`: confirma que os limites são
  numéricos, crescentes, não sobrepostos e coerentes nas fronteiras entre
  zonas consecutivas; lança erro claro caso contrário.
- `classificar_zona(fc_bpm: float, zonas: dict) -> int`: devolve a zona de 1
  a 5 correspondente a uma FC, respeitando a zona aberta sem limite
  superior.
- `perfil_percurso(df_atividade: pandas.DataFrame) -> pandas.DataFrame`:
  usa apenas distância e altitude. Ordena por distância, trata distâncias
  repetidas de forma explícita, interpola a altitude numa grelha regular de
  10 m, suaviza com média móvel centrada de cinco pontos (explica em
  comentário porque não constitui fuga de dados) e divide o percurso em
  troços de 50 m, com o último troço a guardar o seu comprimento real.
  Calcula `declive_pct`, `altitude_media_m`, `altitude_normalizada`,
  `distancia_inicio_km` e `subida_acumulada_m` por troço. Calcula
  `altitude_normalizada = altitude_media_m / 1000`, usa esta versão no
  modelo e preserva a absoluta para auditoria e gráficos. A escala fixa
  mantém diferenças de altitude entre percursos e evita fuga de validação.
- `criar_intervalos(df_atividade: pandas.DataFrame) -> pandas.DataFrame`:
  ordena por timestamp, calcula `dt`, `dx` e o ponto médio da distância entre
  registos consecutivos e associa cada intervalo ao troço correspondente.
- `estabilidade_fc(df_atividade: pandas.DataFrame) -> pandas.DataFrame`:
  aplica o filtro causal descrito abaixo, guarda a máscara de estabilidade e
  o motivo de rejeição, e nunca consulta FC futura.
- `agregar_amostras(intervalos: pandas.DataFrame, trocos: pandas.DataFrame,
  zonas: dict) -> pandas.DataFrame`: classifica cada intervalo válido na
  zona antes de agregar por `atividade_id`, `troco_id` e `zona_ordem`,
  mantém apenas grupos com pelo menos dois intervalos válidos e cobertura
  entre 60% e 120% do comprimento do troço, e calcula o alvo pelas somas de
  distância e tempo do grupo.
- `calcular_pesos_atividade(amostras: pandas.DataFrame) -> pandas.DataFrame`:
  adiciona uma coluna de peso por amostra, de forma que cada atividade tenha
  o mesmo peso total.
- `main() -> None`: orquestra tudo, aceita `--idade` e `--limites-fc` como
  argumentos opcionais e grava todos os artefactos.

REGRA DE VALIDAÇÃO DE INTERVALOS
Um intervalo é válido apenas se `0 < dt <= 15 s`, `dx > 0`, `dx <= 10 × dt`,
`fc_bpm > 0` e `dx/dt >= 0.5 m/s`. Regista como motivos separados:
timestamp inválido ou repetido, pausa, recuo de distância, salto temporal,
salto de distância, FC ausente ou não positiva e velocidade inferior ao
limiar.

REGRA DE ESTABILIDADE CARDÍACA CAUSAL
Em cada instante, usa apenas os 40 segundos anteriores, incluindo o instante
atual. Exige que as observações cubram pelo menos 30 segundos, com pelo menos
quatro medições, amplitude robusta entre os percentis 10 e 90 até 15 bpm,
diferença absoluta entre a FC atual e a média temporal da janela até 8 bpm,
e pelo menos 120 segundos desde o início da atividade. Reinicia a janela
após um intervalo inválido ou uma falha superior a 15 segundos. Estes
critérios devem funcionar com gravação a cada segundo e com gravação
inteligente do Garmin. A escolha de 40 segundos deve ser sustentada por uma
análise de sensibilidade contra janelas de 60 e 30 segundos.

FÓRMULA DO ALVO
`ritmo_alvo_s_km = 1000 × soma(dt_valido) / soma(dx_valido)` por grupo. Nunca
uses a média simples do ritmo nem o campo de velocidade do FIT para o alvo.

FEATURES PERMITIDAS, NESTA ORDEM EXATA
`zona_ordem`, `zona_limite_inferior_bpm`, `zona_limite_superior_bpm`,
`zona_aberta`, `declive_pct`, `altitude_normalizada`, `distancia_inicio_km`,
`subida_acumulada_m`, `comprimento_troco_m`.

REGRAS TÉCNICAS
- Não remove outliers automaticamente nesta etapa; apenas identifica e
  regista.
- Cria ou atualiza `Dados/processados/manifesto_processamento.json` com os
  caminhos e hashes SHA-256 dos FIT de origem, data de geração, parâmetros
  de processamento, configuração de zonas e nomes dos datasets produzidos.
- Marca os datasets processados como desatualizados e regenera-os sempre
  que os hashes dos FIT, a configuração de zonas ou os parâmetros do perfil
  mudarem.
- Usa `pathlib`, seed 42, cria pastas antes de escrever, type hints,
  docstrings, comentários em português europeu sem gerúndios.
- Termina com `if __name__ == "__main__": main()`.

SAÍDAS
- `Dados/processados/trocos_percurso.csv`
- `Dados/processados/intervalos_auditados.csv`, com colunas `valido` e
  `motivo_exclusao` para todas as linhas, incluindo as inválidas.
- `Dados/processados/amostras_modelo.csv`, com as nove features na ordem
  exata, o alvo e os identificadores, sem FC exata.
- `Dados/processados/qualidade_dados.csv`
- `Dados/processados/manifesto_processamento.json`

No fim, imprime uma confirmação explícita de que `amostras_modelo.csv`
contém exatamente as nove features permitidas, o alvo e os identificadores,
sem nenhuma coluna de FC exata, velocidade ou potência.
```

### Ficheiro a criar

`preparar_dados.py`

### Entradas e artefactos necessários

- `Dados/brutos/*.fit`, `Dados/processados/registos_normalizados.csv` (se reaproveitado), `configuracao/zonas_fc.json` (se existir), argumentos `--idade` e `--limites-fc`.

### Funções obrigatórias e responsabilidades

`carregar_ou_criar_zonas`, `validar_zonas`, `classificar_zona`, `perfil_percurso`, `criar_intervalos`, `estabilidade_fc`, `agregar_amostras`, `calcular_pesos_atividade`, `main`.

### Bibliotecas, regras técnicas e validações

`pandas`, `numpy`, `hashlib` (para SHA-256), `json`, `pathlib`; regras literais de perfil, intervalos, estabilidade e agregação descritas no contrato; nenhuma FC exata nas features finais.

### Saídas e critérios de aceitação verificáveis

- Os quatro CSV e o manifesto existem em `Dados/processados/`.
- `amostras_modelo.csv` tem exatamente as nove features na ordem contratada, mais o alvo e identificadores.
- `intervalos_auditados.csv` preserva linhas inválidas com motivo de exclusão.
- Sem `--idade` nem `--limites-fc` e sem `zonas_fc.json` válido, o script termina com código de saída diferente de zero e sem gerar `amostras_modelo.csv`.

### Após receber o código

1. Guarda o ficheiro como `preparar_dados.py` na raiz do projeto, ao lado de `analisar_fits.py`.
2. Executa primeiro sem `configuracao/zonas_fc.json` e confirma que o script para com um diagnóstico claro em vez de inventar zonas.
3. Cria `configuracao/zonas_fc.json` (ou usa `--idade`) e volta a executar.
4. Confirma as cinco saídas em `Dados/processados/`.
5. Abre `amostras_modelo.csv` e confirma a ausência de qualquer coluna de FC exata, velocidade ou potência.

---

## 📊 PROMPT 3 — Análise exploratória do dataset de modelação

### O que vais aprender

- Validar um dataset de modelação antes de o usar em treino.
- Analisar distribuições, relações e colinearidade entre features de corrida.
- Identificar outliers sem os remover automaticamente.
- Avaliar a cobertura do espaço de features por atividade.

### Prompt pronto a copiar

```text
Cria o ficheiro Python `analisar_dataset_modelo.py`, que faz a análise
exploratória do dataset final de modelação.

CONTEXTO
Lê exclusivamente `Dados/processados/amostras_modelo.csv`, produzido por
`preparar_dados.py`, depois de validar que o `manifesto_processamento.json`
correspondente existe e está atualizado. Não relê os FIT nem os registos
normalizados.

FUNÇÕES OBRIGATÓRIAS
- `validar_dataset(df: pandas.DataFrame, manifesto: dict) -> None`:
  confirma dimensão, tipos, ausências, duplicados, presença das nove
  features na ordem esperada, do alvo e dos identificadores; lança erro
  claro se o manifesto estiver desatualizado face ao ficheiro.
- `analisar_distribuicoes(df: pandas.DataFrame, pasta_saida: Path) -> None`:
  estatísticas e histogramas de ritmo, declive, altitude, distância e
  subida acumulada.
- `analisar_por_zona(df: pandas.DataFrame, pasta_saida: Path) -> None`:
  contagem de amostras por atividade e por zona, boxplot de ritmo por zona
  e por atividade.
- `analisar_relacoes(df: pandas.DataFrame, pasta_saida: Path) -> None`:
  relação entre ritmo e declive com cor por zona, matriz de correlação
  numérica e identificação de pares de features colineares.
- `detetar_outliers(df: pandas.DataFrame) -> pandas.DataFrame`: identifica
  outliers por IQR em ritmo e declive, sem os remover; devolve uma tabela
  com os candidatos e a razão.
- `criar_resumo_eda(...) -> None`: escreve `resumo_eda.md` com as
  conclusões derivadas apenas dos dados analisados nesta execução,
  incluindo a análise da cobertura do espaço de features por atividade
  (que combinações de zona/declive/distância cada atividade cobre).
- `main() -> None`: orquestra a leitura, validação, análises e escrita dos
  artefactos.

REGRAS TÉCNICAS
- Nunca remove outliers automaticamente; apenas identifica, visualiza e
  documenta o impacto potencial.
- Usa `pathlib`, cria pastas antes de escrever, guarda gráficos com pelo
  menos 200 dpi, fecha todas as figuras, usa seed 42 quando aplicável.
- Type hints, docstrings, comentários em português europeu sem gerúndios.
- Termina com `if __name__ == "__main__": main()`.

SAÍDAS
- Tabelas de distribuições, correlações e outliers em
  `artefactos/analise_dataset/`.
- `artefactos/analise_dataset/resumo_eda.md`.
- Gráficos obrigatórios desta etapa em `artefactos/graficos/`:
  `distribuicoes_dataset.png`, `ritmo_por_zona.png`,
  `ritmo_declive_zona.png`.

Mostra no fim um resumo impresso com o número de amostras, atividades,
zonas cobertas e outliers detetados.
```

### Ficheiro a criar

`analisar_dataset_modelo.py`

### Entradas e artefactos necessários

- `Dados/processados/amostras_modelo.csv` e `Dados/processados/manifesto_processamento.json`.

### Funções obrigatórias e responsabilidades

`validar_dataset`, `analisar_distribuicoes`, `analisar_por_zona`, `analisar_relacoes`, `detetar_outliers`, `criar_resumo_eda`, `main`.

### Bibliotecas, regras técnicas e validações

`pandas`, `numpy`, `matplotlib`; validação do dataset e do manifesto antes de qualquer análise; deteção de outliers por IQR sem remoção automática.

### Saídas e critérios de aceitação verificáveis

- `resumo_eda.md` existe e reflete apenas números calculados nesta execução.
- Os três gráficos obrigatórios desta etapa existem em `artefactos/graficos/`.
- A tabela de outliers existe e não altera `amostras_modelo.csv`.

### Após receber o código

1. Guarda como `analisar_dataset_modelo.py` na raiz do projeto.
2. Executa `python analisar_dataset_modelo.py` depois de confirmar que `amostras_modelo.csv` existe.
3. Abre `resumo_eda.md` e confirma que os números batem com o CSV.
4. Verifica os três gráficos e a tabela de outliers.

---

## 🤖 PROMPT 4 — Treino, validação e seleção dos modelos

### O que vais aprender

- Validar modelos de séries por grupos completos com `LeaveOneGroupOut`.
- Comparar vários algoritmos de regressão sob as mesmas condições exatas.
- Aplicar pesos por atividade e restrições monotónicas de forma correta.
- Selecionar e persistir um único modelo final de forma reprodutível.

### Prompt pronto a copiar

```text
Cria o ficheiro Python `treinar_modelos.py`, que treina, valida e seleciona
o modelo final de previsão de ritmo.

CONTEXTO
Lê `Dados/processados/amostras_modelo.csv`. As features entram nos modelos
exatamente por esta ordem: `zona_ordem`, `zona_limite_inferior_bpm`,
`zona_limite_superior_bpm`, `zona_aberta`, `declive_pct`,
`altitude_normalizada`, `distancia_inicio_km`, `subida_acumulada_m`,
`comprimento_troco_m`. O alvo é `ritmo_alvo_s_km`. O agrupamento para
validação é `atividade_id`.

FUNÇÕES OBRIGATÓRIAS
- `criar_modelos() -> dict`: devolve os seis candidatos com os parâmetros
  fixos abaixo, sem tuning no conjunto de teste.
- `calcular_pesos(df: pandas.DataFrame) -> numpy.ndarray`: pesos iguais por
  atividade, com pesos iguais dentro de cada atividade.
- `calcular_metricas(y_real, y_previsto, pesos=None) -> dict`: MAE e RMSE em
  `s/km`, R² como diagnóstico secundário, erro médio assinado, e as mesmas
  métricas convertidas para `mm:ss/km` quando aplicável.
- `avaliar_logo(X, y, grupos, pesos, modelos) -> pandas.DataFrame`: executa
  `LeaveOneGroupOut`, treina e avalia cada modelo em cada fold com os mesmos
  folds, linhas e pesos, ajusta qualquer scaler apenas no treino do fold,
  mede tempo de treino e de previsão, e guarda todas as previsões fora do
  treino.
- `selecionar_vencedor(resultados: pandas.DataFrame) -> str`: escolhe o
  menor MAE macro fora do treino; em empate usa o menor RMSE macro e depois
  o menor tempo médio de previsão; o `DummyRegressor` nunca pode vencer.
- `treinar_final(nome_modelo, X, y, pesos) -> object`: treina o modelo
  vencedor com todas as atividades, só depois de a comparação estar
  concluída.
- `guardar_pacote_modelo(modelo, metadados: dict, caminho: Path) -> None`:
  guarda em Joblib o pipeline completo, a ordem das features, a
  configuração de zonas, os parâmetros do perfil, as unidades, as métricas
  fora do treino e as versões das bibliotecas usadas.
- `main() -> None`: orquestra tudo e grava as saídas.

MODELOS E PARÂMETROS FIXOS
- `Pipeline(StandardScaler(), Ridge(alpha=10))`
- `RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
  max_features=0.8, random_state=42, n_jobs=-1)`
- `GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
  max_depth=2, min_samples_leaf=5, loss="huber", random_state=42)`
- `HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15,
  min_samples_leaf=20, learning_rate=0.05, l2_regularization=5,
  monotonic_cst=[-1, -1, -1, 0, 1, 0, 0, 0, 0], random_state=42)`, com o
  vetor de restrições validado no código contra a ordem real das nove
  colunas: aumentar a zona ou os seus limites não deve aumentar o ritmo em
  s/km; aumentar o declive não deve reduzir o ritmo; as restantes features
  ficam sem restrição.
- `Pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=10,
  gamma="scale"))`
- `Pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16),
  activation="relu", solver="adam", alpha=0.01, batch_size=64,
  learning_rate_init=0.001, max_iter=1000, early_stopping=True,
  validation_fraction=0.15, n_iter_no_change=40, random_state=42))`
- `DummyRegressor(strategy="mean")`

REGRAS TÉCNICAS
- Nunca uses `train_test_split` nem K-Fold aleatório; usa sempre
  `LeaveOneGroupOut` por `atividade_id`, com cada fold a testar uma
  atividade completa.
- Passa `sample_weight` a todos os estimadores que o suportem, com o nome
  correto da etapa em cada pipeline; se uma versão instalada não suportar
  pesos nalgum estimador, regista a limitação no log e continua a
  comparação sem ocultar o aviso.
- Se existirem menos de duas atividades elegíveis, não executa a validação
  e mostra um diagnóstico sem métricas inventadas.
- Nunca calcules métricas de seleção a partir do modelo final treinado com
  todos os dados; a seleção usa apenas as métricas fora do treino dos
  folds.
- Guarda os pipelines de cada fold em `artefactos/modelos/folds/{modelo}/`.
- Usa `pathlib`, seed 42, cria pastas antes de escrever, type hints,
  docstrings, comentários em português europeu sem gerúndios.
- Termina com `if __name__ == "__main__": main()`.

SAÍDAS
- `artefactos/metricas/metricas_por_atividade.csv`
- `artefactos/metricas/resumo_modelos.csv`, com a melhoria percentual face
  ao `DummyRegressor`.
- `artefactos/previsoes_cv/previsoes_fora_do_treino.csv`
- Modelos por fold em `artefactos/modelos/folds/{modelo}/`.
- `artefactos/modelos/final/modelo_ritmo_final.joblib`
- `artefactos/modelos/final/metadados_modelo.json`

Imprime no fim o modelo vencedor, o MAE macro em `s/km` e `mm:ss/km`, e a
melhoria percentual face ao baseline.
```

### Ficheiro a criar

`treinar_modelos.py`

### Entradas e artefactos necessários

- `Dados/processados/amostras_modelo.csv`.

### Funções obrigatórias e responsabilidades

`criar_modelos`, `calcular_pesos`, `calcular_metricas`, `avaliar_logo`, `selecionar_vencedor`, `treinar_final`, `guardar_pacote_modelo`, `main`.

### Bibliotecas, regras técnicas e validações

`scikit-learn` (`Ridge`, `RandomForestRegressor`, `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `SVR`, `MLPRegressor`, `DummyRegressor`, `LeaveOneGroupOut`, `Pipeline`, `StandardScaler`), `joblib`, `pandas`, `numpy`; mesmos folds, linhas e pesos para todos os modelos; `DummyRegressor` nunca vence.

### Saídas e critérios de aceitação verificáveis

- Todas as saídas listadas existem.
- O `DummyRegressor` aparece nas métricas mas nunca é o vencedor guardado.
- Nenhuma atividade aparece simultaneamente no treino e no teste do mesmo fold.
- O pacote final contém a ordem das features, zonas, parâmetros do perfil e métricas fora do treino.

### Após receber o código

1. Guarda como `treinar_modelos.py` na raiz do projeto.
2. Confirma que `amostras_modelo.csv` tem pelo menos duas atividades distintas.
3. Executa `python treinar_modelos.py` e acompanha os prints por fold.
4. Confirma o vencedor impresso e verifica `resumo_modelos.csv`.
5. Confirma que `modelo_ritmo_final.joblib` e `metadados_modelo.json` foram criados.

---

## 📈 PROMPT 5 — Visualização e diagnóstico dos resultados

### O que vais aprender

- Construir visualizações de comparação de modelos e de diagnóstico de resíduos.
- Calcular importância por permutação apenas nos folds de teste.
- Construir uma curva de aprendizagem por atividades completas, de forma determinística.
- Separar conclusões descritivas de conclusões causais.

### Prompt pronto a copiar

```text
Cria o ficheiro Python `visualizar_resultados.py`, que gera os gráficos e
diagnósticos do modelo vencedor.

CONTEXTO
Lê exclusivamente `artefactos/previsoes_cv/previsoes_fora_do_treino.csv`,
`artefactos/metricas/metricas_por_atividade.csv`,
`artefactos/metricas/resumo_modelos.csv` e o pacote em
`artefactos/modelos/final/`. Não relê os dados brutos nem retreina modelos,
exceto para calcular importância por permutação e a curva de aprendizagem,
que podem exigir novos ajustes por fold sobre os dados já preparados.

FUNÇÕES OBRIGATÓRIAS
- `grafico_comparacao_modelos(...) -> None`: MAE e RMSE macro por modelo.
- `grafico_por_atividade(...) -> None`: MAE por atividade e modelo.
- `grafico_tempos(...) -> None`: tempos de treino e previsão por modelo.
- `grafico_previsto_real(...) -> None`: previsões fora do treino do
  vencedor versus valores reais, com linha de identidade.
- `grafico_residuos(...) -> None`: distribuição dos resíduos e resíduos
  versus previsto.
- `avaliar_por_zona_declive(...) -> pandas.DataFrame`: métricas do vencedor
  por zona e por intervalos de declive, com gráfico correspondente.
- `calcular_importancia_permutacao(...) -> pandas.DataFrame`: importância
  por permutação do vencedor, calculada apenas nos folds de teste, agregada
  por feature.
- `curva_aprendizagem_por_atividades(...) -> pandas.DataFrame`: desempenho
  do vencedor com números crescentes de atividades completas de treino,
  acrescentando sempre atividades inteiras, nunca linhas aleatórias, com
  ordens determinísticas por seed 42 e mostrando média e dispersão entre
  repetições válidas.
- `main() -> None`: orquestra tudo e grava os artefactos.

REGRAS TÉCNICAS
- Todos os gráficos com anotações legíveis, valores e unidades corretas
  (`s/km` e `mm:ss/km` onde fizer sentido).
- Escreve conclusões automáticas apenas descritivas em
  `artefactos/metricas/diagnostico_modelo.md`, sem afirmações causais sobre
  as features.
- Usa `pathlib`, seed 42, cria pastas antes de escrever, guarda gráficos com
  pelo menos 200 dpi, fecha todas as figuras.
- Type hints, docstrings, comentários em português europeu sem gerúndios.
- Termina com `if __name__ == "__main__": main()`.

SAÍDAS
- `artefactos/graficos/comparacao_modelos.png`
- `artefactos/graficos/metricas_por_atividade.png`
- `artefactos/graficos/tempos_modelos.png`
- `artefactos/graficos/previsto_vs_real.png`
- `artefactos/graficos/residuos_modelo.png`
- `artefactos/graficos/erro_por_zona_declive.png`
- `artefactos/graficos/importancia_permutacao.png`
- `artefactos/graficos/curva_aprendizagem_atividades.png`
- `artefactos/metricas/metricas_por_zona_declive.csv`
- `artefactos/metricas/importancia_permutacao.csv`
- `artefactos/metricas/curva_aprendizagem.csv`
- `artefactos/metricas/diagnostico_modelo.md`

Mostra no fim as três features mais importantes por permutação e o número
de atividades usado na curva de aprendizagem.
```

### Ficheiro a criar

`visualizar_resultados.py`

### Entradas e artefactos necessários

- `artefactos/previsoes_cv/previsoes_fora_do_treino.csv`, `artefactos/metricas/metricas_por_atividade.csv`, `artefactos/metricas/resumo_modelos.csv`, `artefactos/modelos/final/`.

### Funções obrigatórias e responsabilidades

`grafico_comparacao_modelos`, `grafico_por_atividade`, `grafico_tempos`, `grafico_previsto_real`, `grafico_residuos`, `avaliar_por_zona_declive`, `calcular_importancia_permutacao`, `curva_aprendizagem_por_atividades`, `main`.

### Bibliotecas, regras técnicas e validações

`matplotlib`, `pandas`, `numpy`, `scikit-learn` (`permutation_importance`); importância por permutação restrita aos folds de teste; curva de aprendizagem por atividades completas, com seed 42.

### Saídas e critérios de aceitação verificáveis

- Os oito gráficos desta etapa existem em `artefactos/graficos/`.
- As três tabelas auxiliares existem em `artefactos/metricas/`.
- `diagnostico_modelo.md` não contém afirmações causais sobre as features.

### Após receber o código

1. Guarda como `visualizar_resultados.py` na raiz do projeto.
2. Confirma que `treinar_modelos.py` já foi executado com sucesso.
3. Executa `python visualizar_resultados.py`.
4. Verifica os oito gráficos e o conteúdo de `diagnostico_modelo.md`.

---

## 🗺️ PROMPT 6 — Previsão do ritmo num percurso

### O que vais aprender

- Reconstruir features de um percurso futuro sem qualquer dado medido durante a corrida.
- Aplicar o modelo final a um percurso novo, por zona ou para todas as zonas.
- Sinalizar previsões com pouco suporte nos dados de treino.
- Comunicar de forma explícita as limitações da previsão.

### Prompt pronto a copiar

```text
Cria o ficheiro Python `prever_percurso.py`, que aplica o modelo final a um
percurso novo.

CONTEXTO
Importa `ler_fit` e `normalizar_registos` de `analisar_fits.py` e
`perfil_percurso` de `preparar_dados.py`. Lê o pacote em
`artefactos/modelos/final/modelo_ritmo_final.joblib` e os metadados em
`metadados_modelo.json`, que incluem a ordem das features e a configuração
de zonas usada no treino.

ARGUMENTOS
- `--modelo`: caminho para o pacote Joblib final.
- `--percurso`: caminho para um `.fit` ou `.csv` com `distancia_m` e
  `altitude_m`.
- `--zona`: zona única de 1 a 5.
- `--todas-zonas`: gera previsão para as cinco zonas em vez de uma só.
- `--saida`: pasta de destino dos resultados.

FUNÇÕES OBRIGATÓRIAS
- Reaproveita `perfil_percurso` para construir o perfil do percurso novo.
- Uma função de reconstrução de features que usa exclusivamente a
  distância e a altitude do ficheiro de percurso, mesmo que ele contenha
  FC, ritmo ou velocidade; esses campos, se existirem, são ignorados na
  previsão.
- Uma função de previsão por troço, para uma zona escolhida ou para as
  cinco zonas quando `--todas-zonas` é usado.
- Uma função de deteção de pouco suporte, baseada na proximidade de
  declive, altitude, distância e zona relativamente aos dados de treino, e
  na presença de pelo menos cinco exemplos vindos de pelo menos duas
  atividades diferentes.
- Uma função de validação de saída, que rejeita ritmos não finitos,
  negativos ou muito fora da gama observada em treino, sem os corrigir
  silenciosamente.
- `main() -> None`: orquestra tudo, valida os argumentos e grava os
  resultados.

REGRAS TÉCNICAS
- Nunca usa FC, ritmo ou velocidade do ficheiro de percurso, mesmo que
  estejam presentes.
- As nove features são reconstruídas na mesma ordem usada em treino, lida
  dos metadados do pacote.
- Usa `pathlib`, cria pastas antes de escrever, type hints, docstrings,
  comentários em português europeu sem gerúndios.
- Termina com `if __name__ == "__main__": main()`.

SAÍDAS
- CSV por troço com o ritmo previsto, em
  `artefactos/previsoes_percurso/`.
- JSON com distância total, tempo estimado sem paragens, ritmo médio
  ponderado e a distância marcada como tendo pouco suporte.
- `artefactos/graficos/previsao_percurso.png`, com altitude, declive e
  ritmo previsto ao longo do percurso.
- Um aviso explícito, impresso e escrito no JSON, de que a previsão assume
  permanência constante na zona escolhida e não simula a resposta cardíaca
  real durante o esforço.
```

### Ficheiro a criar

`prever_percurso.py`

### Entradas e artefactos necessários

- `artefactos/modelos/final/modelo_ritmo_final.joblib`, `metadados_modelo.json`, um `.fit` ou `.csv` de percurso novo.

### Funções obrigatórias e responsabilidades

Reconstrução de perfil e features, previsão por troço/zona, deteção de pouco suporte, validação de saída, `main`.

### Bibliotecas, regras técnicas e validações

`joblib`, `pandas`, `numpy`, `matplotlib`, `argparse`; nenhum uso de FC, ritmo ou velocidade observados no percurso futuro.

### Saídas e critérios de aceitação verificáveis

- CSV por troço e JSON de resumo existem em `artefactos/previsoes_percurso/`.
- `previsao_percurso.png` existe com as três séries pedidas.
- O aviso sobre a limitação da previsão aparece na consola e no JSON.

### Após receber o código

1. Guarda como `prever_percurso.py` na raiz do projeto.
2. Executa com um percurso de teste e uma única zona, depois repete com `--todas-zonas`.
3. Confirma que o CSV, o JSON e o gráfico foram criados.
4. Verifica que o aviso de limitação aparece no JSON.

---

## 📄 PROMPT 7 — Relatório automático

### O que vais aprender

- Compilar um relatório final apenas a partir de artefactos já produzidos.
- Estruturar um relatório técnico com secções claras e verificáveis.
- Discutir limitações metodológicas de forma explícita e honesta.
- Referenciar visualizações e tabelas por caminho relativo.

### Prompt pronto a copiar

```text
Cria o ficheiro Python `gerar_relatorio.py`, que produz o relatório final
do projeto a partir dos artefactos reais já gerados.

CONTEXTO
Lê apenas artefactos já existentes em `Dados/processados/`,
`configuracao/zonas_fc.json`, `artefactos/analise_inicial/`,
`artefactos/analise_dataset/`, `artefactos/metricas/`,
`artefactos/previsoes_cv/`, `artefactos/previsoes_percurso/` (se existir) e
`artefactos/graficos/`. Nunca escreve métricas, tabelas ou conclusões que
não estejam sustentadas por esses artefactos.

FUNÇÕES OBRIGATÓRIAS
Uma função por secção do relatório, mais `main()`:
- objetivo
- dados
- análise exploratória inicial
- qualidade
- perfil e alvo
- zonas
- dataset final
- validação por atividades
- modelos comparados
- resultados
- diagnóstico do vencedor
- previsão de percurso (se o artefacto existir; caso contrário, indica que
  a etapa não foi executada)
- limitações
- conclusões

Cada função lê os ficheiros de que precisa e devolve o texto Markdown da
sua secção.

REGRAS TÉCNICAS
- Inclui tabelas com métricas por atividade, médias macro, tempos de
  treino e previsão, e a melhoria percentual face ao `DummyRegressor`.
- Incorpora todas as visualizações relevantes através de caminhos
  relativos a partir da raiz do projeto.
- Na secção de limitações, discute de forma explícita: outliers
  identificados e não removidos, dimensão da amostra, cobertura das zonas,
  risco de fuga de dados e como foi mitigado, generalização a percursos
  não vistos e os limites da fórmula `220 - idade` quando usada.
- Usa `pathlib`, type hints, docstrings, comentários em português europeu
  sem gerúndios.
- Termina com `if __name__ == "__main__": main()`.

SAÍDAS
- `RELATORIO_FINAL.md` na raiz do projeto.
- Cópia idêntica em `artefactos/relatorios/`.

Imprime no fim a lista de secções escritas e qualquer secção que não pôde
ser preenchida por falta de artefactos, sem inventar o conteúdo em falta.
```

### Ficheiro a criar

`gerar_relatorio.py`

### Entradas e artefactos necessários

- Todos os artefactos produzidos pelos prompts 1 a 6.

### Funções obrigatórias e responsabilidades

Uma função por secção listada acima, mais `main`.

### Bibliotecas, regras técnicas e validações

`pandas`, `json`, `pathlib`; leitura exclusiva de artefactos já existentes; nenhuma métrica ou conclusão inventada.

### Saídas e critérios de aceitação verificáveis

- `RELATORIO_FINAL.md` existe na raiz e em `artefactos/relatorios/`, com as catorze secções pedidas.
- Todas as tabelas do relatório correspondem a valores presentes nos CSV lidos.
- A secção de previsão de percurso reflete corretamente se essa etapa foi ou não executada.

### Após receber o código

1. Guarda como `gerar_relatorio.py` na raiz do projeto.
2. Confirma que os prompts 1 a 5 (e opcionalmente 6) já foram executados.
3. Executa `python gerar_relatorio.py`.
4. Abre `RELATORIO_FINAL.md` e confirma que os caminhos das imagens abrem corretamente.

---

## 🧩 PROMPT 8 — Orquestrador

### O que vais aprender

- Encadear scripts Python independentes através de `subprocess`.
- Validar pré-condições de cada etapa antes de a executar.
- Construir um pipeline retomável, com registo de duração e erros por etapa.
- Separar deteção de dependências de instalação automática.

### Prompt pronto a copiar

```text
Cria o ficheiro Python `lab_orquestrador.py`, que executa o pipeline
completo do projeto.

CONTEXTO
Este script não reimplementa nenhuma lógica dos scripts anteriores; apenas
os invoca pela ordem correta através de `subprocess`, usando
`sys.executable`.

ORDEM DE EXECUÇÃO
`analisar_fits.py`, `preparar_dados.py`, `analisar_dataset_modelo.py`,
`treinar_modelos.py`, `visualizar_resultados.py`, `gerar_relatorio.py`.
`prever_percurso.py` só é executado quando forem fornecidos argumentos de
percurso e zona.

ARGUMENTOS (`argparse`)
- seleção de etapas a executar (todas por defeito);
- retomar a execução a partir de uma etapa específica;
- `--idade` e `--limites-fc`, repassados a `preparar_dados.py`;
- `--reprocessar`, para forçar a reconstrução dos datasets processados
  quando a origem ou a configuração de zonas mudaram, sem nunca alterar os
  `.fit` crus;
- argumentos de percurso e zona, repassados a `prever_percurso.py` quando
  presentes;
- opção para parar ou continuar após uma etapa falhar.

FUNÇÕES OBRIGATÓRIAS
- Uma função de validação de pré-condições antes de cada etapa: existência
  de `Dados/brutos/`, estado de `Dados/processados/`, existência e validade
  de `configuracao/zonas_fc.json` quando necessário, existência do
  manifesto, existência dos artefactos da etapa anterior e verificação de
  dependências Python instaladas.
- Uma função que gera `requirements.txt` como ficheiro de texto simples,
  sem instalar nada automaticamente; se faltar uma dependência, mostra o
  comando exato de instalação e para essa etapa.
- Uma função de execução de cada etapa via `subprocess`, com captura de
  stdout e stderr e medição da duração.
- `main() -> None`: interpreta os argumentos, valida pré-condições, executa
  as etapas pela ordem correta e escreve o resumo final.

REGRAS TÉCNICAS
- Regista tudo em `artefactos/logs/execucao.log`, em UTF-8, com duração por
  etapa e um resumo final de sucesso ou falha.
- Usa códigos de saída corretos: sucesso, falha numa etapa, pré-condição em
  falta.
- Não apresenta menus interativos obrigatórios; tudo é controlado por
  argumentos de linha de comandos.
- Usa `pathlib`, type hints, docstrings, comentários em português europeu
  sem gerúndios.
- Termina com `if __name__ == "__main__": main()`.

SAÍDAS
- `artefactos/logs/execucao.log`
- `requirements.txt` na raiz do projeto
- Resumo final impresso na consola com o estado de cada etapa e a duração
  total.
```

### Ficheiro a criar

`lab_orquestrador.py`

### Entradas e artefactos necessários

- Todos os scripts anteriores na raiz do projeto; `Dados/brutos/`; opcionalmente `configuracao/zonas_fc.json`.

### Funções obrigatórias e responsabilidades

Validação de pré-condições, geração de `requirements.txt`, execução por `subprocess` com captura de stdout/stderr e duração, `main`.

### Bibliotecas, regras técnicas e validações

`subprocess`, `sys`, `argparse`, `pathlib`, `logging`; nenhuma instalação automática de dependências; nenhuma alteração aos `.fit` crus mesmo com `--reprocessar`.

### Saídas e critérios de aceitação verificáveis

- `execucao.log` regista todas as etapas executadas, com duração e estado.
- `requirements.txt` existe e lista as dependências detetadas.
- O pipeline completo corre do início ao fim sem intervenção manual quando todas as pré-condições estão satisfeitas.
- Uma dependência em falta interrompe a etapa correspondente com o comando de instalação exato, sem instalar nada sozinho.

### Após receber o código

1. Guarda como `lab_orquestrador.py` na raiz do projeto, junto dos restantes sete scripts.
2. Executa `python lab_orquestrador.py` numa pasta com `Dados/brutos/` já preenchida.
3. Acompanha `artefactos/logs/execucao.log` durante a execução.
4. Confirma que `RELATORIO_FINAL.md` é o último artefacto criado com sucesso.
5. Testa `--reprocessar` depois de alterar `configuracao/zonas_fc.json` e confirma que os datasets processados são regenerados sem tocar em `Dados/brutos/`.
