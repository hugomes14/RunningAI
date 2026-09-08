# Meta Prompt Completo — Projeto Prático 2: previsão de ritmo por percurso e zona cardíaca

## QUEM ÉS

És especialista em Machine Learning aplicado à corrida, análise de ficheiros FIT, avaliação por grupos, visualização de dados e Prompt Engineering. A tua tarefa é produzir um **GUIÃO DE PROMPTS completo, coerente e executável** para desenvolver o modelo do Projeto Prático 2.

Escreve em português europeu, sem gerúndios, num tom pedagógico e direto. O público sabe executar scripts Python, mas precisa de instruções claras, nomes de ficheiros, funções, entradas, saídas e verificações.

Produz um guião para gerar código através de outro LLM. Não implementes o projeto nesta resposta, não apresentes resultados simulados e não inventes dados pessoais.

## ENTRADA — ENUNCIADO DO PROJETO

O projeto pretende prever o **ritmo pessoal de corrida** para um percurso conhecido, com base em:

- distância e posição ao longo do percurso;
- altitude e perfil altimétrico;
- declive de cada troço;
- desnível positivo acumulado;
- zona de treino cardíaca escolhida, de Z1 a Z5.

Os dados históricos crus encontram-se em ficheiros Garmin `Dados/brutos/*.fit`. Cada atividade pode conter timestamp, distância acumulada, altitude, frequência cardíaca e velocidade. A potência não faz parte deste projeto.

A pasta `Dados/` separa de forma explícita a origem e os resultados do tratamento:

- `Dados/brutos/`: exportações diretas do Garmin Connect, imutáveis;
- `Dados/processados/`: datasets derivados, limpos ou agregados, que podem ser recriados a partir dos dados crus.

O modelo não recebe a frequência cardíaca exata. A FC observada serve apenas para classificar cada intervalo histórico numa zona. As features do modelo representam a zona através da sua ordem e dos respetivos limites.

O projeto deve experimentar vários algoritmos de regressão e selecionar um único modelo final com base em validação por atividades completas. Os candidatos são:

1. `Ridge`, como referência linear;
2. `RandomForestRegressor`;
3. `GradientBoostingRegressor`;
4. `HistGradientBoostingRegressor`;
5. `SVR` com kernel RBF;
6. `DummyRegressor`, apenas como baseline constante.

O produto final inclui análise exploratória inicial, preparação dos dados, análise do dataset de modelação, treino e comparação dos modelos, visualização dos resultados, seleção do vencedor, previsão de ritmo num percurso e relatório final.

Aplicação web, interface gráfica e API ficam fora do âmbito.

## OBJETIVO DA TUA RESPOSTA

Produz um único documento intitulado:

`Guião de Prompts para Projeto Prático 2 — Previsão de Ritmo por Percurso e Zona Cardíaca`

O documento deve conter exatamente oito prompts encadeados, prontos a copiar para outro LLM. Cada prompt cria um ficheiro Python concreto e usa apenas entradas ou artefactos já disponíveis.

## CONTRATO DO PROBLEMA

### Separação entre dados crus e processados

- Lê ficheiros de atividade apenas de `Dados/brutos/`.
- Trata todos os `.fit` dessa pasta como dados crus exportados diretamente do Garmin Connect. Nunca os altera, renomeia, substitui ou regrava.
- Ignora ficheiros auxiliares com o sufixo `:Zone.Identifier`; são metadados de proveniência do Windows e não dados de atividade.
- Guarda em `Dados/processados/` qualquer CSV ou JSON que resulte de descodificação, normalização, limpeza, cálculo de intervalos, construção do perfil ou agregação para o modelo.
- Não mistura outputs de análise, gráficos, métricas ou modelos com os datasets: esses elementos permanecem em `artefactos/`.
- Cada script lê ficheiros processados por nome exato. Não deve usar um glob genérico que possa misturar datasets de fases diferentes.
- Cria `Dados/processados/manifesto_processamento.json` com os caminhos e hashes SHA-256 dos FIT de origem, data de geração, parâmetros de processamento, configuração de zonas e nomes dos datasets produzidos.
- Considera os dados processados derivados e substituíveis. Se os hashes dos FIT, a configuração das zonas ou os parâmetros do perfil mudarem, marca-os como desatualizados e volta a gerá-los antes do treino.
- A análise exploratória inicial parte sempre dos FIT crus. A análise do dataset de modelação parte apenas de `Dados/processados/amostras_modelo.csv`.

### Variável-alvo e unidades

- O alvo é `ritmo_alvo_s_km`, expresso em segundos por quilómetro.
- Calcula o alvo em cada amostra por `ritmo_alvo_s_km = 1000 × soma(dt_valido) / soma(dx_valido)`.
- Não usa a média simples do ritmo ou da velocidade dos registos.
- Não usa o campo de velocidade do FIT para criar o alvo; esse campo serve apenas para auditoria exploratória.
- Apresenta o ritmo ao utilizador também no formato `mm:ss/km`.
- Calcula MAE e RMSE em `s/km` e mostra uma coluna adicional de MAE no formato `mm:ss/km`.

### Features permitidas

O modelo pode usar apenas variáveis conhecidas antes da corrida:

- `zona_ordem`, com valores inteiros de 1 a 5;
- `zona_limite_inferior_bpm`;
- `zona_limite_superior_bpm`;
- `zona_aberta`, caso uma zona não tenha limite superior;
- `declive_pct`;
- `altitude_normalizada`, calculada por `altitude_media_m / 1000`;
- `distancia_inicio_km`;
- `subida_acumulada_m`;
- `comprimento_troco_m`.

A frequência cardíaca exata, a velocidade observada, o ritmo observado, o tempo real futuro, a potência e qualquer valor medido depois do início do troço não podem entrar nas features.

### Leitura dos FIT

- Usa `garmin-fit-sdk` através de `Decoder` e `Stream`.
- Valida `check_integrity()` e os erros devolvidos por `read()`.
- Processa todos os ficheiros `Dados/brutos/*.fit` sem fixar nomes no código.
- Aceita atividades de corrida e variantes de corrida reconhecidas pelo FIT. Regista e ignora outros desportos com justificação.
- Se um ficheiro tiver mais de uma sessão, separa as sessões quando for possível; caso contrário, exclui o ficheiro com diagnóstico claro.
- Normaliza os campos para: `atividade_id`, `timestamp`, `distancia_m`, `altitude_m`, `fc_bpm` e `velocidade_fit_m_s`.
- Prefere campos `enhanced_*` quando existirem e documenta os fallbacks para os campos normais.
- Preserva os dados crus e regista todas as exclusões.

### Perfil do percurso

- Constrói o perfil apenas com distância e altitude, pois estas variáveis são conhecidas antes da corrida.
- Ordena por distância, trata distâncias repetidas de forma explícita e interpola a altitude numa grelha regular de 10 m.
- Suaviza a altitude com média móvel centrada de cinco pontos.
- A suavização centrada não constitui fuga de dados porque todo o perfil do percurso é conhecido antes da partida. Explica esta decisão.
- Divide o percurso em troços de 50 m. O último troço pode ser mais curto e deve guardar o seu comprimento real.
- Calcula `declive_pct = 100 × (altitude_fim - altitude_inicio) / comprimento_troco_m`.
- Calcula `altitude_media_m`, `altitude_normalizada = altitude_media_m / 1000`, `distancia_inicio_km` e `subida_acumulada_m` apenas a partir do perfil. Usa a altitude normalizada como feature e preserva a altitude absoluta para auditoria e gráficos. A escala fixa mantém a diferença de altitude entre percursos e não usa estatísticas de atividades de validação.

### Intervalos, limpeza e agregação

- Ordena os registos por timestamp e calcula intervalos entre registos consecutivos: `dt`, `dx` e ponto médio da distância.
- Associa cada intervalo ao troço através do ponto médio da distância.
- Considera um intervalo válido se: `0 < dt <= 15 s`, `dx > 0`, `dx <= 10 × dt`, `fc_bpm > 0` e `dx/dt >= 0.5 m/s`.
- Regista como motivos separados: timestamp inválido ou repetido, pausa, recuo de distância, salto temporal, salto de distância, FC ausente ou não positiva e velocidade inferior ao limiar.
- Classifica cada intervalo válido na zona cardíaca **antes** da agregação.
- Agrega por `atividade_id`, `troco_id` e `zona_ordem`. Uma transição pode criar dois grupos no mesmo troço, mas nunca cria uma zona intermédia artificial.
- Mantém apenas amostras com pelo menos dois intervalos válidos e cobertura entre 60% e 120% do comprimento do troço.
- Calcula o alvo através das somas de distância e tempo do grupo.
- Não remove outliers automaticamente. Identifica-os, visualiza-os e documenta o impacto.

### Estabilidade cardíaca causal

- Antes da agregação, aplica um filtro de estabilidade da FC a cada atividade.
- Em cada instante, usa apenas os 40 segundos anteriores, incluindo o instante atual.
- Exige pelo menos quatro medições que cubram 30 segundos da janela, amplitude robusta entre os percentis 10 e 90 até 15 bpm, diferença absoluta entre a FC atual e a média temporal da janela até 8 bpm e pelo menos 120 segundos desde o início da atividade.
- Reinicia a janela após uma sequência inválida ou uma falha temporal superior a 15 segundos, aceitando tanto gravação a cada segundo como gravação inteligente do Garmin.
- Guarda a máscara e o motivo de rejeição. Nunca consulta FC futura.
- Documenta a análise de sensibilidade que compara janelas de 60, 40 e 30 segundos.

### Configuração das zonas

- Lê `configuracao/zonas_fc.json`.
- O JSON guarda `idade`, `fc_max_estimada`, cinco zonas com limites inferior e superior, origem dos limites, data e notas.
- Se existirem limites pessoais fornecidos pelo utilizador, dá-lhes prioridade.
- Se existir idade e faltarem limites, estima `fc_max = 220 - idade` e cria Z1–Z5 entre 50% e 100% de `fc_max`, com seis fronteiras arredondadas. Declara que a fórmula é uma aproximação.
- Não inventa idade nem limites pessoais.
- Se o JSON não existir ou estiver incompleto, cria um modelo com valores `null`, mostra um exemplo válido e termina a preparação com código de saída claro. Não treina modelos com zonas inventadas.
- Valida que os limites são numéricos, crescentes, não sobrepostos e coerentes nas fronteiras.

### Avaliação sem fuga de dados

- Usa `atividade_id` como grupo em `LeaveOneGroupOut`.
- Cada fold testa uma atividade completa e treina nas restantes.
- Nunca usa `train_test_split`, K-Fold aleatório ou divisão de intervalos da mesma atividade entre treino e teste.
- Em cada fold, calcula pesos para que cada atividade de treino tenha o mesmo peso total. Dentro de uma atividade, cada amostra recebe o mesmo peso.
- Usa exatamente os mesmos folds, linhas e pesos para todos os modelos.
- Qualquer transformação ou scaler deve ser ajustado apenas nos dados de treino do fold.
- Seleciona o vencedor pelo menor MAE macro fora do treino. Em caso de empate, usa o menor RMSE macro e depois o menor tempo médio de previsão.
- O `DummyRegressor` não pode vencer; serve apenas para confirmar se os modelos aprendem algo acima da média.
- Guarda todas as previsões fora do treino. Só depois da comparação treina o modelo vencedor final com todas as atividades.
- Se existirem menos de duas atividades elegíveis, não executa a validação e apresenta um diagnóstico sem métricas inventadas.

### Modelos e parâmetros iniciais

Usa parâmetros fixos para a primeira comparação, sem tuning no conjunto de teste:

- `Ridge(alpha=10)` dentro de `Pipeline(StandardScaler(), Ridge())`;
- `RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.8, random_state=42, n_jobs=-1)`;
- `GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, min_samples_leaf=5, loss="huber", random_state=42)`;
- `HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, min_samples_leaf=20, learning_rate=0.05, l2_regularization=5, monotonic_cst=[-1, -1, -1, 0, 1, 0, 0, 0, 0], random_state=42)` com a ordem exata das nove features já definida;
- `SVR(kernel="rbf", C=10, epsilon=10, gamma="scale")` dentro de um pipeline com `StandardScaler`;
- `DummyRegressor(strategy="mean")`.

Para o `HistGradientBoostingRegressor`, a hipótese monotónica é: um aumento da zona ou dos seus limites não deve aumentar o ritmo em segundos por quilómetro; um aumento do declive não deve reduzir o ritmo. As restantes features ficam sem restrição. Confirma no código que o vetor de restrições corresponde à ordem real das colunas.

Passa `sample_weight` aos estimadores que o suportam. Para pipelines, usa o nome correto da etapa. Se uma versão instalada não suportar pesos num estimador, regista a limitação e mantém a comparação, sem ocultar o aviso.

### Métricas e diagnósticos

Calcula e guarda:

- MAE e RMSE em `s/km`, por atividade e modelo;
- MAE e RMSE macro entre atividades;
- R² como diagnóstico secundário, sem o usar isoladamente para escolher o vencedor;
- erro médio assinado, para detetar subestimação ou sobrestimação do ritmo;
- tempo de treino e tempo de previsão;
- métricas por zona e por intervalos de declive;
- melhoria percentual relativamente ao `DummyRegressor`.

### Visualizações obrigatórias

Todos os resultados importantes devem ter uma representação visual. Guarda pelo menos:

1. `perfil_atividades.png`: altitude, ritmo observado e FC ao longo da distância para cada atividade;
2. `qualidade_dados.png`: ausências e exclusões por atividade;
3. `distribuicoes_dataset.png`: ritmo, declive, altitude e distância;
4. `ritmo_por_zona.png`: boxplot do ritmo por zona;
5. `ritmo_declive_zona.png`: relação entre declive e ritmo, com cor por zona;
6. `comparacao_modelos.png`: MAE e RMSE macro por modelo;
7. `metricas_por_atividade.png`: MAE por atividade e modelo;
8. `tempos_modelos.png`: tempos de treino e previsão;
9. `previsto_vs_real.png`: valores fora do treino do vencedor, com linha de identidade;
10. `residuos_modelo.png`: distribuição dos resíduos e resíduos versus previsto;
11. `erro_por_zona_declive.png`: desempenho do vencedor por zona e declive;
12. `importancia_permutacao.png`: importância por permutação do vencedor, calculada apenas nos folds de teste;
13. `curva_aprendizagem_atividades.png`: desempenho do vencedor com números crescentes de atividades completas de treino;
14. `previsao_percurso.png`: altitude, declive e ritmo previsto ao longo do percurso.

A curva de aprendizagem deve acrescentar atividades inteiras, nunca linhas aleatórias. Usa ordens determinísticas com seed 42 e mostra média e dispersão entre repetições válidas.

### Reprodutibilidade

- Usa `pathlib` e caminhos relativos à raiz do projeto.
- Usa sementes fixas sempre que exista aleatoriedade.
- Cria as pastas antes de guardar ficheiros.
- Guarda tabelas em CSV, modelos em Joblib, resumos em JSON e gráficos em PNG com pelo menos 200 dpi.
- Fecha todas as figuras.
- Usa comentários úteis, type hints, docstrings, nomes claros e prints informativos.
- Cada script define `main()` e termina com `if __name__ == "__main__": main()`.
- Não cria classes, notebooks ou módulos Python além dos oito ficheiros indicados.
- Os nomes dos módulos não começam por algarismos, para permitir importações entre etapas.
- Não instala bibliotecas a partir dos scripts. O orquestrador deteta dependências ausentes e mostra o comando necessário.

## ESTRUTURA DE PASTAS E ARTEFACTOS

O guião deve impor esta estrutura:

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

Artefactos mínimos:

- `Dados/processados/registos_normalizados.csv`;
- `artefactos/analise_inicial/resumo_atividades.csv`;
- gráficos da análise inicial;
- `Dados/processados/intervalos_auditados.csv`;
- `Dados/processados/trocos_percurso.csv`;
- `Dados/processados/amostras_modelo.csv`;
- `Dados/processados/qualidade_dados.csv`;
- `Dados/processados/manifesto_processamento.json`;
- `artefactos/metricas/metricas_por_atividade.csv`;
- `artefactos/metricas/resumo_modelos.csv`;
- `artefactos/metricas/metricas_por_zona_declive.csv`;
- `artefactos/metricas/importancia_permutacao.csv`;
- `artefactos/metricas/curva_aprendizagem.csv`;
- `artefactos/previsoes_cv/previsoes_fora_do_treino.csv`;
- modelos de cada fold em `artefactos/modelos/folds/{modelo}/`;
- `artefactos/modelos/final/modelo_ritmo_final.joblib`;
- `artefactos/modelos/final/metadados_modelo.json`;
- todos os gráficos obrigatórios em `artefactos/graficos/`;
- `artefactos/metricas/resultado.json`;
- `RELATORIO_FINAL.md`;
- `artefactos/logs/execucao.log`.

## ESTRUTURA OBRIGATÓRIA DO GUIÃO

O documento gerado deve conter, por esta ordem:

1. título;
2. `📚 Introdução ao Prompt Engineering`, com os princípios Sê específico, Dá contexto, Pede exemplos, Itera e Estrutura a tarefa;
3. `Assunções e Inferências`, com os dados disponíveis, valores pessoais em falta, unidades, algoritmos, validação e artefactos;
4. os oito prompts abaixo.

### PROMPT 1 — Análise exploratória inicial dos ficheiros FIT

Ficheiro: `analisar_fits.py`.

Deve pedir:

- funções `ler_fit`, `normalizar_registos`, `validar_atividade`, `resumir_atividade`, `visualizar_atividade`, `visualizar_qualidade` e `main`;
- leitura e normalização de todos os FIT;
- leitura exclusiva de `Dados/brutos/*.fit`, com exclusão dos ficheiros `:Zone.Identifier`;
- validação de integridade, sessões, timestamps, unidades e campos disponíveis;
- análise de valores em falta, duplicados, pausas, recuos, saltos e valores impossíveis;
- estatísticas por atividade: data, distância, duração, tempo em movimento, subida, ritmo, FC, frequência dos registos e cobertura de campos;
- visualizações iniciais de altitude, ritmo e FC ao longo do tempo e da distância;
- histogramas globais de FC e ritmo;
- gravação dos registos normalizados em `Dados/processados/`, do resumo e tabelas exploratórias em `artefactos/analise_inicial/` e dos gráficos em `artefactos/graficos/`;
- conclusões exploratórias calculadas a partir dos dados, sem texto ou números inventados.

### PROMPT 2 — Perfil, zonas cardíacas e preparação dos dados

Ficheiro: `preparar_dados.py`.

Deve pedir:

- importação de `ler_fit` e `normalizar_registos` a partir de `analisar_fits.py`;
- funções `carregar_ou_criar_zonas`, `validar_zonas`, `classificar_zona`, `perfil_percurso`, `criar_intervalos`, `estabilidade_fc`, `agregar_amostras`, `calcular_pesos_atividade` e `main`;
- argumentos opcionais `--idade` e `--limites-fc` para criar uma configuração válida sem editar código;
- implementação literal das regras de perfil, limpeza, estabilidade, classificação e agregação;
- testes das fronteiras das cinco zonas;
- conservação de todas as linhas auditadas, com colunas `valido` e `motivo_exclusao`;
- criação de `trocos_percurso.csv`, `intervalos_auditados.csv`, `amostras_modelo.csv` e `qualidade_dados.csv` dentro de `Dados/processados/`;
- criação ou atualização de `Dados/processados/manifesto_processamento.json`, com proveniência e parâmetros suficientes para reproduzir os datasets;
- confirmação explícita de que `Dados/processados/amostras_modelo.csv` contém as nove features permitidas, o alvo e os identificadores, sem FC exata.

### PROMPT 3 — Análise exploratória do dataset de modelação

Ficheiro: `analisar_dataset_modelo.py`.

Deve pedir:

- funções `validar_dataset`, `analisar_distribuicoes`, `analisar_por_zona`, `analisar_relacoes`, `detetar_outliers`, `criar_resumo_eda` e `main`;
- dimensão, tipos, ausências, duplicados, número de atividades, número de amostras por atividade e zona e estatísticas das features e do alvo;
- distribuições de ritmo, declive, altitude, distância e subida acumulada;
- boxplots de ritmo por zona e atividade;
- relação entre ritmo e declive com cor por zona;
- correlações numéricas e identificação de colinearidade;
- identificação de outliers por IQR, sem remoção automática;
- análise da cobertura do espaço de features por atividade;
- leitura exclusiva de `Dados/processados/amostras_modelo.csv`, depois de validar o respetivo manifesto;
- gravação das tabelas e de `resumo_eda.md` em `artefactos/analise_dataset/`, com os gráficos obrigatórios em `artefactos/graficos/` e conclusões derivadas dos dados.

### PROMPT 4 — Treino, validação e seleção dos modelos

Ficheiro: `treinar_modelos.py`.

Deve pedir:

- funções `criar_modelos`, `calcular_pesos`, `calcular_metricas`, `avaliar_logo`, `selecionar_vencedor`, `treinar_final`, `guardar_pacote_modelo` e `main`;
- carregamento das nove features pela ordem contratada;
- treino dos cinco candidatos e do baseline em folds `LeaveOneGroupOut` idênticos;
- scalers ajustados apenas no treino dentro dos pipelines;
- pesos iguais por atividade;
- previsões, métricas e tempos por fold;
- guarda dos pipelines completos de cada fold;
- seleção automática do vencedor pelos critérios definidos;
- treino final do vencedor com todas as atividades;
- pacote Joblib final com pipeline, ordem das features, zonas, parâmetros do perfil, unidades, métricas OOF e versão das bibliotecas;
- CSV e JSON consolidados, sem resultados escritos manualmente.

### PROMPT 5 — Visualização e diagnóstico dos resultados

Ficheiro: `visualizar_resultados.py`.

Deve pedir:

- funções `grafico_comparacao_modelos`, `grafico_por_atividade`, `grafico_tempos`, `grafico_previsto_real`, `grafico_residuos`, `avaliar_por_zona_declive`, `calcular_importancia_permutacao`, `curva_aprendizagem_por_atividades` e `main`;
- leitura exclusiva das previsões e métricas fora do treino;
- criação de todos os gráficos de avaliação listados no contrato;
- importância por permutação do vencedor em cada atividade de teste, seguida de agregação por feature;
- curva de aprendizagem que acrescenta atividades completas e nunca linhas aleatórias;
- anotações legíveis com valores e unidades;
- tabelas auxiliares em CSV e pequenas conclusões automáticas em `artefactos/metricas/diagnostico_modelo.md`;
- ausência de conclusões causais sobre as features.

### PROMPT 6 — Previsão do ritmo num percurso

Ficheiro: `prever_percurso.py`.

Deve pedir:

- importação de `ler_fit` e `normalizar_registos` de `analisar_fits.py` e de `perfil_percurso` de `preparar_dados.py`;
- argumentos `--modelo`, `--percurso`, `--zona` e `--saida`, com opção `--todas-zonas`;
- aceitação de um FIT ou CSV com `distancia_m` e `altitude_m`;
- uso exclusivo da distância e altitude do FIT futuro, mesmo que ele contenha FC, ritmo ou velocidade;
- reconstrução das nove features com os metadados e zonas guardados no pacote final;
- previsão por troço para uma zona escolhida ou para as cinco zonas;
- validação de ritmos não finitos, negativos ou fora da gama observada, sem correção silenciosa;
- heurística de pouco suporte com base na proximidade de declive, altitude, distância e zona e na presença de pelo menos cinco exemplos de duas atividades;
- CSV por troço, JSON com distância, tempo estimado sem paragens, ritmo médio ponderado e distância com pouco suporte;
- gráfico com altitude, declive e ritmo previsto ao longo do percurso;
- aviso de que a previsão assume permanência na zona escolhida e não simula a resposta cardíaca.

### PROMPT 7 — Relatório automático

Ficheiro: `gerar_relatorio.py`.

Deve pedir:

- funções por secção e `main`;
- leitura dos artefactos reais, sem métricas ou conclusões pré-escritas;
- secções: objetivo, dados, análise exploratória inicial, qualidade, perfil e alvo, zonas, dataset final, validação por atividades, modelos comparados, resultados, diagnóstico do vencedor, previsão de percurso, limitações e conclusões;
- tabelas com métricas por atividade, médias macro, tempos e melhoria face ao baseline;
- incorporação de todas as visualizações relevantes através de caminhos relativos;
- discussão de outliers, dimensão da amostra, cobertura das zonas, fuga de dados, generalização a novos percursos e limites da fórmula de FC máxima;
- criação de `RELATORIO_FINAL.md` e cópia em `artefactos/relatorios/`.

### PROMPT 8 — Orquestrador

Ficheiro: `lab_orquestrador.py`.

Deve pedir:

- execução pela ordem `analisar_fits.py`, `preparar_dados.py`, `analisar_dataset_modelo.py`, `treinar_modelos.py`, `visualizar_resultados.py` e `gerar_relatorio.py`;
- execução opcional de `prever_percurso.py` quando existirem argumentos de percurso e zona;
- `argparse` para escolher etapas, retomar a execução, indicar idade ou limites de FC e decidir se para após um erro;
- validação de `Dados/brutos/`, `Dados/processados/`, configuração, manifesto, artefactos e dependências antes de cada etapa;
- opção `--reprocessar` para reconstruir os datasets processados quando a origem ou a configuração mudar, sem alterar os FIT crus;
- `subprocess` através de `sys.executable`;
- captura de stdout/stderr, duração por etapa, resumo e log UTF-8;
- criação de `requirements.txt` como ficheiro de texto, sem instalação automática;
- códigos de saída corretos e ausência de menus interativos obrigatórios.

## FORMATO OBRIGATÓRIO DE CADA PROMPT

Cada uma das oito secções deve incluir:

1. cabeçalho com emoji e número;
2. `O que vais aprender`, com três a cinco pontos;
3. um único bloco `text` com o prompt pronto a copiar;
4. nome exato do ficheiro a criar;
5. entradas e artefactos necessários;
6. funções obrigatórias e responsabilidades;
7. bibliotecas, regras técnicas e validações;
8. saídas e critérios de aceitação verificáveis;
9. `Após receber o código`, com passos para guardar, executar e confirmar os artefactos.

Cada prompt deve ser autónomo o suficiente para uso isolado, mas não deve repetir o enunciado completo. Evita teoria genérica e redundância.

## CRITÉRIOS DE ACEITAÇÃO DO GUIÃO

Antes de responder, verifica silenciosamente:

- existem exatamente oito prompts;
- o primeiro prompt contém uma análise exploratória inicial real e visual;
- existe apenas uma tarefa de previsão de ritmo por zonas de FC;
- não existem modelos por potência nem modelos com FC exata;
- são comparados cinco algoritmos de regressão e um baseline;
- apenas um vencedor é guardado como modelo final;
- cada artefacto consumido foi criado numa etapa anterior;
- os dados crus e processados ficam em pastas distintas e nenhum script escreve em `Dados/brutos/`;
- o manifesto permite relacionar cada dataset processado com os FIT e parâmetros que lhe deram origem;
- todos os modelos usam os mesmos folds, linhas e pesos;
- nenhuma atividade aparece ao mesmo tempo no treino e no teste de um fold;
- o dataset do modelo não contém FC exata, potência, ritmo futuro ou velocidade futura;
- as métricas, treinos, erros e previsões têm tabelas e visualizações;
- valores pessoais e resultados numéricos não foram inventados;
- não existem app, API, interface gráfica, notebooks ou módulos extra.

## CONTRAEXEMPLOS — NÃO FAZER

- Não cries variantes por FC exata, potência exata ou gamas de potência.
- Não uses potência em nenhuma fase do pipeline.
- Não uses FC exata como feature; usa-a apenas para classificar os intervalos históricos em Z1–Z5.
- Não uses separações aleatórias de registos, intervalos ou troços.
- Não calcules métricas no modelo final treinado com todos os dados e as apresentes como validação.
- Não escolhas o vencedor com métricas do treino.
- Não ajustes parâmetros com a atividade de teste.
- Não calcules o alvo a partir da velocidade do FIT.
- Não elimines dados ou outliers sem auditoria.
- Não inventes idade, zonas, colunas, métricas ou conclusões.
- Não apresentes apenas tabelas quando o contrato exige gráficos.
- Não cries app web, API, GUI, classes, notebooks ou ficheiros Python extra.
- Não uses gerúndios.

## SAÍDA

Produz **apenas** o documento final `Guião de Prompts para Projeto Prático 2 — Previsão de Ritmo por Percurso e Zona Cardíaca`, pronto a copiar. Não acrescentes notas sobre o teu raciocínio, código Python final ou resultados simulados fora do guião.
