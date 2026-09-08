# Aplicação RunningAI

## Iniciar

Na raiz do projeto:

```bash
.venv/bin/python app.py
```

Abrir no navegador: <http://127.0.0.1:5000>

Para usar outra porta:

```bash
APP_PORT=8000 .venv/bin/python app.py
```

## Serviços

A página inicial apresenta os serviços da plataforma. O **RitmoAI** está disponível; **PosturaAI** e **TreinadorAI** aparecem como serviços futuros.

## RitmoAI

- Escolher uma atividade existente em `RitmoAI/Dados/brutos/` ou enviar um ficheiro `.fit`/`.csv`.
- Um CSV deve conter as colunas `distancia_m` e `altitude_m`.
- Selecionar uma zona cardíaca entre Z1 e Z5.
- Consultar o ritmo médio, tempo estimado, distância, subida acumulada, gráfico e previsão por troço.
- Os uploads são processados num ficheiro temporário e eliminados após a previsão.

O modelo e o dataset são recarregados automaticamente quando os respetivos ficheiros são atualizados durante uma nova sessão de treino.
