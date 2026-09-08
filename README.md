# RunningAI

RunningAI é uma aplicação Flask que reúne serviços de inteligência artificial aplicados à corrida. Cada serviço corresponde ao projeto final de um módulo do curso e vive numa pasta própria, com o seu código, dados, artefactos e documentação.

## Serviços

| Serviço | Estado | Objetivo |
|---|---|---|
| [RitmoAI](RitmoAI/README.md) | Disponível | Prever o ritmo num percurso para uma zona cardíaca escolhida. |
| [PosturaAI](PosturaAI/README.md) | Planeado | Analisar a postura e a técnica de corrida. |
| [TreinadorAI](TreinadorAI/README.md) | Planeado | Apoiar a criação e adaptação de planos de treino. |

## Estrutura do repositório

```text
Projeto Prático/
├── app.py                  # aplicação web e integração dos serviços
├── templates/              # páginas e componentes Flask
├── static/                 # estilos, scripts e ícones da interface
├── requirements.txt        # dependências da aplicação completa
├── INSTRUCOES_APP.md       # utilização rápida da aplicação web
├── RitmoAI/                # projeto de previsão de ritmo
├── PosturaAI/              # futuro projeto de análise de postura
└── TreinadorAI/            # futuro projeto de apoio ao treino
```

A raiz contém apenas a aplicação web e os ficheiros globais necessários para instalar e documentar o repositório. Os detalhes de implementação de cada projeto ficam na respetiva pasta.

## Instalação e execução

Requer Python 3.11 ou superior. A partir da raiz do repositório:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Abra <http://127.0.0.1:5000>. O catálogo apresenta todos os serviços; o RitmoAI está acessível em <http://127.0.0.1:5000/ritmo-ai>.

Os dados pessoais e os modelos treinados não são versionados. Consulte o [README do RitmoAI](RitmoAI/README.md) para preparar os dados e treinar o primeiro serviço.
