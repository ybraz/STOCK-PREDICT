# 📈 LSTM Stock Price Prediction

Projeto para previsão de preços de ações usando LSTM (Long Short-Term Memory) em Python, com indicadores técnicos e avaliação robusta.

Funcionalidades:
- Scripts para treino e avaliação de modelos.
- API para inferência via FastAPI.
- Métricas Prometheus para monitoramento.
- Dockerfile para deploy simples.

---

## 📦 Estrutura do Projeto

lstm-stock-prediction/
    api/
        main.py
        schemas.py
    data/
        raw/              # CSVs baixados automaticamente
    models/               # Modelos e scalers salvos
    src/
        train.py
        evaluate.py
        utils/
            data_utils.py
    .gitignore
    .env.example
    Dockerfile
    Makefile
    requirements.txt
    README.md

---

## 🚀 Como usar

### 1. Preparar ambiente

    make setup

Ou manualmente:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

---

### 2. Treinar o modelo

Treino normal:

    make train SYMBOL=DIS

Treino robusto (recomendado):

    make strong-train SYMBOL=DIS

No treino forte são usados:
- Histórico desde 2005/2010
- Sequência de 180 dias
- 200 épocas
- Busca por hiperparâmetros e arquitetura LSTM mais profunda

Arquivos salvos:
- models/lstm_<SYMBOL>.keras
- models/scalers_<SYMBOL>.pkl

---

### 3. Avaliar o modelo

    make evaluate SYMBOL=DIS

Exibe as métricas:
- MAE (Erro Médio Absoluto)
- RMSE (Erro Quadrático Médio)
- MAPE (Erro Percentual Médio)
- SMAPE (Erro Percentual Médio Simétrico)

---

### 4. Rodar a API localmente

    make api

- A API estará disponível em: http://localhost:8000
- Documentação Swagger: http://localhost:8000/docs
- Métricas Prometheus: http://localhost:8000/metrics

---

### 5. Fazer uma previsão

    make predict SYMBOL=DIS

Ou manualmente via curl:

    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"symbol": "DIS", "seq_length": 180}'

---

### 6. Monitorar com TensorBoard

    make tensorboard

Acesse:
- http://localhost:6006

---

### 7. Deploy com Docker

Buildar a imagem:

    make docker-build

Rodar o container:

    make docker-run

---