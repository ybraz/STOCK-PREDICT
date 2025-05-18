# 📈 LSTM Stock Prediction com Indicadores Técnicos

Pipeline completo para previsão de **retorno diário** (Close / Open − 1) de ações, usando:
- LSTM regularizado (Dropout)
- Indicadores técnicos (SMA, EMA, RSI, MACD)
- Otimização de hiperparâmetros via Keras-Tuner (RandomSearch)
- API FastAPI para previsão online
- Métricas Prometheus para monitoramento
- Pronto para Docker

---

## 📁 Estrutura do Projeto

lstm-stock-prediction/
    api/
        main.py
        schemas.py
    data/
        raw/                # CSVs baixados automaticamente
    models/                 # Modelos treinados e scalers
    src/
        train.py
        evaluate.py
        utils/
            data_utils.py
    Makefile
    requirements.txt
    README.md

---

## 🚀 Como Usar

### 1. Preparar ambiente

Testado apenas com Python 3.11.9

Crie um ambiente virtual e instale as dependências:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

---

### 2. Treinar o modelo

Treine o modelo com hiperparâmetros otimizados via Keras-Tuner (RandomSearch):

    make train-AAPL

Por padrão, são usados:
- Sequência de 180 dias
- Busca por hiperparâmetros (número de unidades LSTM, dropout, learning-rate)
- Indicadores técnicos como features

O modelo treinado e o scaler serão salvos em `models/`.

> **Importante:** Para ações brasileiras (B3), utilize TICKER.SA

    make train-SBSP3.SA

---

### 3. Avaliar o modelo

Para avaliar a performance (MAE, RMSE, MAPE, SMAPE):

    make evaluate-AAPL

Exibe as métricas de erro para o retorno previsto vs. o real.

---

### 4. Rodar a API

Inicie a API localmente:

    make api

Acesse:
- Swagger UI: http://localhost:8000/docs
- Métricas Prometheus: http://localhost:8000/metrics

---

### 5. Fazer uma previsão

Via makefile:

    make predict-AAPL

Ou manualmente, por curl:

    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"symbol": "AAPL", "seq_length": 180}'

A resposta inclui o preço de fechamento previsto e o retorno percentual esperado.

---

### 6. Monitorar com TensorBoard

Se habilitar logs do tuner:

    make tensorboard

Acesse em http://localhost:6006

---

### 7. Deploy com Docker

Build da imagem:

    make docker-build

Rodar o container:

    make docker-run

---

## 🧠 Sobre Keras-Tuner

O projeto utiliza [Keras-Tuner](https://keras.io/keras_tuner/) para buscar automaticamente os melhores hiperparâmetros (número de unidades LSTM, taxa de dropout e learning-rate), testando diferentes combinações via RandomSearch. Isso permite um modelo mais robusto e ajustado ao ativo em questão.

---

## 📊 Métricas

- **MAE**: Erro Médio Absoluto do retorno previsto (em pontos percentuais)
- **RMSE**: Raiz do Erro Quadrático Médio
- **MAPE**: Erro Percentual Absoluto Médio (ajustado para retornos próximos de zero)
- **SMAPE**: Erro Percentual Absoluto Médio Simétrico

Foque especialmente em **MAE** e **RMSE** para avaliação de performance em finanças.

---

## 📌 Notas Finais

- Os dados são baixados automaticamente do Yahoo Finance usando yfinance.
- O pipeline inclui indicadores técnicos padrão do mercado.
- O código está modularizado para facilitar experimentação e customização.

Dúvidas? Sugestões? Abra uma issue ou contribua!