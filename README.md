# 📈 LSTM Stock Prediction com Indicadores Técnicos

Este repositório apresenta um pipeline completo para prever o **retorno diário** (Close/Open - 1) de ações utilizando redes LSTM. A solução inclui treinamento, API para inferência e monitoramento pronto para produção.

## Principais recursos
- LSTM com dropout para evitar overfitting
- Indicadores técnicos (SMA, EMA, RSI, MACD)
- Otimização de hiperparâmetros via [Keras-Tuner](https://keras.io/keras_tuner/)
- API em FastAPI para previsões online
- Monitoramento com Prometheus e Grafana
- Deploy simplificado com Docker

---

## 📁 Estrutura do Projeto
```
lstm-stock-prediction/
    api/
        main.py
        schemas.py
    data/
        raw/                 # CSVs baixados automaticamente
    models/                  # Modelos treinados e scalers
    src/
        train.py
        evaluate.py
        utils/
            data_utils.py
    Makefile
    requirements.txt
    README.md
```

---

## 🚀 Como Usar

### 1. Preparar ambiente
Testado com Python 3.11+. Crie um ambiente virtual e instale as dependências:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Treinar o modelo
Execute:
```bash
make train-AAPL
```
Padões utilizados:
- Sequência de 180 dias
- Busca de hiperparâmetros (unidades LSTM, dropout, learning rate)
- Indicadores técnicos como features

O modelo e o scaler são salvos em `models/`.
Para papéis da B3 use o sufixo `.SA`, exemplo:
```bash
make train-SBSP3.SA
```

### 3. Avaliar o modelo
Para obter MAE, RMSE, MAPE e SMAPE:
```bash
make evaluate-AAPL
```

### 4. Executar a API
Inicie a API localmente:
```bash
make api
```
Acesse:
- Swagger UI: <http://localhost:8000/docs>
- Métricas Prometheus: <http://localhost:8000/metrics>

### 5. Fazer uma previsão
Com o Makefile:
```bash
make predict-AAPL
```
Ou manualmente:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "seq_length": 180}'
```
A resposta inclui o preço de fechamento previsto e o retorno esperado.

### 6. Monitorar com TensorBoard
Se habilitar logs do tuner:
```bash
make tensorboard
```
Acesse em <http://localhost:6006>

### 7. Deploy com Docker Compose
Para construir as imagens e iniciar todos os serviços:
```bash
docker-compose up --build
```

### 8. Grafana e Prometheus
- **Grafana**: <http://localhost:3000> (user: `admin`, password: `admin`, dashboard: *Flask monitoring*)
- **Prometheus**: <http://localhost:9090>

---

## 🧠 Sobre Keras-Tuner
O projeto utiliza o Keras-Tuner para testar diferentes combinações de hiperparâmetros (unidades LSTM, dropout e learning rate), resultando em modelos mais robustos.

---

## 📊 Métricas
- **MAE** – Erro Médio Absoluto
- **RMSE** – Raiz do Erro Quadrático Médio
- **MAPE** – Erro Percentual Absoluto Médio (ajustado para retornos próximos de zero)
- **SMAPE** – Erro Percentual Absoluto Médio Simétrico

---

## 📌 Notas Finais
- Dados obtidos automaticamente do Yahoo Finance via `yfinance`
- Pipeline modular para facilitar customização
- Contribuições são bem-vindas! Abra uma issue ou PR
