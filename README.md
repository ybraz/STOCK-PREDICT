# 📈 LSTM Stock Price Prediction

Projeto de previsão de preços de ações usando LSTM (Long Short-Term Memory) em Python.

Disponibiliza:
- Scripts de treino e avaliação de modelo.
- API para inferência via FastAPI.
- Métricas Prometheus para monitoramento.
- Dockerfile para deploy fácil.

## 📦 Estrutura do Projeto

```
lstm-stock-prediction/
├── api/
│   ├── main.py
│   └── schemas.py
├── data/
│   └── raw/              # CSVs baixados
├── models/               # Modelos e scalers salvos
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── utils/
│       └── data_utils.py
├── .gitignore
├── .env.example
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

## 🚀 Como usar

### 1. Preparar ambiente

```bash
make setup
```

(ou manualmente: criar `.venv`, ativar, instalar `requirements.txt`)

---

### 2. Treinar o modelo

Treino normal:

```bash
make train SYMBOL=DIS
```

Treino forte (recomendado):

```bash
make strong-train SYMBOL=DIS
```

**O treino forte usa:**
- Histórico desde 2010
- 180 dias de sequência
- 200 épocas
- Modelo LSTM mais profundo (128-64-32)

Os arquivos salvos serão:

```
models/lstm_<SYMBOL>.h5
models/scaler_<SYMBOL>.pkl
```

---

### 3. Avaliar o modelo

```bash
make evaluate SYMBOL=DIS
```

Exibe as métricas:

- MAE (Erro Médio Absoluto)
- RMSE (Erro Quadrático Médio)
- MAPE (Erro Percentual Médio)

---

### 4. Rodar a API localmente

```bash
make api
```

- A API estará em: `http://localhost:8000`
- Documentação Swagger: `http://localhost:8000/docs`
- Métricas Prometheus: `http://localhost:8000/metrics`

---

### 5. Fazer uma previsão

```bash
make predict SYMBOL=DIS
```

Ou manualmente via curl:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "DIS", "seq_length": 180}'
```

---

### 6. Monitorar com TensorBoard

```bash
make tensorboard
```

Acesse:

```
http://localhost:6006
```

---

### 7. Deploy com Docker

Buildar a imagem:

```bash
make docker-build
```

Rodar o container:

```bash
make docker-run
```
