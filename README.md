# 📈 LSTM Stock Price Prediction API

Projeto de previsão de preços de ações usando uma rede **LSTM** treinada em séries temporais históricas e disponibilizada via **FastAPI**.

---

## 🚀 Funcionalidades

- Treina um modelo **LSTM** com dados históricos do Yahoo Finance.
- Avalia o modelo com métricas **MAE**, **RMSE** e **MAPE**.
- Expõe uma **API REST** para inferência do próximo preço.
- Monitoramento de métricas via **Prometheus** (`/metrics`).
- Deploy fácil via **Docker**.

---

## 📦 Estrutura de Diretórios

```text
lstm-stock-prediction/
├── api/
│   ├── main.py
│   └── schemas.py
├── data/
│   └── raw/               # CSVs baixados
├── models/
│   ├── lstm_stock.h5       # Modelo treinado
│   └── scaler.pkl          # Scaler salvo
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── utils/
│       └── data_utils.py
├── .env.example
├── requirements.txt
├── Dockerfile
├── Makefile
└── README.md
```

---

## ⚙️ Instalação

1. Clone o projeto:

```bash
git clone https://github.com/seu-usuario/lstm-stock-prediction.git
cd lstm-stock-prediction
```

2. Crie o ambiente virtual e instale dependências:

```bash
make setup
```

---

## 🏋️‍♂️ Treino do Modelo

Treina o modelo com dados históricos:

```bash
make train SYMBOL=DIS
```

- O modelo será salvo em `models/lstm_stock.h5`
- O scaler será salvo em `models/scaler.pkl`

---

## 📊 Avaliação

Avalia o modelo no conjunto de teste:

```bash
make evaluate SYMBOL=DIS
```

Exibe:

- MAE
- RMSE
- MAPE

---

## 🖥️ Subir a API Localmente

```bash
make api
```

API disponível em:

```
http://localhost:8000
```

---

## 🔮 Fazer Previsão

Com a API rodando, rode:

```bash
make predict SYMBOL=DIS
```

Ou manualmente via `curl`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "DIS", "seq_length": 60}'
```

🔵 Retorno esperado:

```json
{"next_price": 120.45}
```

---

## 📈 Monitoramento

- `GET /metrics` → métricas Prometheus
- `GET /health` → checagem de saúde

---

## 🐳 Deploy com Docker

Build da imagem:

```bash
make docker-build
```

Run do container:

```bash
make docker-run
```

---

## 📋 Variáveis de Ambiente

O projeto usa `.env` para parâmetros padrão.  
Exemplo (`.env.example`):

```bash
SYMBOL=DIS
START_DATE=2018-01-01
END_DATE=2024-07-20
SEQ_LENGTH=60
TEST_SIZE=0.2
VAL_SIZE=0.1
EPOCHS=50
BATCH_SIZE=32
```

---

## ✨ Melhorias Futuras

- Adicionar suporte a múltiplos ativos simultaneamente.
- Melhorar gestão de cache dos dados Yahoo.
- Implementar autenticação na API.
- Deploy automático em nuvem (AWS, GCP, Railway).