# Variáveis
SYMBOL=SBSP3.SA

# Comandos

setup:
	python -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

train:
	python -m src.train --symbol $(SYMBOL)

strong-train:
	python -m src.train --symbol $(SYMBOL) --seq_length 180 --start_date 2010-01-01 --end_date 2025-01-01 --epochs 200

evaluate:
	python -m src.evaluate --symbol $(SYMBOL)

api:
	uvicorn api.main:app --reload

predict:
	curl -X POST http://localhost:8000/predict \
	  -H "Content-Type: application/json" \
	  -d '{"symbol": "$(SYMBOL)", "seq_length": 180}'

docker-build:
	docker build -t lstm-api .

docker-run:
	docker run -p 8000:8000 lstm-api

tensorboard:
	tensorboard --logdir=runs

clean:
	rm -rf models/*.h5 models/*.pkl data/raw/*.csv runs/

.PHONY: setup train strong-train evaluate api predict docker-build docker-run tensorboard clean
