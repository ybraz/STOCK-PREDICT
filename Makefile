PY=python
SRC=src
API=api

train-%:
	PYTHONPATH=. $(PY) $(SRC)/train.py $* --seq 180 --epochs 30

strong-train-%:
	PYTHONPATH=. $(PY) $(SRC)/train.py $* --seq 180 --epochs 200 --trials 20 --start 2005-01-01

evaluate-%:
	PYTHONPATH=. $(PY) $(SRC)/evaluate.py $* --seq 180

api:
	uvicorn $(API).main:app --reload --port 8000

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

predict-%:
	curl -X POST http://localhost:8000/predict \
	  -H "Content-Type: application/json" \
	  -d '{"symbol": "$*", "seq_length": 180}'

tensorboard:
	tensorboard --logdir tuner

docker-build:
	docker build -t lstm-stock-prediction .

docker-run:
	docker run -p 8000:8000 lstm-stock-prediction