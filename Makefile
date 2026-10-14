PY ?= python
export PYTHONPATH := src

.PHONY: install corpus ingest eval calibrate run test lint demo

install:          ## install package + dev tools
	$(PY) -m pip install -e ".[dev]"

corpus:           ## regenerate the synthetic PDF corpus
	$(PY) scripts/make_corpus.py --out data/docs

ingest:           ## extract, chunk and index the PDFs
	$(PY) -m docanswer.ingest

calibrate:        ## choose the refusal threshold on the dev question set
	$(PY) -m docanswer.eval --calibrate

eval:             ## run the evaluation suite -> eval/results.{json,md}
	$(PY) -m docanswer.eval

run:              ## start API + UI on http://localhost:8000
	$(PY) -m uvicorn docanswer.api:create_app --factory --host 0.0.0.0 --port 8000

test:             ## unit + API tests
	$(PY) -m pytest

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

demo: install corpus ingest eval run   ## everything, offline, one command
