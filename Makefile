.PHONY: help install install-gemini test check run-generation run-vector-search run-rag run-judge run-gemini clean-artifacts clean

PYTHON := poetry run python
PYTEST := poetry run pytest

help:
	@echo "ExplainableLLM commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install project with Poetry"
	@echo "  make install-gemini   Install project with Gemini optional dependency"
	@echo ""
	@echo "Quality:"
	@echo "  make check            Validate Poetry config and run tests"
	@echo "  make test             Run test suite"
	@echo ""
	@echo "Examples:"
	@echo "  make run-generation   Run token generation example"
	@echo "  make run-vector-search Run vector search example"
	@echo "  make run-rag          Run RAG example"
	@echo "  make run-judge        Run LLM-as-a-judge example"
	@echo "  make run-gemini       Run Gemini Flash Lite example"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean-artifacts  Remove generated JSON artifacts"
	@echo "  make clean            Remove caches and generated artifacts"

install:
	poetry install

install-gemini:
	poetry install --extras gemini

test:
	$(PYTEST)

check:
	poetry check
	$(PYTEST)

run-generation:
	$(PYTHON) examples/basic_generation.py

run-vector-search:
	$(PYTHON) examples/vector_search.py

run-rag:
	$(PYTHON) examples/rag_answering.py

run-judge:
	$(PYTHON) examples/llm_as_judge.py

run-gemini:
	$(PYTHON) examples/gemini_flash_lite.py

clean-artifacts:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('artifacts').rglob('*.json')]"

clean: clean-artifacts
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('.').rglob('*.pyc')]; [p.rmdir() for p in sorted(Path('.').rglob('__pycache__'), reverse=True) if p.exists()]"
