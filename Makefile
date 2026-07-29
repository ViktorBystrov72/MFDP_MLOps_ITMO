SHELL := /bin/bash

SRC_DIRS := src scripts tests
VENV_ACTIVATE := $(firstword $(wildcard .venv/bin/activate venv/bin/activate))
ifeq ($(VENV_ACTIVATE),)
  RUN =
else
  RUN = source $(VENV_ACTIVATE) &&
endif

.PHONY: lint fix test migrate

lint:
	$(RUN) ruff check $(SRC_DIRS) && $(RUN) ruff format --check $(SRC_DIRS)

fix:
	$(RUN) ruff check --fix $(SRC_DIRS) && $(RUN) ruff format $(SRC_DIRS)

test:
	$(RUN) export PYTHONPATH="$(CURDIR)/src:$$PYTHONPATH" && pytest tests

