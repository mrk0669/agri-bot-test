# AgriBot - common tasks.  Run `make help` for the list.
#
# Windows users without GNU make: every recipe is a single command, so read the
# target you want and run it directly, or use `bash verify.sh` for `make verify`.

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest

.DEFAULT_GOAL := help
.PHONY: help install install-dev test test-fast lint sim sim-rows fusion tune bench preflight verify figures clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the runtime only (what the robot needs)
	$(PYTHON) -m pip install -e .

install-dev:  ## Install everything including test and simulation extras
	$(PYTHON) -m pip install -r requirements-dev.txt && $(PYTHON) -m pip install -e .

test:  ## Run the full test suite
	$(PYTEST) -q

test-fast:  ## Run everything except the slow seed sweeps
	$(PYTEST) -q -m "not slow"

cov:  ## Run the suite with a coverage report
	$(PYTEST) -q --cov=agribot --cov-report=term-missing:skip-covered

sim:  ## One-row software-in-the-loop mission
	$(PYTHON) -m agribot.app.simulate --seconds 45 --rows 1 --report

sim-rows:  ## Two-row mission, exercising the row-end turn
	$(PYTHON) -m agribot.app.simulate --seconds 120 --rows 2 --report

fusion:  ## Reproduce the Section 5.3 sensor-fusion study and figure
	$(PYTHON) tools/kalman_sim.py --sweep 8

tune:  ## Score the configured PID gains against the closed-loop model
	$(PYTHON) tools/tune_pid.py --check

bench:  ## Benchmark the perception pipeline
	$(PYTHON) tools/bench_perception.py --frames 120

preflight:  ## Config and detector checks (no hardware needed)
	$(PYTHON) -m agribot.app.preflight --skip-hardware

verify:  ## Everything in docs/VERIFICATION.md, in order
	@bash verify.sh

figures:  ## Regenerate reports/fig5_kalman.png
	$(PYTHON) tools/kalman_sim.py

clean:  ## Remove caches and generated reports
	rm -rf .pytest_cache .coverage htmlcov reports/*.png
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
