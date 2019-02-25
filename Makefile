# BEGIN-EVAL makefile-parser --make-help Makefile

SHELL = /bin/bash
PYTHON = python
PIP = pip

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps       pip install -r requirements.txt"
	#@echo "    deps-test  pip install -r requirements_test.txt"
	@echo ""
	@echo "    install    pip install -e ."
	#@echo "    test       python -m pytest test"

# END-EVAL

deps:
	$(PIP) install -r requirements.txt

#deps-test:
#	$(PIP) install -r requirements_test.txt

install:
	$(PIP) install -e .

#test: test/assets
#	test -f model_dta_test.h5 || keraslm-rate train -m model_dta_test.h5 test/assets/*.txt
#	keraslm-rate test -m model_dta_test.h5 test/assets/*.txt
#	$(PYTHON) -m pytest test $(PYTEST_ARGS)

#test/assets:
#	test/prepare_gt.bash $@

.PHONY: help deps deps-test install test
