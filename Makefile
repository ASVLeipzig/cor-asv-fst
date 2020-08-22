
# Directory to install to ('$(PREFIX)')
PREFIX ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV),$(CURDIR)/.local)

SHELL = /bin/bash
PYTHON = python
PIP = pip

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps-ubuntu  Dependencies for deployment in an ubuntu/debian linux"
	@echo "                 we need libstdc++ > 5.0 (for codecvt, std::make_unique etc)"
	@echo "                 since pynini 2.0.9, we need libfst-dev > 1.7"
	@echo ""
	@echo "  Variables"
	@echo ""
	@echo "    PREFIX  Directory to install to ('$(PREFIX)')"

# END-EVAL

PKG_CONFIG_PATH := $(PREFIX)/lib/pkgconfig
export PKG_CONFIG_PATH
deps: $(PREFIX)/lib/libfst.so.17
	$(PIP) install -r requirements.txt

#deps-test:
#	$(PIP) install -r requirements_test.txt

# Dependencies for deployment in an ubuntu/debian linux
# we need libstdc++ > 5.0 (for codecvt, std::make_unique etc)
# since pynini 2.0.9, we need libfst-dev > 1.7
deps-ubuntu:
	apt-get install -y \
		g++ libfst-dev \
		wget tar gzip

$(PREFIX)/lib/libfst.so.17: openfst-1.7.5.tar.gz
	tar zxvf $<
	cd openfst-1.7.5 && ./configure --enable-grm --enable-python --prefix=$(PREFIX) && $(MAKE) install

openfst-1.7.5.tar.gz:
	wget -nv http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.5.tar.gz

# pip install . (incl. deps)
install: deps
	$(PIP) install .

# Run unit tests
test:
	$(PYTHON) -m pytest --continue-on-collection-errors tests
	$(warning This test only covers basic pynini functions. There are no smoke/regressions tests for cor-asv-fst, yet!)

#test: test/assets
#	test -f model_dta_test.h5 || keraslm-rate train -m model_dta_test.h5 test/assets/*.txt
#	keraslm-rate test -m model_dta_test.h5 test/assets/*.txt
#	$(PYTHON) -m pytest test $(PYTEST_ARGS)

#test/assets:
#	test/prepare_gt.bash $@

.PHONY: help deps-ubuntu deps deps-test install test
