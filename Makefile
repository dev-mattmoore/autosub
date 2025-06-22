# Makefile for AutoSub

.PHONY: help install lint format test clean run tail

help:
	@echo "AutoSub Makefile commands:"
	@echo "  install   Install Python dependencies into venv"
	@echo "  lint      Run flake8 on source files"
	@echo "  format    Run black on source files"
	@echo "  test      Run tests (if any)"
	@echo "  clean     Remove Python cache and build artifacts"
	@echo "  tail      Tail the autosub-latest.log file"

install:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

lint:
	. venv/bin/activate && flake8 autosub_cli.py

format:
	. venv/bin/activate && black autosub_cli.py

test:
	@echo "No tests defined yet."

clean:
	rm -rf *.log

tail:
	tail -f autosub-latest.log
