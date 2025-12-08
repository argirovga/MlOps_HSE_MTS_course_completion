.PHONY: help deps seed run test docs lint fmt full

DBT := dbt

help:
	@echo "Available targets: deps seed run test docs lint fmt full"

deps:
	$(DBT) deps

seed:
	$(DBT) seed

run:
	$(DBT) run

test:
	$(DBT) test

docs:
	$(DBT) docs generate

lint:
	sqlfluff lint .

fmt:
	sqlfluff fix .

full: deps seed run test
