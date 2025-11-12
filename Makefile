.PHONY: smoke-test refresh-data

smoke-test:
	python3 -m scripts.smoke_test

refresh-data:
	python3 -m scripts.build_combined_properties
