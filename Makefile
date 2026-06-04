.PHONY: help env-bootstrap env-check data-download data-link smoke-plan

help:
	@echo "Targets:"
	@echo "  make env-bootstrap   # bootstrap conda env in tmux"
	@echo "  make env-check       # run env health check in tmux"
	@echo "  make data-download   # download official data in tmux"
	@echo "  make data-link       # link downloaded assets to upstream repo"
	@echo "  make smoke-plan SCENE=old_union CONFIG=<config.yml> METHOD=sfc-1"

env-bootstrap:
	bash scripts/run_in_tmux.sh env_bootstrap bootstrap_env "bash scripts/bootstrap_env.sh splatnav"

env-check:
	bash scripts/run_in_tmux.sh env_check env_health "bash scripts/env_health_check.sh splatnav"

data-download:
	bash scripts/run_in_tmux.sh data_download official_data "bash scripts/download_official_data.sh splatnav /data/howard/splatnav/datasets/splatnav_official"

data-link:
	bash scripts/link_official_assets.sh /data/howard/splatnav/datasets/splatnav_official

smoke-plan:
	@if [ -z "$(SCENE)" ] || [ -z "$(CONFIG)" ]; then \
		echo "Usage: make smoke-plan SCENE=old_union CONFIG=/path/to/config.yml [METHOD=sfc-1]"; \
		exit 1; \
	fi
	bash scripts/run_splatplan_smoke.sh "$(SCENE)" "$(CONFIG)" "$(or $(METHOD),sfc-1)"
