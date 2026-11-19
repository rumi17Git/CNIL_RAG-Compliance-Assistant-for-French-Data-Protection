.PHONY: help start stop restart clean logs status ingest query test eval

help:
	@echo "Available targets:"
	@echo "  start               - Start all services"
	@echo "  stop                - Stop all services"
	@echo "  restart             - Restart all services"
	@echo "  clean               - Full reset (remove containers, volumes, cache)"
	@echo "  logs                - Tail logs of all services"
	@echo "  status              - Show container status"
	@echo "  ingest              - Upload the CNIL PDF"
	@echo "  query Q='question'  - Ask a question; add P=v2 to pick a pipeline (v0, v1, v2, v2-faiss)"
	@echo "  test                - Run the sample queries"
	@echo "  eval                - Run the 40-question benchmark (needs evaluation/requirements.txt)"

start:
	docker compose up -d
	@sleep 5
	@$(MAKE) status

stop:
	docker compose down

restart: stop start

clean:
	docker compose down -v

logs:
	docker compose logs -f

status:
	docker compose ps

ingest:
	@echo "Ingesting CNIL PDF..."
	@curl -s -X POST http://localhost:8001/ingest \
	  -F "file=@./pdfs/cnil_guide_securite_personnelle_2024.pdf" \
	  | jq .

query:
	@if [ -z "$(Q)" ]; then \
		echo "Usage: make query Q='Your question here'"; \
		echo "Example: make query Q='Comment gérer les habilitations ?'"; \
		exit 1; \
	fi
	@echo "Question: $(Q)"
	@echo "Answer:"
	@curl -s -X POST http://localhost:8002/query \
	  -H "Content-Type: application/json" \
	  -d "{\"question\": \"$(Q)\", \"top_k\": 6, \"pipeline\": \"$(or $(P),v0)\"}" \
	  | jq -r '.answer'
	@echo ""

test:
	@./test_queries.sh

eval:
	cd evaluation && python evaluation.py