build:
	docker compose build

innit: build
	docker-compose up airflow-init

run:
	docker compose up
