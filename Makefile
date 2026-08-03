.PHONY: up down logs dev topics smoke rec airflow-up airflow-down

up:            ## start broker + stores + simulator + consumer + api
	docker compose up -d --build

dev:           ## same, plus Redpanda Console at http://localhost:8080
	docker compose --profile dev up -d --build

down:
	docker compose down

logs:
	docker compose logs -f simulator consumer api

topics:        ## create the topic with 3 partitions (key = user_id)
	docker compose exec redpanda rpk topic create user_events -p 3 -r 1 || true

smoke:         ## consume 10 events off the stream
	docker compose exec redpanda rpk topic consume user_events -n 10

rec:           ## sample call against the serving endpoint
	curl -s "http://localhost:8000/recommend?user_id=u_42&k=5" | python3 -m json.tool

airflow-up:
	docker compose -f docker-compose.airflow.yml up -d

airflow-down:
	docker compose -f docker-compose.airflow.yml down
