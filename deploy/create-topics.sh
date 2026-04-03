#!/bin/bash

set -euo pipefail

BOOTSTRAP_SERVER="${BOOTSTRAP_SERVER:-kafka:9092}"
TOPICS=(
  "raw-sensor-data"
  "cleaned-features"
  "anomalies-tagged"
)

for topic in "${TOPICS[@]}"; do
  kafka-topics \
    --bootstrap-server "${BOOTSTRAP_SERVER}" \
    --create \
    --if-not-exists \
    --topic "${topic}" \
    --partitions 1 \
    --replication-factor 1
done

echo "Kafka topics ready:"
kafka-topics --bootstrap-server "${BOOTSTRAP_SERVER}" --list
