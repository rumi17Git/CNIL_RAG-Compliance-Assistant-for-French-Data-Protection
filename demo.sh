#!/bin/bash
cd "$(dirname "$0")"
make start

# Wait for ingestion to be healthy
until curl -s http://localhost:8001/ping > /dev/null; do
    sleep 2
done

# Check if collection has points
POINTS=$(curl -s "http://localhost:6333/collections/cnil_docs" | jq -r '.result.points_count // 0')
if [ "$POINTS" -eq 0 ]; then
    echo "No vectors found. Running ingestion..."
    make ingest
else
    echo "Vectors already present ($POINTS points)."
fi

echo "System ready. You can now run 'make query' or open Streamlit UI."