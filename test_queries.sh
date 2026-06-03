#!/bin/bash
URL="http://localhost:8002/query"
QUERIES=(
    "Quelles sont les précautions élémentaires pour piloter la sécurité des données ?"
    "Quelles sont les règles pour les mots de passe selon la CNIL ?"
    "Comment gérer les habilitations des utilisateurs ?"
    "Quelles mesures pour sécuriser l'informatique mobile ?"
    "Comment protéger le réseau informatique ?"
    "Quelles sont les bonnes pratiques pour les sauvegardes ?"
    "Que faire en cas de violation de données personnelles ?"
    "Quels sont les risques liés au cloud et comment les maîtriser ?"
    "Quelles précautions pour développer un système d'IA ?"
    "Comment sécuriser une API qui échange des données personnelles ?"
    "Quel est le montant de l'amende en cas de non-conformité au RGPD ?"
)

for q in "${QUERIES[@]}"; do
    echo -e "\n\n===== QUESTION: $q ====="
    curl -s -X POST "$URL" \
         -H "Content-Type: application/json" \
         -d "{\"question\": \"$q\", \"top_k\": 3}" \
         | jq -r '.answer'
    echo -e "\n---"
done