#!/usr/bin/env python3
"""
RAGAS evaluation script for CNIL RAG system.
Uses Groq via LangChain as the evaluation LLM and local HuggingFace embeddings.
RAGAS version: 0.2.6
"""

import os
import requests
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, answer_relevancy
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig

# Force CPU — Quadro P1000 GPU not supported by current PyTorch build
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Prevent RAGAS from calling OpenAI
os.environ["OPENAI_API_KEY"] = "dummy"

# ── Configuration ─────────────────────────────────────────────────────────────
GENERATION_URL = "http://localhost:8002/query"
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY environment variable")

# ── Golden test set ───────────────────────────────────────────────────────────
test_data = [
    {
        "question": "Quelles sont les précautions élémentaires pour piloter la sécurité des données ?",
        "ground_truth": "Impliquer la direction, recenser les traitements et supports, définir un plan d'action, contrôler périodiquement les mesures, assurer une revue de direction annuelle."
    },
    {
        "question": "Quelles sont les règles pour les mots de passe selon la CNIL ?",
        "ground_truth": "Privilégier l'authentification multifacteur, limiter les tentatives d'accès, entropie adaptée, ne pas forcer le renouvellement périodique pour les utilisateurs, stocker les mots de passe hachés avec un sel."
    },
    {
        "question": "Comment gérer les habilitations des utilisateurs ?",
        "ground_truth": "Définir des profils d'habilitation, faire valider par un responsable, supprimer les accès obsolètes, réaliser une revue annuelle des habilitations."
    },
    {
        "question": "Quelles mesures pour sécuriser l'informatique mobile ?",
        "ground_truth": "Sensibiliser les utilisateurs, chiffrer les équipements et les communications, utiliser un VPN pour l'accès distant, verrouillage automatique des smartphones."
    },
    {
        "question": "Comment protéger le réseau informatique ?",
        "ground_truth": "Limiter les flux réseau, utiliser des pare‑feux, sécuriser le Wi‑Fi (WPA3), imposer un VPN pour l'accès distant, cloisonner le réseau (DMZ)."
    },
    {
        "question": "Quelles sont les bonnes pratiques pour les sauvegardes ?",
        "ground_truth": "Sauvegardes fréquentes (incrémentales quotidiennes), stocker une copie hors site, isoler une sauvegarde hors ligne, tester régulièrement la restauration, règle 3‑2‑1."
    },
    {
        "question": "Que faire en cas de violation de données personnelles ?",
        "ground_truth": "Analyser les traces, évaluer le risque, tenir un registre interne, notifier la CNIL dans les 72 heures, informer les personnes concernées en cas de risque élevé."
    },
    {
        "question": "Quels sont les risques liés au cloud et comment les maîtriser ?",
        "ground_truth": "Risques : accès du fournisseur aux données, transferts hors UE, mauvaise configuration. Maîtrise : analyser les risques, formaliser les obligations contractuelles, chiffrer les données, configurer les accès."
    },
    {
        "question": "Quelles précautions pour développer un système d'IA ?",
        "ground_truth": "Équipe pluridisciplinaire, tests continus, vérifier la qualité des données, éviter les copies inutiles, documenter le fonctionnement et les limitations."
    },
    {
        "question": "Comment sécuriser une API qui échange des données personnelles ?",
        "ground_truth": "Identifier les acteurs, limiter les données partagées, séparer les fonctions courantes de l'administration, journaliser les échanges, maintenir la documentation à jour."
    }
]


# ── RAG query helper ──────────────────────────────────────────────────────────
def query_rag(question: str, top_k: int = 3) -> dict:
    """Call the generation service and return answer + retrieved contexts."""
    resp = requests.post(
        GENERATION_URL,
        json={"question": question, "top_k": top_k},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Query failed [{resp.status_code}]: {resp.text}")
    data     = resp.json()
    contexts = [c["text"] for c in data.get("citations", [])]
    return {
        "answer":   data["answer"],
        "contexts": contexts,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Evaluation LLM — Groq LLaMA as judge
    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=2048,
    )

    # Local embedding model — same as ingestion/generation services
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cpu"},
    )

    # ── Collect RAG responses ─────────────────────────────────────────────────
    rows = []
    for item in test_data:
        q = item["question"]
        print(f"Querying: {q[:60]}...")
        try:
            result = query_rag(q)
            rows.append({
                "question":     q,
                "answer":       result["answer"],
                "contexts":     result["contexts"],
                "ground_truth": item["ground_truth"],
            })
        except Exception as e:
            print(f"  ERROR querying RAG: {e}")
            rows.append({
                "question":     q,
                "answer":       "",
                "contexts":     [],
                "ground_truth": item["ground_truth"],
            })

    # ── Run RAGAS evaluation ──────────────────────────────────────────────────
    dataset = Dataset.from_list(rows)

    print("\nRunning RAGAS evaluation — this may take a few minutes...")
    run_config = RunConfig(
        max_workers=1,        # one job at a time — avoids Groq rate limits
        timeout=120,          # 2 minutes per job
        max_retries=3,        # retry on transient failures
    )
    result = evaluate(
        dataset,
        metrics=[context_precision, faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,
    )

    # ── Display results ───────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("RAGAS Evaluation Results (10 questions)")
    print("=" * 50)
    df_results = result.to_pandas()
    for metric in ["context_precision", "faithfulness", "answer_relevancy"]:
        if metric in df_results.columns:
            score = df_results[metric].mean()
            print(f"  {metric:<25} {score:.4f}")
        else:
            print(f"  {metric:<25} N/A (column not found)")
    print("=" * 50)

    # ── Save detailed per-question results ────────────────────────────────────
    df_results.to_csv("evaluation_results.csv", index=False)
    print("\nDetailed results saved to evaluation_results.csv")


if __name__ == "__main__":
    main()