#!/usr/bin/env python3
"""
RAGAS evaluation script for CNIL RAG system.

Two‑phase execution:
  - EVAL_MODE=collect  : only query the RAG generation service and save raw responses
  - EVAL_MODE=evaluate : load previously saved raw responses and run RAGAS only
  - EVAL_MODE=full     : do both (default)

This separation lets you tweak the judge model or metrics without waiting
for the generation step again.
"""

import os
import json
import requests
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, answer_relevancy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from ragas.run_config import RunConfig

# Read settings (judge choice, API keys, model names) from the project's .env,
# so they don't have to be typed on the command line. Real environment
# variables still win.
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ── Mode selection ────────────────────────────────────────────────────────────
EVAL_MODE = os.getenv("EVAL_MODE", "full")   # "collect" | "evaluate" | "full"

# Force CPU — Quadro P1000 GPU not supported by current PyTorch build
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Prevent RAGAS from calling OpenAI
os.environ["OPENAI_API_KEY"] = "dummy"

# ── Configuration ─────────────────────────────────────────────────────────────
GENERATION_URL = "http://localhost:8002/query"

# Judge LLM runs locally via Ollama.
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "llama3.1:8b")

# Which model scores the answers. The RAG itself always runs locally; only the
# scoring step can use a hosted model, which is much faster and more reliable
# than a local 8B judge.
#   JUDGE_PROVIDER=ollama   (default, local, uses OLLAMA_JUDGE_MODEL)
#   JUDGE_PROVIDER=mistral  (needs MISTRAL_API_KEY)
#   JUDGE_PROVIDER=groq     (needs GROQ_API_KEY)
#   JUDGE_PROVIDER=anthropic (needs ANTHROPIC_API_KEY from the Claude Console)
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "ollama").lower()
DEFAULT_HOSTED_JUDGE = {"mistral": "mistral-small-latest", "groq": "llama-3.3-70b-versatile",
                        "anthropic": "claude-haiku-5-5"}
JUDGE_MODEL = os.getenv("JUDGE_MODEL", DEFAULT_HOSTED_JUDGE.get(JUDGE_PROVIDER, OLLAMA_JUDGE_MODEL))


def make_judge():
    """Return the LangChain model used by RAGAS as judge."""
    if JUDGE_PROVIDER == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=JUDGE_MODEL, temperature=0, max_retries=10)
    if JUDGE_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=JUDGE_MODEL, temperature=0, max_tokens=2048, max_retries=10)
    if JUDGE_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        kwargs = {}
        if JUDGE_MODEL.startswith("openai/gpt-oss"):
            # Reasoning model: keep its thinking short (saves tokens against the daily
            # limit) and leave room for the answer after it.
            kwargs["reasoning_effort"] = "low"
        return ChatGroq(model=JUDGE_MODEL, temperature=0, max_tokens=8192, max_retries=10, **kwargs)
    return Ollama(
        model=OLLAMA_JUDGE_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_predict=2048,   # judge replies are short JSON; long limits only let it ramble
        format="json",      # Ollama JSON mode: forces valid JSON, which RAGAS parses
    )
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Number of chunks retrieved per question. Chunks are now ~900 characters, so 6
# gives about as much context as 2 of the old full-page chunks.
EVAL_TOP_K = int(os.getenv("EVAL_TOP_K", 6))

# Which pipeline to evaluate (v0, v1, v2, v2-faiss). When set, result files get the
# pipeline name in them (raw_responses_v2.csv...) so runs don't overwrite each other.
# RUN_NAME overrides that suffix, e.g. RUN_NAME=v2_k8 for a variant.
EVAL_PIPELINE = os.getenv("EVAL_PIPELINE", "")
RUN_NAME = os.getenv("RUN_NAME", EVAL_PIPELINE)
_SUFFIX = f"_{RUN_NAME}" if RUN_NAME else ""

# How long to wait for one answer. A local 8B model on CPU occasionally needs
# more than 5 minutes for a long answer.
QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", 1200))  # the v3 agent makes several model calls per question

# Output files, relative to this script so it runs on any machine
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.getenv("RESULTS_DIR", os.path.join(SCRIPT_DIR, "results"))
RAW_RESPONSES_PATH = os.path.join(RESULTS_DIR, f"raw_responses{_SUFFIX}.csv")
EVALUATION_RESULTS_PATH = os.path.join(RESULTS_DIR, f"evaluation_results{_SUFFIX}.csv")

# The exact refusal string from the generation service's prompt template.
EXACT_FLAG = "je ne trouve pas cette information dans les documents fournis"

REFUSAL_MARKERS_FALLBACK = [
    "ne mentionne pas",
    "n'aborde pas",
    "ne traite pas",
    "n'est pas abordé",
    "n'est pas traité",
    "hors du champ",
    "hors de portée",
    "ne dispose pas de cette information",
    "ne fournit pas cette information",
    "aucune information",
    "le corpus ne",
    "la documentation ne",
    "ne précise pas",
]

def is_refusal(answer: str, refused_flag=None) -> tuple[bool, bool]:
    """Returns (flagged_as_refusal, matched_exact_flag).

    The generation service now returns a `refused` field (the model's
    [HORS_CORPUS] tag). When present it is the reliable signal; the text
    checks below are kept for older result files that don't have it.
    """
    # Results collected with the current service carry an explicit flag: trust it,
    # so partial answers that mention the refusal sentence are not counted as refusals.
    flag = str(refused_flag).strip().lower()
    answer = answer if isinstance(answer, str) else ""   # a CSV gap comes back as NaN
    a_stripped = answer.strip().lower()
    if flag == "true":
        return True, True
    if flag == "false":
        # Backup for replies where the model wrote the refusal sentence but forgot
        # the [HORS_CORPUS] tag: a short reply that opens with it is a refusal.
        bare = a_stripped.startswith(EXACT_FLAG) and len(a_stripped) < 200
        return bare, bare
    a = (answer or "").lower()
    if EXACT_FLAG in a:
        return True, True
    if any(marker in a for marker in REFUSAL_MARKERS_FALLBACK):
        return True, False
    return False, False

## ── Golden test set ───────────────────────────────────────────────────────────
# category values: "paraphrase", "cross_document", "everyday", "unanswerable"
test_data = [
    # ── Paraphrase (original 10 — close to the source chunk headings) ─────────
    {"category": "paraphrase", "question": "Quelles sont les précautions élémentaires pour piloter la sécurité des données ?",
     "ground_truth": "Il faut impliquer la direction, recenser les traitements et les supports utilisés, définir un plan d'action, contrôler périodiquement les mesures mises en place, et organiser une revue de direction annuelle."},
    {"category": "paraphrase", "question": "Quelles sont les règles pour les mots de passe selon la CNIL ?",
     "ground_truth": "La CNIL recommande de privilégier l'authentification multifacteur, de limiter les tentatives d'accès, d'utiliser une entropie adaptée, de ne pas forcer le renouvellement périodique pour les utilisateurs, et de stocker les mots de passe hachés avec un sel."},
    {"category": "paraphrase", "question": "Comment gérer les habilitations des utilisateurs ?",
     "ground_truth": "Il faut définir des profils d'habilitation, faire valider les accès par un responsable, supprimer les accès obsolètes, et réaliser une revue annuelle des habilitations."},
    {"category": "paraphrase", "question": "Quelles mesures pour sécuriser l'informatique mobile ?",
     "ground_truth": "Il faut sensibiliser les utilisateurs, chiffrer les équipements et les communications, utiliser un VPN pour l'accès distant, et activer le verrouillage automatique des smartphones."},
    {"category": "paraphrase", "question": "Comment protéger le réseau informatique ?",
     "ground_truth": "Il faut limiter les flux réseau, utiliser des pare-feux, sécuriser le Wi-Fi avec WPA3, imposer un VPN pour l'accès distant, et cloisonner le réseau via des zones démilitarisées (DMZ)."},
    {"category": "paraphrase", "question": "Quelles sont les bonnes pratiques pour les sauvegardes ?",
     "ground_truth": "Il faut faire des sauvegardes fréquentes et incrémentales quotidiennes, stocker une copie hors site, isoler une sauvegarde hors ligne, tester régulièrement la restauration, et suivre la règle 3-2-1."},
    {"category": "paraphrase", "question": "Que faire en cas de violation de données personnelles ?",
     "ground_truth": "Il faut analyser les traces disponibles, évaluer le risque pour les personnes, tenir un registre interne des violations, notifier la CNIL dans les 72 heures, et informer les personnes concernées en cas de risque élevé."},
    {"category": "paraphrase", "question": "Quels sont les risques liés au cloud et comment les maîtriser ?",
     "ground_truth": "Les risques incluent l'accès du fournisseur aux données, les transferts hors UE, et les mauvaises configurations. Pour les maîtriser, il faut analyser les risques, formaliser les obligations contractuelles, chiffrer les données, et configurer correctement les accès."},
    {"category": "paraphrase", "question": "Quelles précautions pour développer un système d'IA ?",
     "ground_truth": "Il faut constituer une équipe pluridisciplinaire, mettre en place des tests continus, vérifier la qualité des données utilisées, éviter les copies inutiles, et documenter le fonctionnement ainsi que les limitations du système."},
    {"category": "paraphrase", "question": "Comment sécuriser une API qui échange des données personnelles ?",
     "ground_truth": "Il faut identifier les acteurs impliqués, limiter les données partagées au strict nécessaire, séparer les fonctions courantes de l'administration, journaliser les échanges, et maintenir la documentation à jour."},

    # ── Cross-document synthesis — requires combining two provisions ──────────
    {"category": "cross_document", "question": "Comment les règles de mots de passe s'articulent-elles avec la gestion des habilitations des utilisateurs ayant des droits d'administration ?",
     "ground_truth": "Les comptes à privilèges administratifs doivent suivre à la fois les règles de mots de passe renforcées (authentification multifacteur, entropie adaptée) et le processus d'habilitation (validation par un responsable, revue annuelle, suppression des accès obsolètes)."},
    {"category": "cross_document", "question": "Si mon fournisseur cloud expose une API pour échanger des données personnelles, quelles précautions cumulées dois-je prendre ?",
     "ground_truth": "Il faut combiner les précautions liées au cloud (analyse des risques, obligations contractuelles, chiffrement, configuration des accès) et celles liées aux API (identification des acteurs, limitation des données partagées, journalisation des échanges)."},
    {"category": "cross_document", "question": "En cas de violation de données causée par une sauvegarde compromise, quelles obligations s'ajoutent aux bonnes pratiques de sauvegarde ?",
     "ground_truth": "Au-delà des bonnes pratiques de sauvegarde (copie hors site, sauvegarde hors ligne, tests de restauration), une violation impose d'analyser les traces, d'évaluer le risque, de notifier la CNIL sous 72 heures et d'informer les personnes concernées si le risque est élevé."},
    {"category": "cross_document", "question": "Comment sécuriser un accès VPN depuis un smartphone en télétravail en combinant les recommandations sur l'informatique mobile et sur le réseau ?",
     "ground_truth": "Il faut chiffrer les communications et verrouiller automatiquement le smartphone conformément aux règles sur l'informatique mobile, tout en imposant un VPN et en sécurisant le Wi-Fi conformément aux règles de protection du réseau."},
    {"category": "cross_document", "question": "Lors du développement d'un système d'IA qui interroge une API externe contenant des données personnelles, quelles précautions faut-il combiner ?",
     "ground_truth": "Il faut appliquer les précautions de développement d'IA (équipe pluridisciplinaire, tests continus, qualité des données, documentation) en plus des précautions de sécurisation d'API (limitation des données partagées, journalisation, documentation à jour)."},
    {"category": "cross_document", "question": "Comment articuler la revue annuelle des habilitations avec le plan d'action de pilotage de la sécurité des données ?",
     "ground_truth": "La revue annuelle des habilitations doit s'inscrire dans le contrôle périodique prévu par le plan d'action de pilotage de la sécurité, et ses résultats doivent alimenter la revue de direction annuelle."},
    {"category": "cross_document", "question": "Si une sauvegarde hors ligne est stockée chez un prestataire cloud, quelles obligations s'ajoutent aux bonnes pratiques de sauvegarde ?",
     "ground_truth": "Il faut respecter les bonnes pratiques de sauvegarde tout en ajoutant les obligations liées au cloud : formaliser les obligations contractuelles avec le prestataire, chiffrer les données stockées, et configurer correctement les accès."},
    {"category": "cross_document", "question": "Quelles mesures de mots de passe et de réseau s'appliquent à un accès distant administrateur ?",
     "ground_truth": "Un accès distant administrateur doit combiner authentification multifacteur et entropie adaptée pour le mot de passe, avec un VPN et un réseau cloisonné pour la connexion."},
    {"category": "cross_document", "question": "Comment le registre des violations de données s'articule-t-il avec le recensement des traitements et des supports ?",
     "ground_truth": "Le recensement des traitements et des supports, prévu dans le pilotage général de la sécurité, sert de base pour analyser l'impact d'une violation et alimenter le registre interne des violations."},
    {"category": "cross_document", "question": "Quelles précautions cumulées faut-il prendre pour un système d'IA hébergé dans le cloud ?",
     "ground_truth": "Il faut combiner les précautions de développement d'IA (qualité des données, documentation, tests continus) avec les précautions cloud (obligations contractuelles, chiffrement, gestion des accès)."},

    # ── Everyday phrasing — same substance, non-legal vocabulary ──────────────
    {"category": "everyday", "question": "Je viens de me faire pirater mon site, dois-je prévenir quelqu'un et sous combien de temps ?",
     "ground_truth": "Oui, il faut évaluer le risque pour les personnes concernées et notifier la CNIL dans les 72 heures si une violation de données personnelles est confirmée, en informant les personnes concernées en cas de risque élevé."},
    {"category": "everyday", "question": "Mes employés utilisent le même mot de passe depuis 3 ans, est-ce grave ?",
     "ground_truth": "La CNIL ne recommande pas de forcer un renouvellement périodique, mais insiste plutôt sur une entropie suffisante et l'authentification multifacteur ; l'ancienneté seule n'est pas le problème principal."},
    {"category": "everyday", "question": "Je stocke mes fichiers clients sur un Google Drive, est-ce que c'est risqué ?",
     "ground_truth": "Cela relève des risques liés au cloud : accès du fournisseur aux données, transferts hors UE possibles, et risque de mauvaise configuration ; il faut analyser ces risques et formaliser des garanties contractuelles."},
    {"category": "everyday", "question": "Un ancien salarié a-t-il encore accès à nos outils après son départ, et est-ce un problème ?",
     "ground_truth": "S'il a encore accès, c'est un problème : les accès obsolètes doivent être supprimés dans le cadre de la gestion des habilitations, avec une revue régulière des droits."},
    {"category": "everyday", "question": "Je veux utiliser une IA pour trier les CV des candidats, y a-t-il des précautions particulières ?",
     "ground_truth": "Oui, il faut vérifier la qualité des données utilisées, documenter le fonctionnement et les limites du système, et s'appuyer sur une équipe pluridisciplinaire avec des tests continus."},
    {"category": "everyday", "question": "Mon ordinateur portable professionnel a été volé dans le train, que dois-je faire ?",
     "ground_truth": "Si l'équipement contenait des données personnelles non chiffrées, il faut traiter cela comme une possible violation : analyser le risque et notifier la CNIL sous 72 heures si nécessaire ; le chiffrement des équipements mobiles est justement la mesure préventive recommandée."},
    {"category": "everyday", "question": "Est-ce grave si je n'ai pas de copie de sauvegarde ailleurs que sur le même disque dur ?",
     "ground_truth": "Oui, cela ne respecte pas la règle 3-2-1 ni la recommandation de stocker une copie hors site et une sauvegarde hors ligne isolée."},
    {"category": "everyday", "question": "On m'a demandé de donner l'accès Wi-Fi de mon entreprise à un prestataire externe, comment faire correctement ?",
     "ground_truth": "Il faut sécuriser l'accès avec un Wi-Fi en WPA3, limiter les flux réseau accordés au prestataire, et idéalement cloisonner cet accès via une zone dédiée plutôt que de donner un accès direct au réseau interne."},
    {"category": "everyday", "question": "Je veux donner à mon commercial l'accès à toutes les données clients, est-ce que je peux faire ça facilement ?",
     "ground_truth": "Non, il faut définir un profil d'habilitation adapté à son rôle plutôt qu'un accès total, faire valider ce profil par un responsable, et le réévaluer lors de la revue annuelle des habilitations."},
    {"category": "everyday", "question": "On me demande de supprimer les données d'un client, mais elles sont dans nos sauvegardes aussi, qu'est-ce que je dois faire ?",
     "ground_truth": "La suppression doit être prise en compte y compris dans les sauvegardes, ce qui implique d'intégrer cette contrainte dans le pilotage de la sécurité et le recensement des traitements et supports concernés."},

    # ── Out of scope — wrong domain/jurisdiction entirely ──────────────────────
    # Nothing in the corpus is even topically close. Mainly a regression check
    # that the "Je ne trouve pas..." flag still fires on obvious cases.
    {"category": "out_of_scope", "question": "Quelles sont les règles de la CNIL concernant le California Consumer Privacy Act (CCPA) ?",
     "ground_truth": "Le CCPA est une loi américaine hors du champ de compétence de la CNIL ; le corpus ne fournit pas d'information à ce sujet."},
    {"category": "out_of_scope", "question": "Quel est le coût moyen facturé par les DPO externes recommandés par la CNIL ?",
     "ground_truth": "Le corpus ne traite pas de tarification des prestations de délégué à la protection des données ; ce n'est pas un sujet de guidance CNIL."},
    {"category": "out_of_scope", "question": "Quelle est la position de la CNIL sur l'utilisation d'implants neuronaux pour l'authentification biométrique ?",
     "ground_truth": "Cette question porte sur une technologie hypothétique non couverte par la CNIL ; le corpus ne fournit pas d'information à ce sujet."},

    # ── In-scope, not covered by this corpus — the harder test ─────────────────
    # Real CNIL subject matter (sanctions, transfers, registres, enforcement
    # stats) that simply isn't addressed by this security guide. The retriever
    # can plausibly pull back a semantically-adjacent chunk here, so this is
    # what actually stresses whether faithfulness holds instead of the model
    # stitching an adjacent chunk into a confident but fabricated answer.
    {"category": "in_scope_gap", "question": "Quel est le montant exact de l'amende CNIL pour une entreprise de 47 salariés en cas de notification tardive de 3 jours ?",
     "ground_truth": "Cette question dépasse le périmètre du guide, qui donne des recommandations de sécurité mais pas de barème de sanctions chiffré par taille d'entreprise ou délai de retard."},
    {"category": "in_scope_gap", "question": "Que dit la CNIL sur la révocation du consentement spécifique à la blockchain ?",
     "ground_truth": "Le guide de sécurité des données ne traite pas de mécanismes spécifiques à la blockchain ; cette question sort du périmètre du corpus."},
    {"category": "in_scope_gap", "question": "Quel est le délai exact de conservation des données de géolocalisation pour les véhicules autonomes selon la CNIL ?",
     "ground_truth": "Cette question porte sur un cas d'usage très spécifique non couvert par le guide général de sécurité des données personnelles."},
    {"category": "in_scope_gap", "question": "La CNIL a-t-elle publié des exigences spécifiques pour les crypto-monnaies et la protection des données des porte-monnaie numériques ?",
     "ground_truth": "Le corpus ne contient pas de recommandations spécifiques aux crypto-monnaies ou aux porte-monnaie numériques."},
    {"category": "in_scope_gap", "question": "Quel formulaire dois-je remplir auprès de la CNIL pour transférer des données vers une filiale basée en Australie ?",
     "ground_truth": "Le guide ne détaille pas de formulaire ou de procédure administrative spécifique pour les transferts internationaux vers l'Australie."},
    {"category": "in_scope_gap", "question": "Combien d'amendes la CNIL a-t-elle prononcées en 2024 pour des violations liées à l'informatique mobile ?",
     "ground_truth": "Le guide fournit des recommandations de sécurité mais pas de statistiques sur le nombre de sanctions prononcées."},
    {"category": "in_scope_gap", "question": "La CNIL impose-t-elle un format de fichier spécifique (par exemple .xml ou .json) pour les registres de traitement ?",
     "ground_truth": "Le guide ne précise pas de format de fichier obligatoire pour la tenue des registres."},
]

# ── RAG query helper ──────────────────────────────────────────────────────────
def query_rag(question: str, top_k: int = EVAL_TOP_K) -> dict:
    """Call the generation service and return answer + retrieved contexts."""
    payload = {"question": question, "top_k": top_k}
    if EVAL_PIPELINE:
        payload["pipeline"] = EVAL_PIPELINE
    resp = requests.post(
        GENERATION_URL,
        json=payload,
        timeout=QUERY_TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Query failed [{resp.status_code}]: {resp.text}")
    data = resp.json()
    contexts = [c["text"] for c in data.get("citations", [])]
    return {
        "answer": data["answer"],
        "refused": data.get("refused"),
        "pipeline": data.get("pipeline", ""),
        "latency_ms": data.get("latency_ms"),
        "llm_calls": data.get("llm_calls"),
        "trace": " | ".join(data.get("trace", [])),
        "contexts": contexts,
    }

# ── Phase 1: Collect raw responses ───────────────────────────────────────────
def _query_row(item: dict) -> dict:
    q = item["question"]
    try:
        result = query_rag(q)
        return {"category": item["category"], "question": q, "answer": result["answer"],
                "refused": result["refused"], "pipeline": result["pipeline"],
                "latency_ms": result["latency_ms"], "llm_calls": result.get("llm_calls"),
                "trace": result.get("trace", ""), "contexts": result["contexts"],
                "ground_truth": item["ground_truth"]}
    except Exception as e:
        print(f"  ERROR querying RAG: {e}")
        return {"category": item["category"], "question": q, "answer": "", "refused": None,
                "pipeline": EVAL_PIPELINE, "latency_ms": None, "contexts": [],
                "ground_truth": item["ground_truth"]}


def is_failed(row: dict) -> bool:
    """A question whose query failed (timeout, server error) has no answer at all."""
    a = row.get("answer")
    return not isinstance(a, str) or not a.strip()


def retry_failed(rows: list) -> list:
    """Ask again only the questions that have no answer."""
    failed = [i for i, r in enumerate(rows) if is_failed(r)]
    if failed:
        print(f"\nRetrying {len(failed)} question(s) that got no answer...")
    for i in failed:
        item = {k: rows[i][k] for k in ("category", "question", "ground_truth")}
        print(f"[{item['category']}] Retrying: {item['question'][:60]}...")
        rows[i] = _query_row(item)
    return rows


def collect_responses() -> list:
    """Query the RAG service for all test questions, then retry any that failed once."""
    rows = []
    for item in test_data:
        print(f"[{item['category']}] Querying: {item['question'][:60]}...")
        rows.append(_query_row(item))
    return retry_failed(rows)


def save_raw(rows: list, path: str):
    out = []
    for row in rows:
        row = dict(row)
        if not isinstance(row["contexts"], str):
            row["contexts"] = json.dumps(row["contexts"])
        out.append(row)
    pd.DataFrame(out).to_csv(path, index=False)
    print(f"Raw responses saved to {path}")


def load_raw(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw responses file '{path}' not found. Run with EVAL_MODE=collect first.")
    rows = pd.read_csv(path).to_dict("records")
    for row in rows:
        c = row.get("contexts")
        row["contexts"] = json.loads(c) if isinstance(c, str) else []
    return rows

# ── Phase 2: RAGAS evaluation on previously collected rows ──────────────────
def checkpoint_key(rows: list) -> str:
    """Short id for this set of answers + judge, so checkpoints never mix runs."""
    import hashlib
    h = hashlib.md5()
    for r in rows:
        h.update((r["question"] + "\x1f" + str(r["answer"])).encode("utf-8"))
    h.update(f"{JUDGE_PROVIDER}/{JUDGE_MODEL}".encode("utf-8"))
    return h.hexdigest()[:10]


def run_ragas_evaluation(rows: list):
    """Compute refusal flags, run RAGAS, print results, and save evaluation CSV."""
    # Refusal check
    for row in rows:
        flagged, exact = is_refusal(row["answer"], row.get("refused"))
        row["flagged_as_refusal"] = flagged
        row["matched_exact_flag"] = exact

    scorable_rows = [r for r in rows if not r["flagged_as_refusal"]]
    skipped_rows = [r for r in rows if r["flagged_as_refusal"]]

    if skipped_rows:
        print(f"\n{len(skipped_rows)} question(s) flagged as a correct refusal — "
              f"skipping RAGAS metrics for these, using refusal check only.")

    # Initialise judge and embeddings only now (they are needed only for RAGAS)
    llm = make_judge()
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cpu"},
    )

    run_config = RunConfig(
        max_workers=1,
        timeout=600,       # 10 minutes per job; a stuck job is skipped, not waited on
        max_retries=6,     # hosted free tiers rate-limit: retry with backoff
        max_wait=60,
    )

    # Run RAGAS one question at a time and save after each one, so a run stopped by
    # a daily token limit (or Ctrl+C) can continue later instead of starting over.
    metrics = [context_precision, faithfulness, answer_relevancy]
    metric_names = ["context_precision", "faithfulness", "answer_relevancy"]
    checkpoint_path = os.path.join(RESULTS_DIR, f"checkpoint_{checkpoint_key(rows)}.csv")
    done = {}
    if os.path.exists(checkpoint_path):
        prev = pd.read_csv(checkpoint_path)
        for rec in prev.to_dict("records"):
            if all(pd.notna(rec.get(m)) for m in metric_names):
                done[rec["question"]] = rec
        print(f"\nResuming: {len(done)} question(s) already fully scored in {checkpoint_path}")

    if scorable_rows:
        todo = [r for r in scorable_rows if r["question"] not in done]
        print(f"\nRunning RAGAS evaluation on {len(todo)} of {len(scorable_rows)} scorable "
              f"question(s) (judge: {JUDGE_PROVIDER}/{JUDGE_MODEL}) — this may take a while...")
        empty_in_a_row = 0
        for i, r in enumerate(todo, start=1):
            dataset = Dataset.from_list([{k: r[k] for k in ("question", "answer", "contexts", "ground_truth")}])
            res = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings,
                           run_config=run_config, raise_exceptions=False, show_progress=False)
            scores = res.to_pandas().iloc[0]
            rec = {"question": r["question"], "category": r["category"]}
            rec.update({m: scores.get(m) for m in metric_names})
            done[r["question"]] = rec
            pd.DataFrame(list(done.values())).to_csv(checkpoint_path, index=False)
            got = sum(pd.notna(rec[m]) for m in metric_names)
            print(f"  [{i}/{len(todo)}] {r['category']}: {got}/3 metrics scored")

            # Two questions in a row with no score at all usually means the judge's
            # rate or daily limit is reached: stop and keep what we have.
            empty_in_a_row = empty_in_a_row + 1 if got == 0 else 0
            if empty_in_a_row >= 2:
                print("\nTwo questions in a row got no score: the judge's token limit is probably "
                      "reached. Progress is saved; run the same command again later to continue.")
                break

        by_q = done
        scored_df = pd.DataFrame([
            {"question": r["question"], "answer": r["answer"], "contexts": r["contexts"],
             "ground_truth": r["ground_truth"],
             **{m: by_q.get(r["question"], {}).get(m) for m in metric_names},
             "category": r["category"], "flagged_as_refusal": r["flagged_as_refusal"],
             "matched_exact_flag": r["matched_exact_flag"], "skipped_scoring": False}
            for r in scorable_rows
        ])
        for m in metric_names:
            scored_df[m] = pd.to_numeric(scored_df[m], errors="coerce")
    else:
        scored_df = pd.DataFrame(columns=[
            "question", "answer", "contexts", "ground_truth",
            "context_precision", "faithfulness", "answer_relevancy",
            "category", "flagged_as_refusal", "matched_exact_flag",
            "skipped_scoring",
        ])

    # Build skipped rows (no RAGAS metrics)
    if skipped_rows:
        skipped_df = pd.DataFrame([
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r["ground_truth"],
                "context_precision": None,
                "faithfulness": None,
                "answer_relevancy": None,
                "category": r["category"],
                "flagged_as_refusal": r["flagged_as_refusal"],
                "matched_exact_flag": r["matched_exact_flag"],
                "skipped_scoring": True,
            }
            for r in skipped_rows
        ])
        df_results = pd.concat([scored_df, skipped_df], ignore_index=True)
    else:
        df_results = scored_df

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"RAGAS Evaluation Results — overall ({len(df_results)} questions)")
    print("=" * 60)
    for metric in ["context_precision", "faithfulness", "answer_relevancy"]:
        if metric in df_results.columns:
            scored = df_results[metric].notna().sum()
            print(f"  {metric:<25} {df_results[metric].mean():.4f}  (scored {scored} answers)")
        else:
            print(f"  {metric:<25} N/A")

    print("\n" + "=" * 60)
    print("Per-category breakdown")
    print("=" * 60)
    for category in ["paraphrase", "cross_document", "everyday", "out_of_scope", "in_scope_gap"]:
        subset = df_results[df_results["category"] == category]
        if subset.empty:
            continue
        print(f"\n  [{category}]  n={len(subset)}")
        for metric in ["context_precision", "faithfulness", "answer_relevancy"]:
            if metric in subset.columns:
                print(f"    {metric:<23} {subset[metric].mean():.4f}  "
                      f"(scored {subset[metric].notna().sum()} of {len(subset)})")

    # Refusal rates for out_of_scope and in_scope_gap
    for category in ["out_of_scope", "in_scope_gap"]:
        subset = df_results[df_results["category"] == category]
        if subset.empty:
            continue
        refusal_rate = subset["flagged_as_refusal"].mean()
        exact_rate = subset["matched_exact_flag"].mean()
        print(f"\n  [{category}] refusal rate: {refusal_rate:.1%} "
              f"(exact flag match: {exact_rate:.1%})")
    print("\n  (Refusal rate = fraction where the system correctly declined)")
    print("   'Exact flag match' isolates cases hitting the literal instructed string.")

    answerable = df_results[~df_results["category"].isin(["out_of_scope", "in_scope_gap"])]
    if not answerable.empty:
        false_refusal_rate = answerable["flagged_as_refusal"].mean()
        print(f"\n  False-refusal rate on answerable questions: {false_refusal_rate:.1%}")

    print("=" * 60)

    # Save detailed results
    df_results.to_csv(EVALUATION_RESULTS_PATH, index=False)
    print(f"\nDetailed results saved to {EVALUATION_RESULTS_PATH}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Mode: {EVAL_MODE}")
    print(f"Judge: {JUDGE_PROVIDER}/{JUDGE_MODEL}")
    print(f"Chunks per question (EVAL_TOP_K): {EVAL_TOP_K}")
    print(f"Pipeline: {EVAL_PIPELINE or 'service default'}  |  results: {RAW_RESPONSES_PATH}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_responses_path = RAW_RESPONSES_PATH

    # Modes: full (collect + evaluate), collect, evaluate, and retry (ask again only
    # the questions that got no answer in the saved file, then save it).
    if EVAL_MODE in ("full", "collect"):
        print("\n--- Phase 1: Collecting RAG responses ---")
        save_raw(collect_responses(), raw_responses_path)

    if EVAL_MODE == "retry":
        print("\n--- Retrying failed questions ---")
        save_raw(retry_failed(load_raw(raw_responses_path)), raw_responses_path)

    if EVAL_MODE in ("full", "evaluate"):
        print("\n--- Phase 2: RAGAS evaluation ---")
        rows = load_raw(raw_responses_path)
        failed = [r for r in rows if is_failed(r)]
        if failed:
            print(f"\nWARNING: {len(failed)} question(s) have no answer (the query failed) and are left out "
                  f"of every metric. Run with EVAL_MODE=retry to ask them again:")
            for r in failed:
                print(f"  - [{r['category']}] {r['question'][:70]}")
        rows = [r for r in rows if not is_failed(r)]
        run_ragas_evaluation(rows)
        latencies = [r.get("latency_ms") for r in rows if pd.notna(r.get("latency_ms"))]
        if latencies:
            print(f"\n  Average answer time: {sum(latencies) / len(latencies) / 1000:.1f} s per question")
        calls = [r.get("llm_calls") for r in rows if pd.notna(r.get("llm_calls"))]
        if calls:
            print(f"  Average model calls: {sum(calls) / len(calls):.1f} per question")

if __name__ == "__main__":
    main()