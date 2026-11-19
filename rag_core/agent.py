"""v3: a LangGraph agent around the v2 hybrid retriever.

  plan -> retrieve -> grade --yes--> generate -> check --yes--> done
                        ^      |no          |refused   |no
                        |      v            v          v
                        +-- rewrite       retry      regenerate (stricter) -> done
                      (max AGENT_MAX_REWRITES, then generate anyway)
  retry: only when the grader judged the passages relevant; one attempt, then
  the refusal stands

- plan:      split a question about two topics into two searches
- retrieve:  hybrid search per query, fused with RRF, then "small-to-big": each
             hit is expanded with its neighbouring chunks, within a size budget
- grade:     do the passages answer the question, at least in part?
- rewrite:   reformulate with the guide's vocabulary ("Dropbox" -> cloud)
- generate:  same grounded prompt as every other pipeline
- check:     is every claim supported by the context? If not, regenerate once
             with a stricter instruction

After the last rewrite the agent still generates: the grounded prompt refuses by
itself when nothing fits, so the agent never adds refusals of its own.
"""
import json
import re
from typing import List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph

from .prompts import PROMPT, TEMPLATE, format_context
from .retrieval import doc_key, rrf_fuse
from .text import parse_answer

PLAN_PROMPT = PromptTemplate.from_template(
    "Tu prépares une recherche dans le guide CNIL « Sécurité des données personnelles ». "
    "Si la question porte sur deux sujets distincts, découpe-la en deux requêtes courtes, une par sujet. "
    "Sinon, renvoie une seule requête : la question elle-même.\n"
    'Réponds uniquement en JSON : {{"requetes": ["...", "..."]}}\n\n'
    "Question : {question}"
)

GRADE_PROMPT = PromptTemplate.from_template(
    "Voici une question et des extraits du guide CNIL sur la sécurité des données personnelles. "
    "Ces extraits contiennent-ils des informations qui permettent de répondre, au moins en partie ?\n"
    'Réponds uniquement en JSON : {{"pertinent": "oui"}} ou {{"pertinent": "non"}}\n\n'
    "Question : {question}\n\nExtraits :\n{excerpts}"
)

REWRITE_PROMPT = PromptTemplate.from_template(
    "La recherche dans le guide CNIL n'a pas trouvé de passage utile pour cette question. "
    "Reformule-la en une requête courte avec le vocabulaire du guide (termes de sécurité informatique "
    "et de protection des données). Exemples : « Dropbox » devient « informatique en nuage (cloud), "
    "sous-traitant » ; « téléphone perdu » devient « perte d'équipement mobile, violation de données, chiffrement » ; "
    "« stagiaire qui part » devient « habilitations, suppression des accès ».\n"
    'Réponds uniquement en JSON : {{"requete": "..."}}\n\n'
    "Question : {question}\nRequêtes déjà essayées : {tried}"
)

CHECK_PROMPT = PromptTemplate.from_template(
    "Contexte :\n{context}\n\nRéponse :\n{answer}\n\n"
    "Chaque affirmation de la réponse est-elle appuyée par le contexte ? "
    "Ignore les phrases qui disent seulement qu'une information n'est pas couverte.\n"
    'Réponds uniquement en JSON : {{"appuye": "oui"}} ou {{"appuye": "non"}}'
)

RETRY_PROMPT = PromptTemplate.from_template(TEMPLATE.replace(
    "RÉPONSE (",
    "IMPORTANT : les extraits ci-dessus ont été jugés pertinents pour cette question. Avant de refuser, vérifie "
    "s'ils contiennent des recommandations générales qui s'appliquent au cas posé, même si le produit ou la "
    "situation n'y est pas nommé. Si oui, réponds avec ces recommandations en précisant que le guide en parle de "
    "façon générale. Ne refuse que si aucune recommandation du contexte ne s'applique.\n\nRÉPONSE (", 1))

STRICT_PROMPT = PromptTemplate.from_template(TEMPLATE.replace(
    "RÉPONSE (",
    "IMPORTANT : une première réponse contenait des affirmations absentes du contexte. "
    "N'écris que ce que le contexte dit explicitement, et signale ce qui n'est pas couvert.\n\nRÉPONSE (", 1))


class AgentState(TypedDict, total=False):
    question: str
    top_k: int
    queries: List[str]
    tried: List[str]
    hits: List[Document]
    docs: List[Document]
    relevant: bool
    rewrites: int
    raw: str
    supported: Optional[bool]
    regenerated: bool
    retried: bool
    llm_calls: int
    trace: List[str]


def parse_json(text: str, key: str, default):
    """Read one field from the model's JSON reply; fall back to a safe default."""
    try:
        return json.loads(text).get(key, default)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                return json.loads(m.group(0)).get(key, default)
            except Exception:
                pass
    return default


def _yes(value) -> bool:
    return str(value).strip().lower() in ("oui", "yes", "true", "1")


def _sections(docs: List[Document]) -> str:
    names = []
    for d in docs:
        t = (d.metadata.get("article_id") or "?").split(" - ")[0]
        if t not in names:
            names.append(t)
    return ", ".join(names)


def build_agent(c, hybrid_retrieve, json_llm):
    """c: pipelines.Components; hybrid_retrieve(question, top_k) -> reranked docs."""
    s = c.settings
    generate_chain = PROMPT | c.llm | StrOutputParser()
    strict_chain = STRICT_PROMPT | c.llm | StrOutputParser()
    retry_chain = RETRY_PROMPT | c.llm | StrOutputParser()

    def plan(st: AgentState):
        reply = (PLAN_PROMPT | json_llm).invoke({"question": st["question"]})
        subs = [q.strip() for q in parse_json(reply, "requetes", []) if isinstance(q, str) and q.strip()][:2]
        # The original question is always searched; sub-questions are added, never
        # substituted, so a poor split cannot lose the right passages.
        queries = [st["question"]] + [q for q in subs if q.lower() != st["question"].strip().lower()]
        return {"queries": queries, "tried": list(queries), "rewrites": 0,
                "llm_calls": st.get("llm_calls", 0) + 1,
                "trace": st.get("trace", []) + ["plan: search " + " | ".join(repr(q) for q in queries)]}

    def retrieve(st: AgentState):
        k = st["top_k"]
        per_query = [hybrid_retrieve(q, k) for q in st["queries"]]
        hits = rrf_fuse(per_query)[:k] if len(per_query) > 1 else per_query[0]
        # Small-to-big: keep every hit, then add neighbouring chunks (best hits
        # first) while the context stays under the size budget
        seen, chosen, size = set(), [], 0

        def add(d, force=False):
            nonlocal size
            key = doc_key(d)
            if key in seen or (not force and size + len(d.page_content) > s.agent_context_chars):
                return
            seen.add(key)
            chosen.append(d)
            size += len(d.page_content)

        for hit in hits:
            add(hit, force=True)
        for hit in hits:
            for d in c.store.neighbours(hit):
                add(d)
        chosen.sort(key=lambda d: (d.metadata.get("source_document", ""), d.metadata.get("chunk_index", 0)))
        return {"hits": hits, "docs": chosen,
                "trace": st["trace"] + [f"retrieve: {len(hits)} hits, {len(chosen)} chunks with neighbours "
                                        f"({size} chars) from {_sections(hits)}"]}

    def grade(st: AgentState):
        excerpts = "\n---\n".join(d.page_content[:400] for d in st["hits"])
        reply = (GRADE_PROMPT | json_llm).invoke({"question": st["question"], "excerpts": excerpts})
        ok = _yes(parse_json(reply, "pertinent", "oui"))
        return {"relevant": ok, "llm_calls": st["llm_calls"] + 1,
                "trace": st["trace"] + [f"grade: {'relevant' if ok else 'not relevant'}"]}

    def rewrite(st: AgentState):
        reply = (REWRITE_PROMPT | json_llm).invoke({"question": st["question"], "tried": " | ".join(st["tried"])})
        q = parse_json(reply, "requete", "") or st["question"]
        # Search the rewritten query alongside the original question
        return {"queries": [st["question"], q], "tried": st["tried"] + [q], "rewrites": st["rewrites"] + 1,
                "llm_calls": st["llm_calls"] + 1, "trace": st["trace"] + [f"rewrite: {q!r}"]}

    def generate(st: AgentState):
        raw = generate_chain.invoke({"context": format_context(st["docs"]), "question": st["question"]})
        return {"raw": raw, "llm_calls": st["llm_calls"] + 1, "trace": st["trace"] + ["generate"]}

    def retry(st: AgentState):
        raw = retry_chain.invoke({"context": format_context(st["docs"]), "question": st["question"]})
        return {"raw": raw, "retried": True, "llm_calls": st["llm_calls"] + 1,
                "trace": st["trace"] + ["retry: passages judged relevant but the answer refused"]}

    def check(st: AgentState):
        answer, _ = parse_answer(st["raw"])
        reply = (CHECK_PROMPT | json_llm).invoke({"context": format_context(st["docs"]), "answer": answer})
        ok = _yes(parse_json(reply, "appuye", "oui"))
        return {"supported": ok, "llm_calls": st["llm_calls"] + 1,
                "trace": st["trace"] + [f"check: {'supported' if ok else 'not fully supported'}"]}

    def regenerate(st: AgentState):
        raw = strict_chain.invoke({"context": format_context(st["docs"]), "question": st["question"]})
        return {"raw": raw, "regenerated": True, "llm_calls": st["llm_calls"] + 1,
                "trace": st["trace"] + ["regenerate (stricter)"]}

    def after_grade(st: AgentState):
        if st["relevant"] or st["rewrites"] >= s.agent_max_rewrites:
            return "generate"
        return "rewrite"

    def after_generate(st: AgentState):
        _, refused = parse_answer(st["raw"])
        if refused:
            # The grader said the passages are relevant but the answer refused:
            # give the model one more try before accepting the refusal.
            if st.get("relevant") and not st.get("retried"):
                return "retry"
            return END
        return "check" if s.agent_grounding_check else END

    def after_check(st: AgentState):
        return END if st["supported"] else "regenerate"

    g = StateGraph(AgentState)
    for name, fn in [("plan", plan), ("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite),
                     ("generate", generate), ("retry", retry), ("check", check), ("regenerate", regenerate)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, ["generate", "rewrite"])
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("generate", after_generate, ["check", "retry", END])
    g.add_conditional_edges("retry", after_generate, ["check", END])
    g.add_conditional_edges("check", after_check, [END, "regenerate"])
    g.add_edge("regenerate", END)
    graph = g.compile()

    def run(question: str, top_k: int) -> dict:
        out = graph.invoke({"question": question, "top_k": top_k, "llm_calls": 0, "trace": []})
        return {"raw": out.get("raw", ""), "docs": out.get("docs", []),
                "trace": out.get("trace", []), "llm_calls": out.get("llm_calls", 0)}

    return run
