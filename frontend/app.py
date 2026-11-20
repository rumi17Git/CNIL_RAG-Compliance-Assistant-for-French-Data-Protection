import os

import streamlit as st
import requests

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://generation:8002/query")
DEFAULT_TOP_K = 6

st.set_page_config(
    page_title="Assistant CNIL",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Header block */
        .app-header {
            padding: 1.25rem 1.5rem;
            border-radius: 12px;
            background: linear-gradient(135deg, #0f3d5c 0%, #14567f 100%);
            color: white;
            margin-bottom: 1.5rem;
        }
        .app-header h1 {
            color: white;
            font-size: 1.6rem;
            margin-bottom: 0.25rem;
        }
        .app-header p {
            color: #d8e6ef;
            margin: 0;
            font-size: 0.95rem;
        }

        /* Source cards */
        .source-tag {
            display: inline-block;
            background-color: #e8f0f7;
            color: #14567f;
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 6px;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #0f3d5c;
        }
        section[data-testid="stSidebar"] * {
            color: #f0f4f8 !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.15);
        }

        /* Chat bubbles tightened */
        .stChatMessage {
            padding-top: 0.4rem;
            padding-bottom: 0.4rem;
        }

        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📘 À propos")
    st.markdown(
        "Cet assistant répond exclusivement à partir du guide CNIL "
        "*« Sécurité des données personnelles »* (2024). Chaque réponse "
        "est appuyée par des citations extraites du document source."
    )

    st.markdown("---")
    st.markdown("### ⚙️ Paramètres")
    top_k = st.slider("Nombre de sources consultées", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    pipeline = st.selectbox(
        "Méthode de recherche",
        ["v0", "v1", "v2", "v2-faiss", "v3"],
        index=0,
        help="v0 : recherche vectorielle (code d'origine) · v1 : même méthode avec LangChain · "
             "v2 : hybride mots-clés (BM25) + vecteurs · v2-faiss : hybride avec un index FAISS · "
             "v3 : agent LangGraph (reformule, vérifie, lit le contexte autour)",
    )

    st.markdown("---")
    st.markdown("### 📊 Fiabilité du système")
    # Update these after each benchmark run (evaluation/results/)
    col1, col2 = st.columns(2)
    col1.metric("Précision du contexte", "0.97")
    col2.metric("Fidélité", "0.75")
    st.metric("Refus corrects (hors corpus)", "100 %")
    st.caption("Scores RAGAS sur un benchmark de 40 questions (version à découpage par page).")

    st.markdown("---")
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>📘 Assistant CNIL — Sécurité des données personnelles</h1>
        <p>Posez une question en français. Les réponses sont strictement basées sur le guide CNIL 2024, avec citations à l'appui.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


def render_citations(citations):
    if not citations:
        return
    st.markdown("**📚 Sources**")
    for j, cit in enumerate(citations, 1):
        source = cit.get("source_document", "inconnu")
        page = cit.get("page_number", "?")
        article = cit.get("article_id", "")
        snippet = cit.get("text", "")
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."

        label = f"Source {j} — {source} (p.{page})"
        if article:
            label += f" · {article}"

        with st.expander(label):
            st.markdown(f'<span class="source-tag">{source}</span>', unsafe_allow_html=True)
            st.write(snippet)


# ----------------------------------------------------------------------------
# Conversation history
# ----------------------------------------------------------------------------
if not st.session_state.history:
    st.info("Aucune question posée pour le moment. Essayez par exemple : *« Quelles sont les précautions pour les mots de passe ? »*")

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant", avatar="📘"):
        st.markdown(entry["answer"])
        render_citations(entry["citations"])

# ----------------------------------------------------------------------------
# Chat input
# ----------------------------------------------------------------------------
question = st.chat_input("Posez votre question sur la sécurité des données personnelles...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📘"):
        with st.spinner("Recherche dans le guide CNIL..."):
            try:
                payload = {"question": question, "top_k": top_k, "pipeline": pipeline}
                resp = requests.post(API_URL, json=payload, timeout=1200)

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    citations = data.get("citations", [])

                    st.markdown(answer)
                    st.caption(f"Méthode {data.get('pipeline', pipeline)} · {data.get('latency_ms', 0) / 1000:.1f} s"
                               f" · {data.get('llm_calls', 1)} appel(s) au modèle")
                    if data.get("trace"):
                        with st.expander("Raisonnement de l'agent"):
                            for step in data["trace"]:
                                st.markdown(f"- {step}")
                    render_citations(citations)

                    st.session_state.history.append(
                        {"question": question, "answer": answer, "citations": citations}
                    )
                else:
                    st.error(f"Erreur {resp.status_code} : {resp.text}")

            except requests.exceptions.Timeout:
                st.error("Le service met trop de temps à répondre. Réessayez dans quelques instants.")
            except requests.exceptions.ConnectionError:
                st.error("Impossible de contacter le service de génération. Vérifiez qu'il est bien démarré.")
            except Exception as e:
                st.error(f"Erreur inattendue : {str(e)}")

st.markdown(
    """
    <div style="text-align:center; color:#999; font-size:0.8rem; margin-top:2rem;">
        💡 Réponses générées à partir du guide CNIL « Sécurité des données personnelles » (2024).
        L'assistant ne répond qu'à partir de ce document.
    </div>
    """,
    unsafe_allow_html=True,
)