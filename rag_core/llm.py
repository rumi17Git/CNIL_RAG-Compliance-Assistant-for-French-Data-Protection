"""The answering model: a local model served by Ollama."""
import requests
from langchain_ollama import OllamaLLM

# Same generation settings as the original service, so pipelines differ only in retrieval
GEN_OPTIONS = {"temperature": 0.2, "num_predict": 800, "top_p": 0.95, "stop": ["<|eot_id|>"]}


def make_llm(settings):
    """LangChain wrapper around Ollama's /api/generate, used by v1 and later."""
    return OllamaLLM(model=settings.ollama_model, base_url=settings.ollama_url,
                     temperature=settings.gen_temperature, num_predict=GEN_OPTIONS["num_predict"],
                     top_p=GEN_OPTIONS["top_p"], stop=GEN_OPTIONS["stop"], num_ctx=settings.num_ctx)


def make_json_llm(settings):
    """Same model in Ollama's JSON mode, for the agent's small decisions
    (split the question? are the passages relevant? is the answer supported?).
    Deterministic and short, so these steps stay cheap and parseable."""
    return OllamaLLM(model=settings.ollama_model, base_url=settings.ollama_url,
                     temperature=0, num_predict=200, format="json", num_ctx=settings.num_ctx)


def generate_raw(settings, prompt: str) -> str:
    """Direct HTTP call to Ollama, exactly as the original v0 service did."""
    resp = requests.post(f"{settings.ollama_url}/api/generate",
                         json={"model": settings.ollama_model, "prompt": prompt,
                               "stream": False, "options": {**GEN_OPTIONS, "temperature": settings.gen_temperature,
                                                               "num_ctx": settings.num_ctx}},
                         timeout=300)
    resp.raise_for_status()
    return resp.json().get("response", "")
