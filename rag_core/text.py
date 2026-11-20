"""Text helpers: French tokenisation for BM25 and parsing the model's reply."""
import re
import unicodedata

REFUSAL_TAG = "[HORS_CORPUS]"
REFUSAL_MESSAGE = "Je ne trouve pas cette information dans les documents fournis."

# Small French stopword list: enough to stop BM25 from matching on "de", "la", "les"...
STOPWORDS = set("""
a à au aux avec ce ces cet cette dans de des du elle en et eux il ils je la le les leur leurs lui ma
mais me même mes moi mon ne nos notre nous on ou où par pas pour qu que qui sa se ses son sur ta te
tes toi ton tu un une vos votre vous y d l j m n s t c est sont été être avoir ai as avons avez ont
plus tout tous toute toutes comme si sans sous entre aussi ainsi doit doivent peut peuvent faut
quel quelle quels quelles quoi comment est-ce
""".split())


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def french_tokenize(text: str) -> list:
    """Lowercase, drop accents and punctuation, remove stopwords.

    Accents are dropped so "securite" in a question still matches "sécurité".
    """
    text = _strip_accents(text.lower().replace("’", "'"))
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def parse_answer(raw: str):
    """Return (answer, refused) from the model's raw reply.

    A full refusal starts with the tag. If the tag only appears later, the model
    answered part of the question and noted what was missing: a partial answer.
    An 8B model sometimes writes the refusal sentence but forgets the tag, so a
    short reply that opens with the refusal sentence also counts as a refusal.
    """
    raw = (raw or "").strip()
    refused = raw.startswith(REFUSAL_TAG)
    answer = raw.replace(REFUSAL_TAG, "").strip()
    if answer.lower().startswith(REFUSAL_MESSAGE[:30].lower()) and len(answer) < 200:
        refused = True
    if refused and not answer:
        answer = REFUSAL_MESSAGE
    return answer, refused
