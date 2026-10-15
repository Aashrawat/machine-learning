"""Local page for the NLP emotion classifier.

Run from anywhere:

    py -3 app.py

Then open http://127.0.0.1:8765

The page cleans a sentence the same way as NLP_ML.ipynb and scores it with
bag-of-words Naive Bayes, TF-IDF Naive Bayes, and TF-IDF logistic regression.
"""

from __future__ import annotations

import json
import pickle
import string
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "train.txt"
CACHE_PATH = ROOT / ".cache" / "models.pkl"
TEMPLATE_PATH = ROOT / "templates" / "index.html"
PIPELINE_VERSION = 1

# NLTK English stopwords (198). Punctuation is removed before this list is
# applied, matching the notebook, so "don't" in the list does not catch "dont".
STOPWORDS = set(
    """
a about above after again against ain all am an and any are aren aren't as at
be because been before being below between both but by can couldn couldn't d
did didn didn't do does doesn doesn't doing don don't down during each few for
from further had hadn hadn't has hasn hasn't have haven haven't having he he'd
he'll her here hers herself he's him himself his how i i'd if i'll i'm in into
is isn isn't it it'd it'll it's its itself i've just ll m ma me mightn
mightn't more most mustn mustn't my myself needn needn't no nor not now o of
off on once only or other our ours ourselves out over own re s same shan
shan't she she'd she'll she's should shouldn shouldn't should've so some such
t than that that'll the their theirs them themselves then there these they
they'd they'll they're they've this those through to too under until up ve
very was wasn wasn't we we'd we'll we're were weren weren't we've what when
where which while who whom why will with won won't wouldn wouldn't y you you'd
you'll your you're yours yourself yourselves you've
""".split()
)

NOTEBOOK_EXAMPLE = (
    "i can go from feeling so hopeless to so damned hopeful just from being "
    "around someone who cares and is awake"
)
NOTEBOOK_CLEANED = (
    "go feeling hopeless damned hopeful around someone cares awake"
)

MODELS = (
    {
        "id": "bow_nb",
        "name": "Bag of words + Naive Bayes",
        "features": "Word counts",
        "learner": "Multinomial Naive Bayes",
        "best": False,
    },
    {
        "id": "tfidf_nb",
        "name": "TF-IDF + Naive Bayes",
        "features": "TF-IDF",
        "learner": "Multinomial Naive Bayes",
        "best": False,
    },
    {
        "id": "tfidf_lr",
        "name": "TF-IDF + logistic regression",
        "features": "TF-IDF",
        "learner": "Logistic regression",
        "best": True,
    },
)


def clean_text(text: str) -> str:
    """Match the notebook: lowercase, then drop punctuation, digits, non-ASCII, stopwords."""
    lowered = text.lower()
    no_punc = lowered.translate(str.maketrans("", "", string.punctuation))
    no_digits = "".join(ch for ch in no_punc if not ch.isdigit())
    ascii_only = "".join(ch for ch in no_digits if ch.isascii())
    kept = [word for word in ascii_only.split() if word not in STOPWORDS]
    return " ".join(kept)


def cleaning_steps(text: str) -> list[dict]:
    lowered = text.lower()
    no_punc = lowered.translate(str.maketrans("", "", string.punctuation))
    no_digits = "".join(ch for ch in no_punc if not ch.isdigit())
    ascii_only = "".join(ch for ch in no_digits if ch.isascii())
    final = " ".join(word for word in ascii_only.split() if word not in STOPWORDS)
    return [
        {"name": "Lowercase", "text": lowered},
        {"name": "Drop punctuation", "text": no_punc},
        {"name": "Drop digits", "text": no_digits},
        {"name": "Drop non-ASCII", "text": ascii_only},
        {"name": "Drop stopwords", "text": final},
    ]


def _cache_token() -> str:
    stat = DATA_PATH.stat()
    return f"{PIPELINE_VERSION}:{stat.st_size}:{stat.st_mtime_ns}"


def _load_frame() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH, sep=";", header=None, names=["text", "emotion"])
    if list(frame.columns) != ["text", "emotion"]:
        raise SystemExit("train.txt did not load as text;emotion")
    return frame


def _emotion_codes(labels: pd.Series) -> dict[str, int]:
    codes: dict[str, int] = {}
    for name in labels.unique():
        codes[str(name)] = len(codes)
    return codes


def _probabilities(model, matrix, names: list[str]) -> list[dict]:
    probs = model.predict_proba(matrix)[0]
    by_class = {int(label): float(prob) for label, prob in zip(model.classes_, probs)}
    return [
        {"emotion": names[index], "probability": by_class.get(index, 0.0)}
        for index in range(len(names))
    ]


def _drivers(bundle: dict, cleaned: str, predicted_index: int) -> list[dict]:
    matrix = bundle["tfidf"].transform([cleaned])
    row = matrix.tocsr()
    if row.nnz == 0:
        return []
    names = bundle["tfidf"].get_feature_names_out()
    weights = bundle["logistic"].coef_[predicted_index]
    pairs = []
    for index, value in zip(row.indices, row.data):
        contribution = float(value * weights[index])
        pairs.append(
            {
                "token": str(names[index]),
                "weight": contribution,
                "tfidf": float(value),
            }
        )
    pairs.sort(key=lambda item: abs(item["weight"]), reverse=True)
    return pairs[:8]


def train() -> dict:
    if len(STOPWORDS) != 198:
        raise SystemExit(f"Expected 198 stopwords, found {len(STOPWORDS)}")
    if clean_text(NOTEBOOK_EXAMPLE) != NOTEBOOK_CLEANED:
        raise SystemExit("Cleaning does not match the notebook example")

    print("Reading train.txt")
    frame = _load_frame()
    codes = _emotion_codes(frame["emotion"])
    names = [name for name, _ in sorted(codes.items(), key=lambda item: item[1])]
    frame = frame.copy()
    frame["label"] = frame["emotion"].map(codes)
    print("Cleaning text")
    cleaned = frame["text"].astype(str).map(clean_text)

    x_train, x_test, y_train, y_test = train_test_split(
        cleaned, frame["label"], test_size=0.2, random_state=42
    )

    print("Fitting bag-of-words Naive Bayes")
    bow = CountVectorizer()
    x_train_bow = bow.fit_transform(x_train)
    x_test_bow = bow.transform(x_test)
    nb_bow = MultinomialNB()
    nb_bow.fit(x_train_bow, y_train)
    acc_bow = float(accuracy_score(y_test, nb_bow.predict(x_test_bow)))

    print("Fitting TF-IDF Naive Bayes")
    tfidf = TfidfVectorizer()
    x_train_tfidf = tfidf.fit_transform(x_train)
    x_test_tfidf = tfidf.transform(x_test)
    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(x_train_tfidf, y_train)
    acc_tfidf = float(accuracy_score(y_test, nb_tfidf.predict(x_test_tfidf)))

    print("Fitting TF-IDF logistic regression")
    logistic = LogisticRegression(max_iter=1000)
    logistic.fit(x_train_tfidf, y_train)
    acc_lr = float(accuracy_score(y_test, logistic.predict(x_test_tfidf)))

    counts = frame["emotion"].astype(str).value_counts()
    emotions = [
        {
            "name": name,
            "count": int(counts.get(name, 0)),
            "share": float(counts.get(name, 0) / len(frame)),
        }
        for name in names
    ]
    print(
        "Test accuracy: "
        f"bow {acc_bow:.6f}  tfidf-nb {acc_tfidf:.6f}  tfidf-lr {acc_lr:.6f}"
    )
    return {
        "token": _cache_token(),
        "names": names,
        "emotions": emotions,
        "n_rows": int(len(frame)),
        "n_train": int(len(x_train)),
        "n_test": int(len(y_test)),
        "vocabulary": int(len(bow.vocabulary_)),
        "accuracies": {"bow_nb": acc_bow, "tfidf_nb": acc_tfidf, "tfidf_lr": acc_lr},
        "bow": bow,
        "nb_bow": nb_bow,
        "tfidf": tfidf,
        "nb_tfidf": nb_tfidf,
        "logistic": logistic,
    }


def load_or_train() -> dict:
    token = _cache_token()
    if CACHE_PATH.is_file():
        try:
            with CACHE_PATH.open("rb") as handle:
                bundle = pickle.load(handle)
            if bundle.get("token") == token and "logistic" in bundle:
                print("Loaded cached models")
                return bundle
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError):
            print("Cache could not be read. Training again.")
    started = time.perf_counter()
    bundle = train()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Trained in {time.perf_counter() - started:.1f}s")
    return bundle


def meta_payload(bundle: dict) -> dict:
    models = []
    for spec in MODELS:
        models.append(
            {
                **spec,
                "accuracy": bundle["accuracies"][spec["id"]],
            }
        )
    return {
        "n_rows": bundle["n_rows"],
        "n_train": bundle["n_train"],
        "n_test": bundle["n_test"],
        "vocabulary": bundle["vocabulary"],
        "emotions": bundle["emotions"],
        "models": models,
    }


def predict_payload(bundle: dict, text: str) -> dict:
    steps = cleaning_steps(text)
    cleaned = steps[-1]["text"]
    tokens = cleaned.split()
    vocabulary = set(bundle["tfidf"].get_feature_names_out())
    missing = [token for token in tokens if token not in vocabulary]

    specs = (
        ("bow_nb", bundle["bow"], bundle["nb_bow"]),
        ("tfidf_nb", bundle["tfidf"], bundle["nb_tfidf"]),
        ("tfidf_lr", bundle["tfidf"], bundle["logistic"]),
    )
    scored = []
    predicted_index = None
    for spec_id, vectorizer, model in specs:
        matrix = vectorizer.transform([cleaned])
        index = int(model.predict(matrix)[0])
        if spec_id == "tfidf_lr":
            predicted_index = index
        spec = next(item for item in MODELS if item["id"] == spec_id)
        scored.append(
            {
                **spec,
                "accuracy": bundle["accuracies"][spec_id],
                "prediction": bundle["names"][index],
                "probabilities": _probabilities(model, matrix, bundle["names"]),
            }
        )

    names = bundle["names"]
    return {
        "original": text,
        "cleaned": cleaned,
        "tokens": tokens,
        "missing": missing,
        "no_content_words": len(tokens) == 0,
        "steps": steps,
        "models": scored,
        "drivers": _drivers(bundle, cleaned, predicted_index if predicted_index is not None else 0),
        "predicted_emotion": names[predicted_index] if predicted_index is not None else names[0],
    }


class Handler(BaseHTTPRequestHandler):
    bundle: dict = {}

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = TEMPLATE_PATH.read_bytes()
            self._send(200, body, "text/html; charset=utf-8")
            return
        if path == "/api/meta":
            self._send_json(200, meta_payload(self.bundle))
            return
        static = {
            "/static/style.css": (ROOT / "static" / "style.css", "text/css; charset=utf-8"),
            "/static/app.js": (ROOT / "static" / "app.js", "text/javascript; charset=utf-8"),
        }
        if path in static:
            file_path, content_type = static[path]
            self._send(200, file_path.read_bytes(), content_type)
            return
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/predict":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length > 100_000:
            self._send_json(400, {"error": "That request is too large."})
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            text = payload.get("text", "")
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            self._send_json(400, {"error": "Send a JSON object with a text field."})
            return
        if not isinstance(text, str):
            self._send_json(400, {"error": "Text has to be a string."})
            return
        text = text.strip()
        if not text:
            self._send_json(400, {"error": "Type a sentence first."})
            return
        if len(text) > 2000:
            self._send_json(400, {"error": "Keep the sentence under 2,000 characters."})
            return
        try:
            payload_out = predict_payload(self.bundle, text)
        except Exception:
            self._send_json(500, {"error": "Classification failed. Try a different sentence."})
            return
        self._send_json(200, payload_out)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")


def serve(bundle: dict) -> None:
    Handler.bundle = bundle
    server = None
    port = None
    for candidate in range(8765, 8775):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
            port = candidate
            break
        except OSError:
            continue
    if server is None or port is None:
        raise SystemExit("No free port between 8765 and 8774")
    url = f"http://127.0.0.1:{port}"
    print(f"Open {url}")
    print("Press Ctrl+C to stop.")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def check(bundle: dict) -> None:
    expected = {
        "bow_nb": 0.768125,
        "tfidf_nb": 0.6609375,
        "tfidf_lr": 0.8628125,
    }
    print("emotions:", ", ".join(bundle["names"]))
    print("rows", bundle["n_rows"], "train", bundle["n_train"], "test", bundle["n_test"])
    for key, target in expected.items():
        actual = bundle["accuracies"][key]
        gap = abs(actual - target)
        print(f"{key}: {actual:.6f}  notebook {target:.6f}  gap {gap:.6f}")
    sample = predict_payload(bundle, NOTEBOOK_EXAMPLE)
    print("example cleaned:", sample["cleaned"])
    print("example predictions:", ", ".join(f"{item['id']}={item['prediction']}" for item in sample["models"]))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    bundle = load_or_train()
    if "--check" in sys.argv:
        check(bundle)
        return
    serve(bundle)


if __name__ == "__main__":
    main()
