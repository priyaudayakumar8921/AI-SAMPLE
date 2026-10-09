import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "training_data.csv"


def ensure_dataset_exists():
    if DATA_FILE.exists():
        return

    sample_data = [
        ["Spam", "Congratulations! You won a free prize"],
        ["Spam", "Claim your free cash reward now"],
        ["Spam", "Click here to win an iPhone"],
        ["Spam", "You have won a lottery"],
        ["Spam", "Win money now"],
        ["Spam", "Get a free gift today"],
        ["Spam", "Claim your reward immediately"],
        ["Spam", "Urgent! Claim your free reward"],
        ["Spam", "You won a free smartphone"],
        ["Spam", "Free offer available now"],
        ["Not Spam", "Hey, how are you?"],
        ["Not Spam", "Are we meeting tomorrow?"],
        ["Not Spam", "Please send me the notes"],
        ["Not Spam", "I will call you later"],
        ["Not Spam", "What time is the class?"],
        ["Not Spam", "See you in the evening"],
        ["Not Spam", "Can you share the assignment?"],
        ["Not Spam", "I will reach home by 6 PM"],
        ["Not Spam", "Have you completed your homework?"],
        ["Not Spam", "Let's meet tomorrow morning"],
    ]

    pd.DataFrame(sample_data, columns=["label", "message"]).to_csv(
        DATA_FILE, index=False
    )
    print(f"Created a sample training dataset at {DATA_FILE}")


def load_data():
    ensure_dataset_exists()
    data = pd.read_csv(DATA_FILE)
    data = data.dropna(subset=["label", "message"]).copy()
    data["label"] = data["label"].astype(str).str.strip()
    data["message"] = data["message"].astype(str).str.strip()
    data = data[
        data["label"].isin(["Spam", "Not Spam"]) & (data["message"] != "")
    ].copy()
    data = data.drop_duplicates(subset=["message", "label"])

    if data.empty:
        raise ValueError("Dataset is empty.")
    if data["label"].nunique() != 2:
        raise ValueError("Dataset must contain Spam and Not Spam.")
    if data["label"].value_counts().min() < 5:
        raise ValueError("At least 5 examples per class are required.")

    duplicate_conflicts = data.groupby("message")["label"].nunique()
    if (duplicate_conflicts > 1).any():
        raise ValueError("Some identical messages have conflicting labels.")

    return data


def train_best_model(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data["message"],
        data["label"],
        test_size=0.20,
        random_state=42,
        stratify=data["label"],
    )

    algorithms = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
        "Linear SVM": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, algorithm in algorithms.items():
        pipeline = make_pipeline(
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
                strip_accents="unicode",
            ),
            algorithm,
        )
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        results[name] = scores.mean()

    best_name = max(results, key=results.get)
    model = make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        ),
        algorithms[best_name],
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n===== AI SPAM MESSAGE DETECTOR =====")
    print("Dataset size:", len(data))
    print("\n===== ALGORITHM COMPARISON =====")
    for name, score in results.items():
        print(f"{name}: {score * 100:.2f}%")

    print("\nBest algorithm:", best_name)
    print(f"Cross-validation accuracy: {results[best_name] * 100:.2f}%")
    print("\n===== TEST RESULTS =====")
    print(f"Test accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["Not Spam", "Spam"],
            zero_division=0,
        )
    )

    model.fit(data["message"], data["label"])
    print("\nModel trained successfully!")
    return model


MODEL = train_best_model(load_data())


def predict_message(message):
    if not isinstance(message, str):
        raise ValueError("Message must be a string.")

    clean = message.strip()
    if not clean:
        raise ValueError("Message cannot be empty.")

    result = MODEL.predict([clean])[0]
    return result


class SpamDetectorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.serve_file("index.html", "text/html; charset=utf-8")
            return
        if path == "/styles.css":
            self.serve_file("styles.css", "text/css; charset=utf-8")
            return
        if path == "/script.js":
            self.serve_file("script.js", "application/javascript; charset=utf-8")
            return

        self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self.send_error(404, "Not found")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.send_json({"error": "Request body must be valid JSON."}, 400)
            return

        message = payload.get("message", "")
        try:
            prediction = predict_message(message)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, 400)
            return

        response = {
            "prediction": prediction,
            "is_spam": prediction == "Spam",
            "message": message,
        }
        self.send_json(response, 200)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def send_json(self, data, status_code):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, filename, content_type):
        file_path = BASE_DIR / filename
        if not file_path.exists():
            self.send_error(404, "File not found")
            return

        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), SpamDetectorHandler)
    print(f"Spam detector server running at http://127.0.0.1:{port}")
    server.serve_forever()
