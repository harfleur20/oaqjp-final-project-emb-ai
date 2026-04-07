"""Flask server for the emotion detection web application."""

import sys
from importlib import import_module
from pathlib import Path

from flask import Flask, render_template, request


INVALID_TEXT_MESSAGE = "Texte invalide ! Veuillez réessayer !."
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

emotion_detector = import_module("EmotionDetection").emotion_detector


app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)


@app.route("/")
def render_index_page():
    """Render the home page."""
    return render_template("index.html")


@app.route("/emotionDetector")
def detect_emotion():
    """Run emotion detection for the provided statement."""
    statement = request.args.get("textToAnalyze", "")

    if not statement.strip():
        return INVALID_TEXT_MESSAGE

    response = emotion_detector(statement)

    if response is None or response["dominant_emotion"] is None:
        return INVALID_TEXT_MESSAGE

    return (
        "For the given statement, the system response is "
        f"'anger': {response['anger']}, "
        f"'disgust': {response['disgust']}, "
        f"'fear': {response['fear']}, "
        f"'joy': {response['joy']} and "
        f"'sadness': {response['sadness']}. "
        f"The dominant emotion is {response['dominant_emotion']}."
    )


if __name__ == "__main__":
    app.run(host="localhost", port=5000)
