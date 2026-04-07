"""Utilities for Watson NLP emotion detection."""

import json

import requests


WATSON_EMOTION_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
WATSON_EMOTION_HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}


def _empty_emotion_result():
    """Return the standard empty-result payload for invalid input."""
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }


def emotion_detector(text_to_analyze):
    """Analyze text emotions using the Watson NLP Emotion Predict endpoint."""
    payload = {"raw_document": {"text": text_to_analyze}}
    try:
        response = requests.post(
            WATSON_EMOTION_URL,
            headers=WATSON_EMOTION_HEADERS,
            json=payload,
            timeout=30,
        )
        if response.status_code == 400:
            return _empty_emotion_result()
        response.raise_for_status()
    except requests.RequestException:
        return None

    response_dict = json.loads(response.text)
    emotions = response_dict["emotionPredictions"][0]["emotion"]

    formatted_response = {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
    }
    formatted_response["dominant_emotion"] = max(
        formatted_response,
        key=formatted_response.get,
    )

    return formatted_response
