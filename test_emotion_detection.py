"""Unit tests for the EmotionDetection package."""

import json
import unittest
from unittest.mock import Mock, patch

from EmotionDetection import emotion_detector


def mock_watson_response(dominant_emotion):
    """Build a Watson-like response payload with a clear dominant emotion."""
    emotions = {
        "anger": 0.05,
        "disgust": 0.05,
        "fear": 0.05,
        "joy": 0.05,
        "sadness": 0.05,
    }
    emotions[dominant_emotion] = 0.8

    response = Mock()
    response.status_code = 200
    response.text = json.dumps(
        {"emotionPredictions": [{"emotion": emotions}]}
    )
    response.raise_for_status = Mock()
    return response


class EmotionDetectionTests(unittest.TestCase):
    """Validate dominant emotion detection results."""

    test_cases = [
        ("Je suis content que cela soit arrivé", "joy"),
        ("Je suis vraiment en colère à propos de cela", "anger"),
        ("Je me sens dégoûté rien qu'en entendant parler de cela", "disgust"),
        ("Je suis tellement triste à propos de cela", "sadness"),
        ("J'ai vraiment peur que cela arrive", "fear"),
    ]

    @patch("final_project.emotion_detection.requests.post")
    def test_dominant_emotions(self, mock_post):
        """Ensure each statement maps to the expected dominant emotion."""
        expected_keys = {
            "anger",
            "disgust",
            "fear",
            "joy",
            "sadness",
            "dominant_emotion",
        }

        for statement, expected_emotion in self.test_cases:
            mock_post.return_value = mock_watson_response(expected_emotion)

            with self.subTest(statement=statement):
                result = emotion_detector(statement)
                self.assertIsInstance(result, dict)
                self.assertEqual(set(result.keys()), expected_keys)
                self.assertEqual(result["dominant_emotion"], expected_emotion)

    @patch("final_project.emotion_detection.requests.post")
    def test_empty_input_returns_none_dictionary(self, mock_post):
        """Ensure empty-input server responses produce a None-filled payload."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        result = emotion_detector("")

        self.assertEqual(
            result,
            {
                "anger": None,
                "disgust": None,
                "fear": None,
                "joy": None,
                "sadness": None,
                "dominant_emotion": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
