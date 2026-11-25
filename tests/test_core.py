import unittest
from datetime import datetime
from pathlib import Path

from chat_parser import parse_chat
from decisions import extract_decisions
from overview import get_overview
from sentiment import load_model, predict
from summary import summarize
from topics import cluster_embeddings, find_topics, topics_from_embeddings


SAMPLE = """12/05/24, 9:10 am - Asha: We should meet tomorrow
to plan the project 🙂
12/05/24, 09:15 - Ravi: Agreed, we will meet at 5
12/05/24, 09:20 - Messages and calls are end-to-end encrypted
12/06/24, 10:00 - Asha: Thanks!"""


class ParserTests(unittest.TestCase):
    def test_parses_people_and_multiline_messages(self):
        rows = parse_chat(SAMPLE)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["sender"], "Asha")
        self.assertIn("plan the project", rows[0]["message"])
        self.assertIn("🙂", rows[0]["message"])
        self.assertEqual(rows[0]["timestamp"], datetime(2024, 5, 12, 9, 10))

    def test_parses_bracket_style_timestamp(self):
        rows = parse_chat("[12/05/24, 9:10 PM] Asha: See you")
        self.assertEqual(rows[0]["message"], "See you")

    def test_skips_system_notices_and_media(self):
        text = """12/05/24, 9:10 - Messages and calls are end-to-end encrypted
12/05/24, 9:11 - Asha: <Media omitted>
12/05/24, 9:12 - Asha: Keep this 🙂"""
        rows = parse_chat(text)
        self.assertEqual(len(rows), 1)
        self.assertIn("🙂", rows[0]["message"])

    def test_sample_group_chat_parses_and_produces_topics(self):
        sample_path = Path(__file__).parent / "fixtures" / "sample_group_chat.txt"
        rows = parse_chat(sample_path.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 9)
        self.assertIn("and a first aid kit", rows[-1]["message"])
        self.assertEqual(find_topics(rows[:1]), [])
        topics = find_topics(rows)
        self.assertEqual(len(topics), 3)
        self.assertTrue(all(topic["messages"] and topic["representative_messages"]
                            and topic["participants"] for topic in topics))

    def test_multilingual_embedding_topics_and_manual_pair_evaluation(self):
        rows = [
            {"message": "Kal presentation slides finalize karte hain, Friday deadline hai", "sender": "A"},
            {"message": "We need finish the presentation before Friday.", "sender": "B"},
            {"message": "Train ticket book kar lena, station se 8 baje nikalna.", "sender": "C"},
            {"message": "Please book the railway tickets for our trip.", "sender": "A"},
            {"message": "Dinner ke liye pizza order kare?", "sender": "B"},
            {"message": "Let us order pizza and food for dinner.", "sender": "C"},
        ]
        expected = ["project", "project", "travel", "travel", "dinner", "dinner"]
        topics = find_topics(rows)
        predicted = {}
        for topic_id, topic in enumerate(topics):
            for row in topic["messages"]:
                predicted[row["message"]] = topic_id
        true_positive = predicted_positive = actual_positive = 0
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                same_prediction = (rows[left]["message"] in predicted
                                   and predicted.get(rows[left]["message"]) == predicted.get(rows[right]["message"]))
                same_label = expected[left] == expected[right]
                predicted_positive += int(same_prediction)
                actual_positive += int(same_label)
                true_positive += int(same_prediction and same_label)
        precision = true_positive / predicted_positive if predicted_positive else 0
        recall = true_positive / actual_positive if actual_positive else 0
        self.assertEqual(len(topics), 3)
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 1.0)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.rows = parse_chat(SAMPLE)

    def test_overview(self):
        overview = get_overview(self.rows)
        self.assertEqual(overview["total_messages"], 3)
        self.assertEqual(overview["participants"], 2)
        self.assertEqual(overview["most_active"], "Asha")

    def test_decision_has_support(self):
        results = extract_decisions(self.rows)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["support"]["sender"], "Asha")
        self.assertEqual(results[0]["status"], "agreed")
        self.assertIn("Ravi", results[0]["agreed_by"])

    def test_proposal_rejection_and_unanswered_are_distinguished(self):
        rows = parse_chat("""12/05/24, 9:10 - Asha: Should we meet tomorrow?
12/05/24, 9:12 - Ravi: No, tomorrow does not work
12/05/24, 9:20 - Maya: We should use the library""")
        proposals = [item for item in extract_decisions(rows) if item["type"] == "proposal"]
        self.assertEqual([item["status"] for item in proposals], ["rejected", "proposed"])
        self.assertEqual(proposals[1]["confidence"], "low")

    def test_explicit_commitment_records_owner_and_only_explicit_deadline(self):
        rows = parse_chat("12/05/24, 9:10 - Asha: I will prepare slides by Friday")
        item = extract_decisions(rows)[0]
        self.assertEqual(item["responsible"], "Asha")
        self.assertEqual(item["deadline"], "Friday")

    def test_hinglish_agreement_needs_a_proposal_and_filler_is_not_a_decision(self):
        rows = parse_chat("""12/05/24, 9:10 - Asha: Kal library mein milte hain?
12/05/24, 9:12 - Ravi: Haan, theek hai
12/05/24, 9:14 - Maya: Hmmm lol""")
        self.assertEqual(extract_decisions(rows), [])
        proposed = parse_chat("""12/05/24, 9:10 - Asha: Should we meet kal?
12/05/24, 9:12 - Ravi: Haan, theek hai""")
        results = extract_decisions(proposed)
        self.assertEqual(results[0]["status"], "agreed")
        self.assertEqual(results[0]["agreed_by"], ["Ravi"])

    def test_summary_includes_local_extractive_details(self):
        result = summarize(self.rows)
        self.assertTrue(result["text"])
        self.assertIn("Asha", result["participants"])
        self.assertEqual(result["dates"], ["2024-05-12", "2024-06-12"])

    def test_saved_sentiment_model_predicts_three_class_labels(self):
        model = load_model()
        self.assertIsNotNone(model)
        self.assertEqual(predict([], model), [])
        results = predict(["This is a great day", "I am very disappointed"], model)
        self.assertEqual(len(results), 2)
        self.assertTrue(set(results).issubset({"positive", "neutral", "negative"}))

    def test_topic_detection_handles_empty_and_duplicate_messages(self):
        self.assertEqual(find_topics([]), [])
        repeated = [{"message": "same words", "sender": "A", "timestamp": datetime(2024, 1, 1)}] * 3
        self.assertEqual(find_topics(repeated), [])

    def test_embedding_clusters_form_topic_cards_with_original_messages(self):
        rows = [
            {"message": "pizza dinner cheese tomato", "sender": "A", "timestamp": datetime(2024, 1, 1)},
            {"message": "pizza cheese dinner oven", "sender": "B", "timestamp": datetime(2024, 1, 1)},
            {"message": "train ticket station travel", "sender": "A", "timestamp": datetime(2024, 1, 2)},
            {"message": "train station platform travel", "sender": "B", "timestamp": datetime(2024, 1, 2)},
        ]
        vectors = [[1, 0], [0.99, 0.01], [0, 1], [0.01, 0.99]]
        self.assertEqual(len(set(cluster_embeddings(vectors))), 2)
        topics = topics_from_embeddings(rows, vectors)
        self.assertEqual(len(topics), 2)
        self.assertTrue(all(topic["name"] and topic["representative_messages"] and
                            topic["participants"] and topic["keywords"] for topic in topics))

    def test_duplicate_text_does_not_create_its_own_topic(self):
        duplicate_rows = [
            {"message": "we need plan project", "sender": "A"},
            {"message": "we need plan project", "sender": "B"},
            {"message": "we should plan project", "sender": "C"},
            {"message": "train tickets book now", "sender": "D"},
        ]
        vectors = [[1, 0], [1, 0], [0.99, 0.01], [0, 1]]
        topics = topics_from_embeddings(duplicate_rows, vectors)
        self.assertEqual(len(topics), 1)
        self.assertEqual(len(topics[0]["messages"]), 3)
        self.assertEqual(len(topics[0]["representative_messages"]), 2)

    def test_summary_preserves_attributed_chronology_without_inventing_outcome(self):
        rows = parse_chat("""12/05/24, 9:10 - Asha: We should meet tomorrow
12/05/24, 9:12 - Ravi: I am not sure""")
        result = summarize(rows)
        self.assertEqual([event["sender"] for event in result["timeline"]], ["Asha", "Ravi"])
        self.assertIn("No clear conclusion", result["conclusion"])


if __name__ == "__main__":
    unittest.main()
