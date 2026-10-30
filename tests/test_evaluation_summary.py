import unittest
from scripts.summarize_cub_evaluation import summarize


class EvaluationSummaryTests(unittest.TestCase):
    def test_selects_train_v2_and_converts_percent(self):
        text = 'Test ACC_v2: All 0.1 | Old 0.2 | New 0.3\nSS-K-Means Train ACC Unlabelled_v1: All 0.9 | Old 0.9 | New 0.9\nSS-K-Means Train ACC Unlabelled_v2: All 0.615881 | Old 0.475650 | New 0.686019'
        self.assertEqual(summarize(text, 'final'), 'final\t61.5881\t47.5650\t68.6019')

    def test_missing_duplicate_or_invalid_metrics_fail(self):
        line = 'SS-K-Means Train ACC Unlabelled_v2: All 0.6 | Old 0.4 | New 0.7'
        for text in ('', line + '\n' + line, line.replace('0.6', '61.0')):
            with self.subTest(text=text), self.assertRaises(ValueError):
                summarize(text, 'best')
