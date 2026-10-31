import unittest
from scripts.summarize_cub_evaluation import summarize


class EvaluationSummaryTests(unittest.TestCase):
    def test_standalone_final_and_best(self):
        for name, values, expected in (
            ('final', ('0.7395', '0.7145', '0.7521'), 'final\t73.9500\t71.4500\t75.2100'),
            ('best', ('0.7093', '0.7372', '0.6954'), 'best\t70.9300\t73.7200\t69.5400'),
        ):
            with self.subTest(checkpoint=name):
                line = 'Epoch None, SS-K-Means Train ACC Unlabelled_v2: All {} | Old {} | New {}'.format(*values)
                text = '\n'.join([line.replace('v2', 'v1'), line.replace('SS-K-Means Train ACC Unlabelled', 'Test ACC'), line])
                self.assertEqual(summarize(text, name), expected)

    def test_missing_duplicate_or_invalid_metrics_fail(self):
        line = 'SS-K-Means Train ACC Unlabelled_v2: All 0.6 | Old 0.4 | New 0.7'
        for text in ('', line + '\n' + line, line + '\n' + line.replace('0.4', 'bad'),
                     line.replace('0.6', '61.0'), line.replace('0.4', '-0.4'),
                     line.replace('0.6', 'NaN'), line[:-3], line + 'oops', line + 'e3'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                summarize(text, 'best')
