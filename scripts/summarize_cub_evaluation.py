"""Extract paper-comparable v2 metrics from a completed evaluation log."""
import re
import sys
from pathlib import Path


def summarize(text, checkpoint):
    metric = 'SS-K-Means Train ACC Unlabelled_v2'
    if text.count(metric) != 1:
        raise ValueError('Expected exactly one final SS-K-Means v2 result')
    fields = text.split(metric, 1)[1].partition('\n')[0].strip()
    number = r'(\d+(?:\.\d+)?)'
    match = re.fullmatch(rf': All {number} \| Old {number} \| New {number}', fields)
    if not match:
        raise ValueError('Malformed SS-K-Means v2 result')
    values = [float(x) for x in match.groups()]
    if not all(0 <= x <= 1 for x in values):
        raise ValueError('Expected fractional accuracies in [0, 1]')
    return checkpoint + '\t' + '\t'.join(f'{x * 100:.4f}' for x in values)


if __name__ == '__main__':
    print(summarize(Path(sys.argv[1]).read_text(), sys.argv[2]))
