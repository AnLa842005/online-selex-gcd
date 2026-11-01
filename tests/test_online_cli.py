from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.validate_online_protocol import main


class OnlineCLITests(unittest.TestCase):
    def invoke(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        # Synthetic 100/100 partition; no dataset or SSB file is opened here.
        with patch('scripts.validate_online_protocol.load_cub_ssb_classes',
                   return_value=(tuple(range(100)), tuple(range(100, 200)))), patch(
                'sys.argv', ['validate_online_protocol.py', *map(str, arguments)]), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                main()
                code = 0
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_schedule_export_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'schedule.json'
            code, stdout, stderr = self.invoke('--output', output)
            self.assertEqual(code, 0, stderr)
            self.assertIn('mode=schedule_only', stdout)
            self.assertEqual(stdout.count('new classes=20'), 5)
            content = output.read_bytes()
            payload = json.loads(content)
            self.assertTrue(all(not s['sample_ids'] for s in payload['sessions']))
            code, _, stderr = self.invoke('--output', output)
            self.assertEqual(code, 2)
            self.assertIn('Protocol validation failed:', stderr)
            self.assertEqual(output.read_bytes(), content)

    def test_metadata_export_is_deterministic_and_rejects_novel_label(self):
        from online_gcd.stream import SampleRecord
        with tempfile.TemporaryDirectory() as folder:
            metadata = Path(folder) / 'metadata.json'
            records = [asdict(SampleRecord(f'image-{c * 5 + j}', c))
                       for c in range(200) for j in range(5)]
            metadata.write_text(json.dumps(records), encoding='utf-8')
            exports = [Path(folder) / name for name in ('first.json', 'second.json')]
            for output in exports:
                code, stdout, stderr = self.invoke('--metadata', metadata, '--output', output)
                self.assertEqual(code, 0, stderr)
                self.assertIn('mode=sample_stream', stdout)
            self.assertEqual(exports[0].read_bytes(), exports[1].read_bytes())
            payload = json.loads(exports[0].read_bytes())
            self.assertEqual(sum(len(s['sample_ids']) for s in payload['sessions']), 1000)
            records[-1]['is_labelled_known'] = True
            metadata.write_text(json.dumps(records), encoding='utf-8')
            rejected = Path(folder) / 'rejected.json'
            code, _, stderr = self.invoke('--metadata', metadata, '--output', rejected)
            self.assertEqual(code, 2)
            self.assertIn('novel samples cannot', stderr)
            self.assertFalse(rejected.exists())

    def test_malformed_metadata_and_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.json'
            for value in ('[', '{}', '[1]', '[{"sample_id": "x"}]'):
                with self.subTest(value=value):
                    path.write_text(value, encoding='utf-8')
                    code, _, stderr = self.invoke('--metadata', path)
                    self.assertEqual(code, 2)
                    self.assertIn('Protocol validation failed:', stderr)
            path.write_text('seed: 1', encoding='utf-8')
            code, _, stderr = self.invoke('--config', path)
            self.assertEqual(code, 2)
            self.assertIn('configuration fields', stderr)


if __name__ == '__main__':
    unittest.main()
