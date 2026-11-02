"""Privileged consolidation evaluation; labels never enter merge/calibration APIs."""
from online_gcd.evaluator import OnlineEvaluator
from online_gcd.metrics import fragmentation


class ConsolidationEvaluator(OnlineEvaluator):
    def _probe_extras(self, labels, predictions):
        return {'active_probe_fragmentation': fragmentation(labels, predictions)}

    def _before_session_probe(self, bank, session_id):
        # Raw stream metrics already exist; no labels or predictions are passed.
        self.last_consolidation = bank.consolidate(progress=self.progress)

    def _probe(self, bank, session_id):
        raw_mapping = self.mapping
        # Probe-only canonical mapping: survivor's original mapping wins; otherwise
        # lexicographically smallest mapped retired ID. Never rematch on probes.
        canonical, conflicts = {}, 0
        for cid in sorted(raw_mapping, key=lambda cid: (cid not in bank._prototypes, cid)):
            target = bank.resolve_alias(cid)
            if target in canonical:
                conflicts += canonical[target] != raw_mapping[cid]
            else:
                canonical[target] = raw_mapping[cid]
        self.mapping = canonical
        try:
            report = super()._probe(bank, session_id)
            report['canonical_mapping_conflicts'] = conflicts
            return report
        finally:
            self.mapping = raw_mapping

    def evaluate_session(self, session_id, records, predictions_by_id, bank):
        report = super().evaluate_session(session_id, records, predictions_by_id, bank)
        stats = self.last_consolidation
        report.update(stats)
        report.update(raw_stream_fragmentation=report['session_local']['fragmentation'],
                      raw_historical_fragmentation=report['cumulative']['fragmentation'],
                      active_probe_fragmentation=report['probe']['active_probe_fragmentation'],
                      raw_stream_accuracy=report['session_local']['hungarian_accuracy'],
                      consolidated_probe_accuracy=report['probe']['fixed_mapping_accuracy'],
                      assignment_threshold=bank.threshold)
        self.last_timing['metric_computation_seconds'] = max(
            0., self.last_timing['metric_computation_seconds'] - stats['consolidation_time_seconds'])
        return report
