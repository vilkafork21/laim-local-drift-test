"""Проверка audited local drift без сети, с точным FAISS и заданными эмбеддингами."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('local_node', ROOT / 'main.py')
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)


class FixedEmbeddings:
    def get_embedding(self, texts):
        return np.array([[1., 0.] if text == 'good' else [0., 1.] for text in texts])


node.Config = lambda: SimpleNamespace(contour_configs={})
node.GigaEmbed = lambda **kwargs: FixedEmbeddings()


def metric():
    return {
        'contract_version': 'laim-monitoring-metric.v2', 'umr_version': 'laim-umr.v2',
        'status': 'computed', 'basket_id': 'synthetic', 'name': 'Доля корректных ответов',
        'score_column': 'main_metric', 'assessment_mode': 'qa',
        'scoring': {'method': 'identity', 'sources': [{
            'source_id': 'source_1', 'column_name': 'main_metric', 'role': 'final_score',
            'normalization': 'numeric', 'polarity': 'direct'}],
            'missing_policy': 'fail', 'majority_denominator': None},
        'aggregation': {'method': 'mean', 'weight_column': None},
        'baseline': {'value': .8, 'scale': 'ratio', 'value_source': 'validation_report',
                     'recomputed_value': .8, 'reported_value': .8, 'reported_scale': 'ratio',
                     'reconciliation': 'match'},
        'primary_validation': {'threshold': None, 'comparator': None, 'scale': 'ratio',
                               'verdict': None, 'affects_monitoring': False},
        'evidence': {},
    }


def frame(texts, scores):
    return pd.DataFrame({'session_id': [f's{i}' for i in range(len(texts))],
                         'query_id': [f'q{i}' for i in range(len(texts))],
                         'input_query': texts, 'output_answer': ['Ответ'] * len(texts),
                         'input_query_count': [1] * len(texts), 'main_metric': scores})


def run(reference, monitoring):
    return node.main(reference, monitoring, metric())['all_results']


def test_prediction_uses_queries_and_known_scale():
    ref = frame(['good'] * 80 + ['bad'] * 20, [1.] * 80 + [0.] * 20)
    results = [run(ref, frame(['good'] * 100, [score] * 100)) for score in (0., 1., None)]
    assert results[0] == results[1] == results[2]
    assert results[0]['metric_value_estimate'] == 1.
    assert results[0]['drop_estimate'] < 0
    constant = frame(['good'] * 100, [.8] * 100)
    result = run(constant, constant)
    assert np.isclose(result['metric_value_estimate'], .8)
    assert result['color'] == 'green'
    raw = metric()
    raw['baseline']['scale'] = 'raw'
    result = node.main(constant, constant, raw)['all_results']
    assert result['color'] == 'gray' and 'диапазон' in result['reason']


def test_minimum_size_and_yellow_platform_color():
    ref = frame(['good'] * 99, [1.] * 99)
    result = run(ref, frame(['good'] * 100, [None] * 100))
    assert result['color'] == 'gray'
    from llm_val.valtest_local_drift_stability import report_valtest_local_drift_stability
    stats = {'mean': .8, 'median': .8, 'q05': .8, 'share_below_threshold': 0}
    for drop, color in [(0.499, 'green'), (.5, 'yellow'), (.8, 'red')]:
        report = report_valtest_local_drift_stability({'target': 1.}, 1. - drop, stats, 'target', 'green')
        assert report['semaphore'] == color
        output = node.report_valtest_local_drift({'report': report, 'precomputed': {}}, 'Дрифт')
        assert output['all_results']['color'] == {'yellow': 'amber'}.get(color, color)


def test_dialogue_scores_and_unlabelled_monitoring():
    data = frame(['good'] * 200, [.6, 1.] * 100)
    data['reference_group_id'] = [f'd{i // 2}' for i in range(200)]
    data['turn_index'] = [i % 2 + 1 for i in range(200)]
    payload = metric()
    payload['assessment_mode'] = 'dialogue'
    output = node.main(data, data.drop(columns='main_metric'), payload)['all_results']
    assert np.isclose(output['metric_value'], .8)
    assert np.isclose(output['metric_value_estimate'], .8)


def test_long_queries_are_truncated_and_gray_is_json_serializable(monkeypatch):
    import json
    texts = []
    class Embeddings:
        def get_embedding(self, batch):
            texts.extend(batch)
            return np.ones((len(batch), 2))
    monkeypatch.setattr(node, 'GigaEmbed', lambda **kwargs: Embeddings())
    data = frame(['x' * 1200] * 100, [.8] * 100)
    assert run(data, data)['color'] == 'green'
    assert texts and max(map(len, texts)) == 1000
    result = run(data.iloc[:0], data)
    assert result['color'] == 'gray'
    json.dumps(result, allow_nan=False)


def test_reference_missing_policy_is_preserved():
    from laim_monitoring.core import _drift_frame
    payload = metric()
    payload['scoring']['missing_policy'] = 'exclude_unit'
    data = frame(['good'] * 101, [1.] * 100 + [None])
    result = _drift_frame(data, payload, require_target=True)
    assert len(result) == 100 and result['target'].eq(1.).all()


def test_real_local_scenarios():
    ref = frame(['good'] * 80 + ['bad'] * 20, [1.] * 80 + [0.] * 20)
    assert run(ref, frame(['bad'] * 100, [None] * 100))['color'] == 'red'
    ref = frame(['good'] * 80 + ['bad'] * 20, [.95] * 80 + [.2] * 20)
    assert run(ref, frame(['bad'] * 100, [None] * 100))['color'] == 'amber'
    unseen = frame(['bad'] * 100, [None] * 100)
    ref = frame(['good'] * 100, [.8] * 100)
    result = run(ref, unseen)
    assert result['color'] == 'gray' and result['share_uncovered'] == 1.
    assert np.isclose(result['metric_value'], .8)
