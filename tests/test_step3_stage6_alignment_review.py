"""Verify the retained CellFM sampled-position limitation, not fixed gene alignment."""
import json

import joblib
import numpy as np
import pytest

from omiclass.foundation.article_head import make_head
from omiclass.foundation.bert_head import torch_reproducibility
from omiclass.foundation.extraction import TokenObserver
from omiclass.foundation.heads import TokenModel
from omiclass.foundation.io import token_record


@pytest.mark.parametrize('views', [1, 20])
def test_valid_cellfm_token_records_can_have_different_gene_identities_in_each_slot(tmp_path, views):
    # Existing CellFM width is 2048 sampled positions, not the full vocabulary.
    # These two selected sets have the same length and different membership.
    genes = np.stack([np.arange(1, 2049), np.arange(2, 2050)])
    values = np.zeros((2, 2048, 2), dtype=np.float32)
    masks = np.ones((2, 2048), dtype=bool)
    cache = tmp_path / 'tokens'
    observer = TokenObserver(cache, 'cellfm', views)
    for view in range(views):
        observer(values, masks, genes, ['a', 'b'], view)
    observer.finish(['a', 'b'])
    manifest = json.loads((cache / 'manifest.json').read_text())
    assert len(manifest['records']) == 2 * views
    with token_record(cache, manifest['records'][0]) as first:
        a = first['gene_ids'].copy()
        assert first['mask'].all()
    with token_record(cache, manifest['records'][1]) as second:
        b = second['gene_ids'].copy()
        assert second['mask'].all()
    assert np.array_equal(a, np.sort(a)) and np.array_equal(b, np.sort(b))
    assert len(a) == len(b) == 2048
    assert a[0] != b[0]  # Sorting has not made slot 0 represent the same gene.
    assert len(np.union1d(a, b)) == 2049  # Unique gene slots cannot fit in 2048 positions.


def positional_head():
    """Small input shapes, same existing LayerNorm/projection/flatten/[64] head."""
    import torch
    model = make_head(width=2, token_count=2, hidden_dims=[64], token_dim=8,
                      dropout=0.0, n_classes=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.norm.weight.fill_(1.0)
        model.projection.weight[0, 0] = 1.0
        model.mlp[0].weight[0, 0] = 1.0
        model.mlp[0].weight[1, 8] = 1.0
        model.mlp[-1].weight[0, 0] = 1.0
        model.mlp[-1].weight[1, 1] = 1.0
    return model.eval()


def test_existing_flattened_head_is_sensitive_to_token_order():
    import torch
    with torch_reproducibility(80, 1), torch.inference_mode():
        model = positional_head()
        values = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])
        mask = torch.ones((1, 2), dtype=torch.bool)
        original = model(values, mask).softmax(-1)
        reordered = model(values.flip(1), mask).softmax(-1)
    assert original.argmax(1).item() == 0
    assert reordered.argmax(1).item() == 1
    assert not torch.equal(original, reordered)


def test_historical_token_model_preserves_sampled_position_predictions(tmp_path):
    # Varying valid membership remains allowed; the head consumes tokens/masks,
    # without assigning a fixed gene identity to each flattened position.
    with torch_reproducibility(80, 1):
        head = positional_head()
        state = {key: value.detach().numpy().copy() for key, value in head.state_dict().items()}
    model = TokenModel(state, dict(family='article', width=2, token_count=2,
        hidden_dims=[64], token_dim=8, dropout=0.0), ['a', 'b'])
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    before = path.read_bytes()
    saved = joblib.load(path)
    values = np.array([[[1.0, -1.0], [-1.0, 1.0]]], dtype=np.float32)
    mask = np.ones((1, 2), dtype=bool)
    caches = []
    for index, genes in enumerate([[[1, 2]], [[2, 3]]]):
        cache = tmp_path / f'tokens{index}'
        observer = TokenObserver(cache, 'cellfm', 1)
        observer(values, mask, np.asarray(genes), ['sample'], 0)
        observer.finish(['sample'])
        caches.append(cache)
    np.testing.assert_array_equal(saved.predict_proba(caches[0], ['sample']),
                                  saved.predict_proba(caches[1], ['sample']))
    assert path.read_bytes() == before
