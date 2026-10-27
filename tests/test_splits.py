import pandas as pd
import pytest

from soundsense.common import ROOT, read_json
from soundsense.download_data import validate_splits


def manifest():
    return pd.read_csv(ROOT / 'data/metadata/manifest.csv', dtype={'src_file': str})


def test_actual_original_folds_and_class_mapping():
    df = manifest()
    validate_splits(df)
    assert df.groupby('split').size().to_dict() == {'test': 80, 'train': 240, 'validation': 80}
    mapping = read_json(ROOT / 'artifacts/class_mapping.json')
    assert [m['index'] for m in mapping] == list(range(10))
    assert [m['esc50_target'] for m in mapping] != list(range(10))
    assert set(df.class_index) == set(range(10))


def test_source_leakage_guard_rejects_corrupted_metadata():
    df = manifest()
    train_source = df[df.split == 'train'].iloc[0].src_file
    df.loc[df[df.split == 'test'].index[0], 'src_file'] = train_source
    with pytest.raises(ValueError, match='leakage'):
        validate_splits(df)


def test_manifest_checksums_and_attributions():
    from soundsense.common import sha256
    df = manifest()
    assert len(list((ROOT / 'data/raw').glob('*.wav'))) == 400
    for row in df.itertuples():
        assert sha256(ROOT / 'data/raw' / row.filename) == row.sha256
    attributions = pd.read_csv(ROOT / 'data/metadata/attributions.csv')
    assert set(attributions.filename) == set(df.filename)
    assert attributions.attribution.notna().all()
