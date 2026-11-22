"""Recompute saved YAMNet predictions and verify the real local HTTP API."""
import time
import numpy as np
import pandas as pd
import torch
from soundsense.common import ROOT, read_json, sha256, write_json
from .core import CACHE, MODELS, REPORT, load_candidate, manifest, split_masks
from .inference import YAMNetPredictor
from .run import verify_saved_runs


def verify_predictions():
    verify_saved_runs()
    df=manifest()
    cache=CACHE/'yamnet_transfer_embeddings.npz'
    if sha256(cache)!=read_json(cache.with_suffix('.json'))['sha256']:
        raise ValueError('YAMNet embedding cache checksum mismatch')
    with np.load(cache,allow_pickle=False) as z:
        if z['filenames'].tolist()!=df.filename.tolist():
            raise ValueError('YAMNet embedding cache order mismatch')
        features=z['embeddings'].copy()
    if features.shape!=(400,1024) or not np.isfinite(features).all():
        raise ValueError('Invalid saved YAMNet embedding cache')
    rows,parts=[],[]
    for fold in range(1,6):
        run=f'yamnet_transfer_fold{fold}' if fold!=5 else 'yamnet_transfer_fixed'
        model,payload=load_candidate(MODELS/f'{run}.pt')
        train,_,test=split_masks(df,fold)
        mean=features[train].mean(axis=0,dtype=np.float64)
        std=np.maximum(features[train].std(axis=0,dtype=np.float64),1e-6)
        np.testing.assert_allclose(payload['mean'],mean,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(payload['std'],std,rtol=1e-12,atol=1e-12)
        if fold in payload['train_folds'] or payload['validation_fold'] in payload['train_folds']:
            raise ValueError('YAMNet fold leakage in checkpoint')
        x=((features[test]-mean)/std).astype(np.float32)
        with torch.inference_mode():
            expected=model(torch.from_numpy(x)).softmax(1).numpy()
        path=REPORT/'predictions'/f'{run}.csv'
        receipt=read_json(path.with_suffix('.json'))
        if receipt['checkpoint_sha256']!=sha256(MODELS/f'{run}.pt') or receipt['csv_sha256']!=sha256(path):
            raise ValueError('YAMNet prediction receipt mismatch')
        frame=pd.read_csv(path)
        np.testing.assert_array_equal(frame.filename.to_numpy(),df.loc[test,'filename'].to_numpy())
        np.testing.assert_array_equal(frame.true_index.to_numpy(),df.loc[test,'class_index'].to_numpy())
        labels=[r['label'] for r in payload['class_mapping']]
        stored=frame[[f'prob_{label}' for label in labels]].to_numpy()
        np.testing.assert_allclose(stored,expected,rtol=2e-4,atol=2e-5)
        np.testing.assert_array_equal(frame.predicted_index,expected.argmax(1))
        parts.append(frame)
        rows.append({'fold':fold,'count':len(frame),'correct':int((frame.true_index==frame.predicted_index).sum()),
                     'checkpoint_sha256':sha256(MODELS/f'{run}.pt'),'max_probability_difference':float(np.max(np.abs(stored-expected)))})
    oof=pd.read_csv(REPORT/'predictions/yamnet_transfer_oof.csv').sort_values('filename').reset_index(drop=True)
    reconstructed=pd.concat(parts,ignore_index=True).sort_values('filename').reset_index(drop=True)
    # Fold CSVs serialize float32 scores with fewer digits than the merged OOF
    # CSV. Preserve both original files and allow only float32 serialization error.
    pd.testing.assert_frame_equal(oof,reconstructed,check_exact=False,rtol=5e-7,atol=1e-8)
    return {'passed':True,'oof_count':len(oof),'oof_correct':int((oof.true_index==oof.predicted_index).sum()),
            'train_only_scalers_verified':5,'folds':rows},features,df


def main():
    torch.set_num_threads(4)
    saved,features,df=verify_predictions()
    index=int(df.index[df.fold==1][0])
    path=ROOT/'data/raw'/df.iloc[index].filename
    start=time.perf_counter()
    with YAMNetPredictor() as predictor:
        predictor.warmup()
        startup=time.perf_counter()-start
        process=predictor.worker.process
        start=time.perf_counter()
        first=predictor.predict(path)
        elapsed=time.perf_counter()-start
        second=predictor.predict(path)
        p=np.asarray(list(first['probabilities'].values()))
        np.testing.assert_allclose(p,list(second['probabilities'].values()),rtol=0,atol=0)
        x=((features[index]-np.asarray(predictor.metadata['mean']))/np.asarray(predictor.metadata['std'])).astype(np.float32)
        with torch.inference_mode():
            expected=predictor.model(torch.from_numpy(x[None])).softmax(1)[0].numpy()
        np.testing.assert_allclose(p,expected,rtol=2e-4,atol=2e-5)
        assert first['waveform'].shape==(80000,)
        assert first['mel'].shape[0]==64 and first['mel'].shape[1]>128
        assert np.isfinite(first['mel']).all() and first['model']=='yamnet_transfer'
        assert len(first['top3'])==3 and predictor.worker.process is process
    assert process.poll() is not None
    live={'passed':True,'input':path.name,'split':'training fold 1','repeat_deterministic':True,
          'training_app_probability_equivalence':True,'worker_reused':True,'worker_closed':True,
          'startup_seconds':startup,'warm_prediction_seconds':elapsed,'mel_shape':list(first['mel'].shape),
          'mel_source':'official YAMNet log-Mel output; natural logarithm',
          'latency_note':'one training clip on this computer; not a benchmark'}
    write_json(REPORT/'inference_verification.json',{'yamnet_transfer':live})
    from app import create_app
    from gradio_client import Client, handle_file
    demo=create_app()
    process=demo.predictors['YAMNet'].worker.process
    try:
        _,url,_=demo.launch(server_name='127.0.0.1',share=False,inbrowser=False,
                          prevent_thread_lock=True,quiet=True,max_file_size='100mb')
        from tools.check_website_models import verify_site
        api=verify_site(url)
        assert demo.predictors['YAMNet'].worker.process is process
    finally:
        demo.close()
    assert process.poll() is not None
    api.update(server_closed=True,worker_closed=True)
    write_json(REPORT/'app_smoke.json',api)
    write_json(REPORT/'verification.json',{'passed':True,'saved_predictions':saved,'live':live,'api':api})
    print(f'PASS: YAMNet 400 OOF predictions, live worker and two-model HTTP API; {saved["oof_correct"]}/400 correct',flush=True)


if __name__=='__main__':
    main()
