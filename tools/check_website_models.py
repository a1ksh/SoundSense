"""Verify the running two-model website through its actual local HTTP API."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse,time
from urllib.parse import urlparse
import httpx
import numpy as np
import pandas as pd
import soundfile as sf
from gradio_client import Client, handle_file
from soundsense.common import write_json
from soundsense.labels import KAZAKH
from soundsense.cli import configure_console


def verify_site(url):
    if urlparse(url).hostname not in ['127.0.0.1','localhost']:
        raise ValueError('This verification is for the local website only')
    for route in ['', 'visualizations', 'guide']:
        response=httpx.get(url.rstrip('/')+'/'+route,timeout=20,trust_env=False)
        response.raise_for_status()
    config=httpx.get(url.rstrip('/')+'/config',timeout=20,trust_env=False).json()
    selector=next(c for c in config['components'] if c['type']=='dropdown')
    assert selector['props']['value']=='YAMNet'
    assert {c[1] for c in selector['props']['choices']}=={'YAMNet','EfficientNet-B0'}
    endpoint=next(d for d in config['dependencies'] if d.get('api_name')=='predict')
    assert len(endpoint['inputs'])==2 and len(endpoint['outputs'])==5
    example=pd.read_csv(ROOT/'colab_two_models/run/predictions/efficientnet_b0_mel_fixed.csv').iloc[0].filename
    path=ROOT/'data/raw'/example
    client=Client(url,verbose=False)
    results={}
    for name,table in [('YAMNet',ROOT/'reports/accuracy_v2/predictions/yamnet_transfer_fixed.csv'),
                       ('EfficientNet-B0',ROOT/'colab_two_models/run/predictions/efficientnet_b0_mel_fixed.csv')]:
        start=time.perf_counter()
        output=client.predict(handle_file(str(path)),name,api_name='/predict')
        elapsed=time.perf_counter()-start
        assert len(output)==5 and all(value for value in output)
        stored=pd.read_csv(table).set_index('filename').loc[example]
        assert output[0]==KAZAKH[stored.predicted_label]
        confidences=output[1]['confidences']
        assert len(confidences)==3
        ordered=sorted(KAZAKH,key=lambda label:-float(stored[f'prob_{label}']))[:3]
        assert [c['label'] for c in confidences]==[KAZAKH[label] for label in ordered]
        np.testing.assert_allclose([c['confidence'] for c in confidences],
            [stored[f'prob_{label}'] for label in ordered],rtol=2e-4,atol=2e-5)
        assert output[2]['plot'].startswith('data:image/') and output[3]['plot'].startswith('data:image/')
        audio=sf.info(output[4])
        assert audio.samplerate==16000 and audio.frames==80000 and audio.channels==1
        results[name]=dict(passed=True,label=output[0],top3_matches_saved_predictions=True,
                           waveform_and_mel_returned=True,preview_16k_five_seconds=True,
                           output_count=5,seconds=elapsed)
    default=client.predict(handle_file(str(path)),api_name='/predict')
    assert len(default)==5 and default[0]==results['YAMNet']['label']
    return dict(passed=True,url=url,models=results,model_selector_enabled=True,
                model_selector_removed=False,input_count=2,output_count=5,
                default_model='YAMNet',single_audio_default_api_compatible=True,
                all_three_routes_ok=True,input_audio=example,public_share=False)


def main():
    configure_console()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',required=True)
    parser.add_argument('--pytest-passed',type=int)
    args=parser.parse_args()
    result=verify_site(args.url)
    if args.pytest_passed is not None:
        result['pytest_passed']=args.pytest_passed
    write_json(ROOT/'reports/website_two_models/verification.json',result)
    print(result,flush=True)


if __name__=='__main__': main()
