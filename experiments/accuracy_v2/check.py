"""Run YAMNet regression tests, real inference/API checks and figure checks."""
import re
import subprocess
import sys
from PIL import Image
from soundsense.common import ROOT, read_json, sha256, write_json
from soundsense.cli import configure_console, print_console, utf8_environment
from .core import REPORT, MODELS


def run_logged(arguments,name):
    REPORT.mkdir(parents=True,exist_ok=True)
    completed=subprocess.run([sys.executable,*arguments],cwd=ROOT,text=True,
             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace',
             env=utf8_environment(),check=False,timeout=600)
    (REPORT/name).write_text(completed.stdout,encoding='utf-8')
    print_console(completed.stdout[-3500:])
    if completed.returncode:
        raise RuntimeError(f'Check failed with exit code {completed.returncode}; see {name}')
    return completed


def main():
    configure_console()
    from .run import verify_saved_runs
    from .report import generate
    verify_saved_runs()
    before={p.relative_to(ROOT).as_posix():sha256(p) for folder,pattern in [
        (MODELS,'yamnet_transfer_*.pt'),(REPORT/'training_history','yamnet_transfer_*'),
        (REPORT/'predictions','yamnet_transfer_*.csv')] for p in folder.glob(pattern)}
    generate()
    tests=run_logged(['-m','pytest','-q','-p','no:cacheprovider'],'pytest.txt')
    run_logged(['-u','-m','experiments.accuracy_v2.verify'],'verification.log')
    for name,digest in before.items():
        if sha256(ROOT/name)!=digest:
            raise ValueError(f'Verification modified a saved YAMNet artifact: {name}')
    plots=sorted((REPORT/'plots').glob('*.png'))
    assert len(plots)==16 and all(p.name.startswith('yamnet_transfer_') for p in plots)
    for path in plots:
        with Image.open(path) as im:
            assert min(im.info.get('dpi',(0,0)))>=299,path.name
            im.verify()
    assert len(list((ROOT/'results/figures').glob('*.png')))==18
    assert len(list(MODELS.glob('*.pt')))==5
    assert {p.name for p in (ROOT/'models/transfer').glob('*.pt')} == {'efficientnet_b0_fixed.pt','efficientnet_b0_backbone.pt'}
    passed=re.search(r'(\d+) passed',tests.stdout)
    skipped=re.search(r'(\d+) skipped',tests.stdout)
    verification=read_json(REPORT/'verification.json')
    fixed=read_json(REPORT/'metrics/yamnet_transfer_fixed.json')
    oof=read_json(REPORT/'metrics/yamnet_transfer_oof.json')
    result={'passed':True,'model':'yamnet_transfer','pytest_passed':int(passed.group(1)),
            'pytest_skipped':int(skipped.group(1)) if skipped else 0,'pytest_exit_code':0,
            'checkpoint_and_history_hashes_preserved':True,'all_saved_predictions_preserved':True,
            'retrained':False,'plot_count':16,'dataset_plot_count':2,'all_plot_dpi_verified':True,
            'two_model_api_verified':verification['api']['passed'], 'app_models':list(verification['api']['models']),
            'fixed_accuracy':fixed['accuracy'],'fixed_correct':fixed['correct'],
            'oof_accuracy':oof['accuracy'],'oof_correct':oof['correct']}
    write_json(REPORT/'quality_checks.json',result)
    print_console(str(result))


if __name__=='__main__':
    main()
