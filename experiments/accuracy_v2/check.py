"""Execute final unit tests, inference/API smoke and immutable-baseline checks."""
import re
import subprocess
import sys

from PIL import Image

from soundsense.common import ROOT, read_json, sha256, write_json
from .core import REPORT


def run_logged(arguments, name):
    completed = subprocess.run([sys.executable, *arguments], cwd=ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='replace')
    (REPORT / name).write_text(completed.stdout, encoding='utf-8')
    print(completed.stdout[-3500:], flush=True)
    if completed.returncode:
        raise RuntimeError(f'Check failed with exit code {completed.returncode}; see {name}')
    return completed


def main():
    baseline = read_json(ROOT / 'results/summary.json')
    for name in ['cnn', 'mlp']:
        assert sha256(ROOT / f'models/{name}_best.pt') == baseline['models'][name]['training']['checkpoint_sha256']
        assert sha256(ROOT / f'results/history/{name}.csv') == baseline['models'][name]['training']['history_sha256']
    tests = run_logged(['-m', 'pytest', '-q', '-p', 'no:cacheprovider'], 'pytest.txt')
    run_logged(['-u', '-m', 'experiments.accuracy_v2.verify'], 'verification.log')
    passed = int(re.search(r'(\d+) passed', tests.stdout).group(1))
    skipped = re.search(r'(\d+) skipped', tests.stdout)
    plots = sorted((REPORT / 'plots').glob('*.png'))
    assert plots, 'No real result plots found'
    for path in plots:
        with Image.open(path) as im:
            dpi = im.info.get('dpi', (0, 0))
            assert min(dpi) >= 299, f'Wrong DPI: {path.name}'
            im.verify()
    result = {'pytest_passed': passed, 'pytest_skipped': int(skipped.group(1)) if skipped else 0,
              'pytest_exit_code': tests.returncode, 'baseline_checkpoint_and_history_hashes_preserved': True,
              'all_plot_dpi_verified': True, 'plot_count': len(plots), 'inference_api_verified': True}
    write_json(REPORT / 'quality_checks.json', result)
    from .report import generate
    generate()
    print(result, flush=True)


if __name__ == '__main__':
    main()
