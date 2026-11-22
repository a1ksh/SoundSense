"""Security, worker lifetime, single-model UI and console regressions."""
import argparse
import sys
import time
from types import SimpleNamespace
import numpy as np
import pytest
from soundsense.common import ROOT

def test_vulnerable_torch_is_rejected_before_deserialization(monkeypatch):
    from soundsense import checkpoints
    monkeypatch.setattr(checkpoints.torch, '__version__', '2.9.1+cu126')
    monkeypatch.setattr(checkpoints.torch, 'load', lambda *a, **kw: pytest.fail('Unsafe torch.load was called'))
    with pytest.raises(RuntimeError, match='CVE-2026-24747'):
        checkpoints.load_weights('untrusted.pt')


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    from experiments.accuracy_v2 import worker_client
    monkeypatch.setattr(worker_client, '__file__', str(tmp_path / 'worker_client.py'))
    target = tmp_path / 'yamnet_worker.py'
    target.write_text('''import base64, json, os, sys, struct
print(json.dumps({'ready': True, 'pid': os.getpid()}), flush=True)
for line in sys.stdin:
    r = json.loads(line)
    assert set(r) == {'id', 'audio'}
    value = struct.unpack('<f', base64.b64decode(r['audio'])[:4])[0]
    print(json.dumps({'id': r['id'], 'embedding': [value] * 1024, 'log_mel': [[value]*500]*64}), flush=True)
''', encoding='utf-8')
    worker = worker_client.YamnetWorker(sys.executable, timeout=5)
    yield worker, target
    worker.close()


def test_yamnet_reuses_process_and_close_terminates_it(fake_worker):
    worker, _ = fake_worker
    np.testing.assert_array_equal(worker.predict(np.ones(80000, dtype=np.float32)), np.ones(1024))
    process = worker.process
    np.testing.assert_array_equal(worker.predict(np.zeros(80000, dtype=np.float32)), np.zeros(1024))
    assert worker.process is process and process.poll() is None
    worker.close()
    assert process.poll() is not None and worker.process is None


@pytest.mark.parametrize('behavior', ['error', 'mismatch', 'timeout', 'non_object'])
def test_yamnet_protocol_failures_close_the_child(fake_worker, behavior):
    worker, target = fake_worker
    programs = {
        'error': "import json;print(json.dumps({'error':'missing tensorflow'}),flush=True)",
        'mismatch': "import json,sys;print(json.dumps({'ready':True}),flush=True);sys.stdin.readline();print(json.dumps({'id':999,'embedding':[0]*1024}),flush=True)",
        'timeout': "import time;time.sleep(10)",
        'non_object': "print('[]',flush=True)",
    }
    target.write_text(programs[behavior], encoding='utf-8')
    worker.timeout = 0.5 if behavior == 'timeout' else 5
    with pytest.raises(RuntimeError):
        worker.predict(np.zeros(80000, dtype=np.float32))
    assert worker.process is None



def test_worker_returns_actual_log_mel_without_resizing(fake_worker):
    worker,_=fake_worker
    features,mel=worker.analyze(np.ones(80000,dtype=np.float32))
    assert features.shape==(1024,) and mel.shape==(64,500)


def test_bad_log_mel_response_closes_worker(fake_worker):
    worker,target=fake_worker
    target.write_text("import json,sys;print(json.dumps({'ready':True}),flush=True);r=json.loads(sys.stdin.readline());print(json.dumps({'id':r['id'],'embedding':[0]*1024,'log_mel':[[0]]}),flush=True)",encoding='utf-8')
    with pytest.raises(RuntimeError,match='log-Mel'):
        worker.analyze(np.zeros(80000,dtype=np.float32))
    assert worker.process is None


def test_failed_yamnet_startup_closes_child_and_explains_error(monkeypatch):
    import app
    closed=[]
    class Broken:
        def __init__(self,checkpoint):
            pass
        def warmup(self):
            raise RuntimeError('TensorFlow weights unavailable')
        def close(self):
            closed.append(True)
    monkeypatch.setattr(app,'Predictor',Broken)
    with pytest.raises(RuntimeError,match='YAMNet іске қосылмады.*weights unavailable'):
        app.create_app()
    assert closed==[True]


def test_two_model_app_queue_retention_and_cleanup(tmp_path,monkeypatch):
    import app
    from gradio.route_utils import delete_files_created_by_app
    closed=[]
    fake=SimpleNamespace(close=lambda:closed.append(True))
    monkeypatch.setattr(app,'ready_yamnet',lambda checkpoint:fake)
    monkeypatch.setattr(app,'ready_efficientnet',lambda:fake)
    demo=app.create_app()
    try:
        assert set(demo.predictors)=={'YAMNet','EfficientNet-B0'}
        selector=next(c for c in demo.config['components'] if c['type']=='dropdown')
        assert selector['props']['value']=='YAMNet'
        assert selector['props']['choices']==[('YAMNet','YAMNet'),('EfficientNet-B0','EfficientNet-B0')]
        predict=next(d for d in demo.config['dependencies'] if d.get('api_name')=='predict')
        assert len(predict['inputs'])==2 and len(predict['outputs'])==5
        assert demo._queue.max_size==8 and demo.delete_cache==(300,3600)
        upload=next(b for b in demo.blocks.values() if b.__class__.__name__=='File')
        old,recent=tmp_path/'old.wav',tmp_path/'recent.wav'
        old.write_bytes(b'old')
        time.sleep(2.1)
        recent.write_bytes(b'recent')
        upload.temp_files.update({str(old),str(recent)})
        delete_files_created_by_app(demo,age=1)
        assert not old.exists() and recent.exists()
    finally:
        demo.close()
    assert closed


def test_windows_legacy_console_cannot_mask_failed_utf8_child(tmp_path, monkeypatch):
    from experiments.accuracy_v2 import check

    class LegacyConsole:
        encoding = 'cp1251'
        text = ''

        def write(self, value):
            value.encode(self.encoding)
            self.text += value

        def flush(self):
            pass

    console = LegacyConsole()
    monkeypatch.setattr(sys, 'stdout', console)
    monkeypatch.setattr(check, 'REPORT', tmp_path)
    text = '\u049a\u0430\u0442\u0435 \u04b1\u04af\u04d9\u0456'
    with pytest.raises(RuntimeError, match='exit code 7'):
        check.run_logged(['-c', f'import sys; print({text!r}); sys.exit(7)'], 'failed.log')
    assert (tmp_path / 'failed.log').read_text(encoding='utf-8').strip() == text
    assert '\\u' in console.text


@pytest.mark.parametrize('arguments,environment,expected', [([], {}, None), (['--port', '7869'], {}, 7869), ([], {'SOUNDSENSE_PORT': '7870'}, 7870)])
def test_port_configuration(arguments, environment, expected, monkeypatch):
    import app
    monkeypatch.delenv('SOUNDSENSE_PORT', raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    captured = {}
    fake = SimpleNamespace(launch=lambda **kw: captured.update(kw), close=lambda: captured.update(closed=True))
    monkeypatch.setattr(app, 'create_app', lambda: fake)
    monkeypatch.setattr(app, 'configure_console', lambda: None)
    app.main(arguments)
    assert captured['server_port'] == expected and captured['server_name'] == '127.0.0.1'
    assert captured['share'] is False and captured['closed']


@pytest.mark.parametrize('port', ['0', '65536'])
def test_invalid_port_is_rejected(port):
    from app import port_number
    with pytest.raises(argparse.ArgumentTypeError):
        port_number(port)
