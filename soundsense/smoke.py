"""Short-lived local Gradio HTTP/API check; always closes its server."""
import os
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

import pandas as pd
from gradio_client import Client, handle_file

from .common import ROOT, write_json


def gradio_smoke():
    from app import create_app
    manifest = pd.read_csv(ROOT / 'data/metadata/manifest.csv')
    row = manifest[manifest.split == 'train'].iloc[0]
    path = ROOT / 'data/raw' / row.filename
    demo = create_app()
    try:
        _, url, _ = demo.launch(server_name='127.0.0.1', share=False, inbrowser=False,
                                prevent_thread_lock=True, quiet=True)
        client = Client(url, verbose=False)
        output = client.predict(handle_file(str(path)), api_name='/predict')
        if len(output) != 5 or not output[0] or not output[1] or not output[4]:
            raise RuntimeError('Unexpected Gradio prediction response')
        result = {'passed': True, 'transport': 'local HTTP and Gradio prediction API',
                  'input': row.filename, 'split': 'train', 'predicted_label': output[0],
                  'outputs': 5, 'server_closed': True, 'public_share': False}
    finally:
        demo.close()
    write_json(ROOT / 'results/gradio_smoke.json', result)
    return result


if __name__ == '__main__':
    print(gradio_smoke())
