"""Run separately: python app.py. Never retrains or starts during imports."""
import os
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from soundsense.infer import KAZAKH, Predictor


def create_app(checkpoint=None):
    predictor = Predictor(checkpoint)

    def classify(path):
        if path is None:
            raise gr.Error('Алдымен аудиофайл жүктеңіз.')
        try:
            result = predictor.predict(path)
        except (ValueError, RuntimeError, OSError) as exc:
            raise gr.Error(f'Аудиофайлды өңдеу мүмкін болмады: {exc}') from exc
        waveform, ax = plt.subplots(figsize=(9, 2.8))
        ax.plot(np.arange(len(result['waveform'])) / result['sample_rate'], result['waveform'], linewidth=.5, color='#16697a')
        ax.set(xlabel='Time (s)', ylabel='Amplitude', title='Uploaded audio · first five seconds (zero-padded if short)')
        waveform.tight_layout()
        mel, ax = plt.subplots(figsize=(9, 3.2))
        im = ax.imshow(result['mel'], origin='lower', aspect='auto', extent=[0, 5, 0, 64], cmap='magma')
        ax.set(xlabel='Time (s)', ylabel='Mel band', title='64-band log-Mel representation')
        mel.colorbar(im, ax=ax, label='dB (reference power = 1)')
        mel.tight_layout()
        top3 = {f'{KAZAKH.get(k, k)} ({k})': v for k, v in result['top3'].items()}
        plt.close(waveform)
        plt.close(mel)
        return f'{result["kazakh_label"]} ({result["label"]})', top3, waveform, mel

    with gr.Blocks(title='SoundSense · ESC-10') as demo:
        gr.Markdown('# SoundSense\nҚоршаған орта дыбыстарын жіктеу · ESC-10 CNN')
        gr.Markdown('Модель тек ESC-10 деректерінің 10 санатын таниды. Өзге дыбыстарды қате жіктеуі мүмкін. '
                    'Softmax ықтималдығы калибрленген сенімділік өлшемі емес. Ұзақ файлдың алғашқы 5 секунды өңделеді.')
        audio = gr.Audio(type='filepath', sources=['upload'], label='Аудиофайл (.wav және қолдау көрсетілетін форматтар)')
        button = gr.Button('Дыбысты жіктеу', variant='primary')
        label = gr.Textbox(label='Болжанған санат')
        top3 = gr.Label(num_top_classes=3, label='Үздік үш санат')
        with gr.Row():
            waveform = gr.Plot(label='Толқын пішіні')
            mel = gr.Plot(label='Mel-спектрограмма')
        button.click(classify, audio, [label, top3, waveform, mel], api_name='predict')
    return demo


if __name__ == '__main__':
    create_app().launch(server_name='127.0.0.1', share=False, inbrowser=False)
