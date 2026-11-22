"""Run separately: python app.py. Never retrains or starts during imports."""
import argparse
import os
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from soundsense.infer import KAZAKH, Predictor
from soundsense.efficientnet import EfficientNetPredictor
from soundsense.audio import SUPPORTED_EXTENSIONS
from soundsense import interface as ui
from soundsense.cli import configure_console

CACHE_RETENTION = (300, 3600)  # Sweep every five minutes; retain files for one hour.
QUEUE_CAPACITY = 8


class SoundSenseApp(gr.Blocks):
    def close(self, verbose=True):
        try:
            super().close(verbose=verbose)
        finally:
            for predictor in getattr(self, 'predictors', {}).values():
                if hasattr(predictor, 'close'):
                    predictor.close()


def ready_yamnet(checkpoint=None):
    predictor = Predictor(checkpoint)
    try:
        predictor.warmup()
        return predictor
    except Exception as exc:
        predictor.close()
        raise RuntimeError(f'YAMNet іске қосылмады: {exc}') from exc


def ready_efficientnet():
    predictor = EfficientNetPredictor()
    try:
        predictor.warmup()
        return predictor
    except Exception as exc:
        predictor.close()
        raise RuntimeError(f'EfficientNet-B0 іске қосылмады: {exc}') from exc


def create_app(checkpoint=None):
    torch.set_num_threads(4)
    predictor = ready_yamnet(checkpoint)
    predictors = {'YAMNet': predictor}
    try:
        predictors['EfficientNet-B0'] = ready_efficientnet()
    except Exception:
        predictor.close()
        raise

    def classify(path, model_name='YAMNet'):
        if path is None:
            raise gr.Error('Алдымен аудиофайл жүктеңіз.')
        try:
            if model_name not in predictors:
                raise ValueError('Белгісіз модель таңдалды.')
            result = predictors[model_name].predict(path)
        except (ValueError, RuntimeError, OSError) as exc:
            message = str(exc)
            if message == 'Audio must be nonempty and finite':
                message = 'Аудио бос болмауы және жарамды сандық үлгілерден тұруы тиіс.'
            raise gr.Error(f'Аудиофайлды өңдеу мүмкін болмады: {message}') from exc
        waveform, ax = plt.subplots(figsize=(8, 2.7), facecolor='#f7fbfd')
        ax.set_facecolor('#f7fbfd')
        ax.plot(np.arange(len(result['waveform'])) / result['sample_rate'], result['waveform'], linewidth=.6, color='#418fae')
        ax.set(xlabel='Уақыт (с)', ylabel='Амплитуда', xlim=(0, 5), title=model_name)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_color('#dcebf2')
        ax.tick_params(colors='#8da6b4', labelsize=9)
        ax.xaxis.label.set_color('#7594a5')
        ax.yaxis.label.set_color('#7594a5')
        ax.grid(axis='y', color='#e8f1f6', linewidth=.5)
        waveform.tight_layout()
        mel, ax = plt.subplots(figsize=(8, 2.7), facecolor='#f7fbfd')
        im = ax.imshow(result['mel'], origin='lower', aspect='auto', extent=[0, result['mel'].shape[1] * result['mel_frame_hop'], 0, 64], cmap='magma')
        ax.set(xlabel='Уақыт (с)', ylabel='Мел жолағы', title=model_name)
        ax.tick_params(colors='#8da6b4', labelsize=9)
        ax.xaxis.label.set_color('#7594a5')
        ax.yaxis.label.set_color('#7594a5')
        colorbar = mel.colorbar(im, ax=ax, label=result.get('mel_scale_label','Log Mel'))
        colorbar.ax.tick_params(colors='#8da6b4', labelsize=8)
        colorbar.ax.yaxis.label.set_color('#7594a5')
        colorbar.outline.set_visible(False)
        mel.tight_layout()
        top3 = {KAZAKH.get(k, k): v for k, v in result['top3'].items()}
        plt.close(waveform)
        plt.close(mel)
        return result['kazakh_label'], top3, waveform, mel, (result['sample_rate'], result['waveform'])

    theme = gr.themes.Base(primary_hue='sky', neutral_hue='slate', font=['Segoe UI', 'Arial', 'sans-serif'])
    with SoundSenseApp(title='SoundSense — Дыбысты тану', theme=theme, css=ui.CSS, js=ui.JS,
                       analytics_enabled=False, delete_cache=CACHE_RETENTION) as demo:
        demo.predictors = predictors
        gr.Navbar(visible=False)
        gr.HTML(ui.header('analysis'))
        gr.HTML(ui.hero('analysis'))
        gr.HTML(ui.STATS)
        gr.HTML(ui.LAST_ANALYSIS)
        with gr.Column(elem_id='analysis'):
            gr.HTML(ui.BOARD_HEADING)
            with gr.Row(elem_id='workspace'):
                with gr.Column(scale=4, elem_id='upload-column'):
                    gr.HTML(ui.UPLOAD_HEADING)
                    model_selector = gr.Dropdown(choices=list(predictors), value='YAMNet',
                        label='Модель', interactive=True, elem_id='model-selector')
                    audio = gr.File(type='filepath', file_types=list(SUPPORTED_EXTENSIONS),
                                    label='Аудиофайл жүктеу', show_label=False, elem_id='audio-upload')
                    gr.HTML(ui.UPLOAD_NOTE)
                    button = gr.Button('Дыбысты талдау  ↗', variant='primary', elem_id='analyze-button')
                    gr.HTML(ui.PROCESSING_STATUS)
                    gr.HTML(ui.LOCAL_NOTE)
                    gr.Markdown('Жүктелген және өңделген файлдар бір сағаттан кейін '
                                'кезекті тазарту кезінде жойылады (әр 5 минут сайын).')
                with gr.Column(scale=6, elem_id='results'):
                    gr.HTML(ui.RESULT_HEADING)
                    gr.HTML(ui.RESULT_CAPTION)
                    label = gr.Textbox(label='Болжанған санат', show_label=False, interactive=False,
                                       placeholder='Әзірге нәтиже жоқ', container=False, elem_id='prediction-label')
                    gr.HTML(ui.RESULT_SUBTITLE)
                    top3 = gr.Label(num_top_classes=3, label='Үздік үш санат', show_label=False, elem_id='top-three')
                    preview = gr.Audio(label='Тыңдау · алғашқы 5 секунд', interactive=False, format='wav', elem_id='audio-preview')
                    gr.HTML(ui.PROBABILITY_NOTE)
                    gr.HTML(ui.VIEW_LINK)
        # Keep five prediction outputs on the analysis page.
        # Only their rendered plot data is carried to the separate view page.
        with gr.Column(elem_id='analysis-plot-data'):
            waveform = gr.Plot(label='Толқын пішіні', show_label=False)
            mel = gr.Plot(label='Модельдің Mel спектрограммасы', show_label=False)
        gr.HTML(ui.BOTTOM)
        prediction = button.click(classify, [audio, model_selector], [label, top3, waveform, mel, preview],
                                  api_name='predict', show_progress='full',
                                  show_progress_on=[label, top3, preview], js=ui.START_ANALYSIS_JS)
        prediction.success(fn=None, inputs=[waveform, mel, label], outputs=None, js=ui.CACHE_RESULT_JS, api_name=False)
        prediction.failure(fn=None, inputs=None, outputs=None, js=ui.FAILED_ANALYSIS_JS, api_name=False)

    with demo.route('Дыбыс көрінісі', '/visualizations', show_in_navbar=False):
        gr.Navbar(visible=False)
        gr.HTML(ui.header('visualizations'))
        gr.HTML(ui.hero('visualizations'))
        gr.HTML(ui.VISUALIZATIONS)
        gr.HTML(ui.BOTTOM)

    with demo.route('Нұсқаулық', '/guide', show_in_navbar=False):
        gr.Navbar(visible=False)
        gr.HTML(ui.header('guide'))
        gr.HTML(ui.hero('guide'))
        gr.HTML(ui.GUIDE)
        with gr.Accordion('Танылатын дыбыстар және ескеретін жайттар', open=False, elem_id='info'):
            gr.Markdown(ui.NOTES)
        gr.HTML(ui.BOTTOM)
    demo.queue(max_size=QUEUE_CAPACITY, default_concurrency_limit=1)
    return demo


def port_number(value):
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError('Port must be between 1 and 65535')
    return number


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=port_number, default=os.environ.get('SOUNDSENSE_PORT'),
                        help='Use this exact port; default searches for a free port starting at 7860')
    args = parser.parse_args(argv)
    demo = create_app()
    try:
        demo.launch(server_name='127.0.0.1', server_port=args.port, share=False,
                    inbrowser=False, max_file_size='100mb')
    finally:
        demo.close()


if __name__ == '__main__':
    main()
