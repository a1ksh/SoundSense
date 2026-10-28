"""Run separately: python app.py. Never retrains or starts during imports."""
import os
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from soundsense.infer import KAZAKH, Predictor
from soundsense.audio import SUPPORTED_EXTENSIONS
from soundsense import interface as ui
from soundsense.common import ROOT


def create_app(checkpoint=None):
    predictor = Predictor(checkpoint)
    predictors = {'CNN v1 (baseline)': predictor}
    v2_ready = (ROOT / 'models/accuracy_v2/best_v2.pt').is_file()

    def classify(path, model_name='CNN v1 (baseline)'):
        if path is None:
            raise gr.Error('Алдымен аудиофайл жүктеңіз.')
        try:
            if model_name not in ('CNN v1 (baseline)', 'Best V2'):
                raise ValueError('Белгісіз модель таңдалды.')
            if model_name == 'Best V2' and model_name not in predictors:
                from experiments.accuracy_v2.inference import V2Predictor
                predictors[model_name] = V2Predictor()
            result = predictors[model_name].predict(path)
        except (ValueError, RuntimeError, OSError) as exc:
            message = str(exc)
            if message == 'Audio must be nonempty and finite':
                message = 'Аудио бос болмауы және жарамды сандық үлгілерден тұруы тиіс.'
            raise gr.Error(f'Аудиофайлды өңдеу мүмкін болмады: {message}') from exc
        waveform, ax = plt.subplots(figsize=(8, 2.7), facecolor='#fbfaff')
        ax.set_facecolor('#fbfaff')
        ax.plot(np.arange(len(result['waveform'])) / result['sample_rate'], result['waveform'], linewidth=.6, color='#8463db')
        ax.set(xlabel='Уақыт (с)', ylabel='Амплитуда', xlim=(0, 5))
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_color('#ddd5eb')
        ax.tick_params(colors='#988ba9', labelsize=9)
        ax.xaxis.label.set_color('#8b7d9d')
        ax.yaxis.label.set_color('#8b7d9d')
        ax.grid(axis='y', color='#eae4f2', linewidth=.5)
        waveform.tight_layout()
        mel, ax = plt.subplots(figsize=(8, 2.7), facecolor='#fbfaff')
        im = ax.imshow(result['mel'], origin='lower', aspect='auto', extent=[0, 5, 0, 64], cmap='magma')
        ax.set(xlabel='Уақыт (с)', ylabel='Мел жолағы')
        ax.tick_params(colors='#988ba9', labelsize=9)
        ax.xaxis.label.set_color('#8b7d9d')
        ax.yaxis.label.set_color('#8b7d9d')
        colorbar = mel.colorbar(im, ax=ax, label='Қуат (дБ)')
        colorbar.ax.tick_params(colors='#988ba9', labelsize=8)
        colorbar.ax.yaxis.label.set_color('#8b7d9d')
        colorbar.outline.set_visible(False)
        mel.tight_layout()
        top3 = {KAZAKH.get(k, k): v for k, v in result['top3'].items()}
        plt.close(waveform)
        plt.close(mel)
        return result['kazakh_label'], top3, waveform, mel, (result['sample_rate'], result['waveform'])

    theme = gr.themes.Base(primary_hue='violet', neutral_hue='slate', font=['Segoe UI', 'Arial', 'sans-serif'])
    with gr.Blocks(title='SoundSense — Дыбысты тану', theme=theme, css=ui.CSS, js=ui.JS, analytics_enabled=False) as demo:
        gr.HTML(ui.HEADER)
        gr.HTML(ui.INTRO)
        gr.HTML(ui.STATS)
        with gr.Column(elem_id='analysis'):
            gr.HTML(ui.BOARD_HEADING)
            with gr.Row(elem_id='workspace'):
                with gr.Column(scale=4, elem_id='upload-column'):
                    gr.HTML(ui.UPLOAD_HEADING)
                    model_selector = gr.Dropdown(choices=['CNN v1 (baseline)', 'Best V2'] if v2_ready else ['CNN v1 (baseline)'],
                                                 value='Best V2' if v2_ready and checkpoint is None else 'CNN v1 (baseline)',
                                                 label='Модель', interactive=True, elem_id='model-selector')
                    audio = gr.File(type='filepath', file_types=list(SUPPORTED_EXTENSIONS),
                                    label='Аудиофайл жүктеу', show_label=False, elem_id='audio-upload')
                    gr.HTML(ui.UPLOAD_NOTE)
                    button = gr.Button('Дыбысты талдау  ↗', variant='primary', elem_id='analyze-button')
                    gr.HTML(ui.LOCAL_NOTE)
                with gr.Column(scale=6, elem_id='results'):
                    gr.HTML(ui.RESULT_HEADING)
                    gr.HTML(ui.RESULT_CAPTION)
                    label = gr.Textbox(label='Болжанған санат', show_label=False, interactive=False,
                                       placeholder='Әзірге нәтиже жоқ', container=False, elem_id='prediction-label')
                    gr.HTML(ui.RESULT_SUBTITLE)
                    top3 = gr.Label(num_top_classes=3, label='Үздік үш санат', show_label=False, elem_id='top-three')
                    preview = gr.Audio(label='Тыңдау · алғашқы 5 секунд', interactive=False, format='wav', elem_id='audio-preview')
                    gr.HTML(ui.PROBABILITY_NOTE)
        with gr.Row(elem_id='visualizations'):
            with gr.Column(elem_classes='ss-plot-card'):
                gr.HTML(ui.plot_heading('wave'))
                waveform = gr.Plot(label='Толқын пішіні', show_label=False, elem_classes='ss-plot')
            with gr.Column(elem_classes='ss-plot-card'):
                gr.HTML(ui.plot_heading('mel'))
                mel = gr.Plot(label='Мел-спектрограмма', show_label=False, elem_classes='ss-plot')
        gr.HTML(ui.GUIDE)
        with gr.Accordion('Танылатын дыбыстар және ескеретін жайттар', open=False, elem_id='info'):
            gr.Markdown(ui.NOTES)
        gr.HTML(ui.BOTTOM)
        button.click(classify, [audio, model_selector], [label, top3, waveform, mel, preview], api_name='predict', show_progress='minimal')
    return demo


if __name__ == '__main__':
    create_app().launch(server_name='127.0.0.1', share=False, inbrowser=False, max_file_size='100mb')
