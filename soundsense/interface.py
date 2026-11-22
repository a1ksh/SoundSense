"""Kazakh presentation for SoundSense. No audio/model processing lives here."""
from .common import ROOT

CSS = (ROOT / 'assets/interface.css').read_text(encoding='utf-8')
JS = (ROOT / 'assets/interface.js').read_text(encoding='utf-8')


def icon(name):
    paths = {
        'wave': '<path d="M3 10v4m4-8v12m5-16v20m5-16v12m4-8v4"/>',
        'grid': '<rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/>',
        'file': '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h5"/>',
        'clock': '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
        'upload': '<path d="M12 16V3m-5 5 5-5 5 5M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/>',
        'lock': '<rect x="4" y="10" width="16" height="11" rx="3"/><path d="M8 10V7a4 4 0 0 1 8 0v3M12 14v3"/>',
        'arrow': '<path d="M5 12h14m-6-6 6 6-6 6"/>',
        'check': '<path d="m5 12 4 4L19 6"/>',
    }
    return f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths[name]}</svg>'


def header(active='analysis'):
    links = [('analysis', '/', 'Дыбысты тану'), ('visualizations', '/visualizations', 'Дыбыс көрінісі'), ('guide', '/guide', 'Нұсқаулық')]
    parts = []
    for key, path, label in links:
        current = ' aria-current="page"' if key == active else ''
        selected = 'active' if key == active else ''
        parts.append(f'<a href="{path}" class="{selected}"{current}>{label}</a>')
    nav = ''.join(parts)
    return f'''<header class="ss-header">
      <a class="ss-brand" href="/" aria-label="SoundSense басты беті"><span class="ss-mark">{icon('wave')}</span><span><strong>SoundSense</strong><small>Әр дыбысты түсін.</small></span></a>
      <nav class="ss-nav" aria-label="Негізгі мәзір">{nav}</nav>
      <div class="ss-header-right"><span class="ss-ready"><i></i>Жүйе дайын</span><span class="ss-language">ҚАЗ</span></div>
    </header>'''


def sound_sculpture():
    """Original static SVG glass equalizer, deliberately without animation."""
    bars = []
    heights = [34, 62, 104, 164, 218, 176, 112, 72, 112, 182, 244, 190, 120]
    for i, height in enumerate(heights):
        x, y, width = 58 + i * 34, 167 - height / 2, 23
        fill = 'glassWhite' if i < 6 else 'glassBlue'
        bars.append(f'''<g filter="url(#glassShadow)">
          <path d="M{x+width} {y+7}l9 -8v{height-1}l-9 8z" fill="url(#glassSide)" stroke="#ffffff99" stroke-width="1"/>
          <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="10" fill="url(#{fill})" stroke="#ffffffee" stroke-width="1.5"/>
          <path d="M{x+5} {y+9}v{max(0,height-19)}" stroke="#ffffffc9" stroke-width="2.8" stroke-linecap="round"/>
          <path d="M{x+11} {y+2}l8 -6q11 -1 13 5l-9 8" fill="#e7faffaa" stroke="#ffffffbb"/>
        </g>''')
    return '''<svg class="ss-sculpture" viewBox="0 0 560 350" fill="none" role="img" aria-label="Шыны тәрізді көгілдір дыбыс толқыны — статикалық эквалайзер">
      <defs>
        <linearGradient id="glassWhite" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#ffffffdd"/><stop offset=".26" stop-color="#f0fcff66"/><stop offset=".65" stop-color="#a1cee4aa"/><stop offset="1" stop-color="#ffffffdd"/></linearGradient>
        <linearGradient id="glassBlue" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#f0fcff"/><stop offset=".22" stop-color="#a1e3f7aa"/><stop offset=".60" stop-color="#1995c5cc"/><stop offset="1" stop-color="#56b8df"/></linearGradient>
        <linearGradient id="glassSide"><stop stop-color="#459dc266"/><stop offset="1" stop-color="#126899cc"/></linearGradient>
        <filter id="glassShadow" x="-70%" y="-30%" width="300%" height="190%"><feDropShadow dx="8" dy="13" stdDeviation="8" flood-color="#5289a4" flood-opacity=".22"/></filter>
        <filter id="softGround"><feGaussianBlur stdDeviation="15"/></filter>
      </defs>
      <ellipse cx="295" cy="303" rx="201" ry="15" fill="#5c98b7" opacity=".22" filter="url(#softGround)"/>
    ''' + ''.join(bars) + '</svg>'


def hero(page='analysis'):
    content = {
        'analysis': ('ДЫБЫСТАР ӘЛЕМІНЕ ҚОШ КЕЛДІҢІЗ', 'Дыбыстың өз<br>тарихы бар.', 'Аудионы жүктеңіз. Қоршаған орта дыбыстарын<br>бірге танып көрейік.'),
        'visualizations': ('ТЫҢДАУДАН — ТҮСІНУГЕ', 'Дыбысты<br>көріп таныңыз.', 'Бір үзіндінің толқын пішіні мен жиілік көрінісі.<br>Соңғы талдауыңыз осы бетте сақталады.'),
        'guide': ('БӘРІ ҚАРАПАЙЫМ', 'Тыңдаңыз.<br>Талдаңыз. Таныңыз.', 'Алғашқы файлдан алғашқы нәтижеге дейін.<br>SoundSense-пен танысуға арналған қысқа нұсқаулық.'),
    }
    eyebrow, title, text = content[page]
    return f'''<section class="ss-hero {"ss-hero-compact" if page != "analysis" else ""}" aria-label="Бет туралы">
      <div class="ss-hero-copy"><p class="ss-eyebrow">{eyebrow}</p><h1>{title}</h1><p class="ss-hero-description">{text}</p></div>
      <div class="ss-hero-art">{sound_sculpture()}<div class="ss-art-caption"><span class="ss-tiny-dot"></span>Дыбыстың жаңа көрінісі<span class="ss-caption-number">01 — 10</span></div></div>
      <a class="ss-hero-help" href="{'/guide' if page != 'guide' else '/'}">{'Қалай жұмыс істейді?' if page != 'guide' else 'Талдауға оралу'}<span>↗</span></a>
    </section>'''


def mini_bars(seed=0):
    return '<div class="ss-mini-bars" aria-hidden="true">' + ''.join(f'<i style="--bar:{18 + ((i*13+seed*7) % 29)}px"></i>' for i in range(36)) + '</div>'


STATS = '<section class="ss-stats" aria-label="Талдау мүмкіндіктері">' + ''.join(
    f'''<article class="ss-stat"><div class="ss-stat-head"><span>{icon(symbol)}{title}</span><span class="ss-round-arrow">↗</span></div>
      <p class="ss-stat-description">{description}</p><div class="ss-stat-value">{value}<small>{unit}</small></div>{mini_bars(i)}<div class="ss-stat-foot">{foot}</div></article>'''
    for i, (symbol, title, description, value, unit, foot) in enumerate([
        ('grid', 'Танылатын дыбыстар', 'Күнделікті ортадағы таныс дыбыстар', '10', 'санат', 'Жаңбырдан — құстардың дауысына дейін'),
        ('file', 'Қолдау көрсетілетін файлдар', 'Телефоннан да, компьютерден де', '16', 'кеңейтім', 'WAV · MP3 · M4A · FLAC және басқалары'),
        ('clock', 'Өңделетін үзінді', 'Жазбаның алғашқы бөлігі алынады', '5', 'секунд', 'Қысқа жазба нөлмен толықтырылады'),
        ('upload', 'Файл көлемінің шегі', 'Аудио және дыбысы бар бейне', '100', 'МБ', 'Файлды таңдаңыз немесе сүйреп әкеліңіз'),
    ])
) + '</section>'

BOARD_HEADING = '<div class="ss-board-heading"><div><p class="ss-eyebrow">СІЗДІҢ ЖЕКЕ ДЫБЫС КЕҢІСТІГІҢІЗ</p><h2>Дыбысыңызды зерттеңіз</h2></div><span><i class="ss-tiny-dot"></i>Барлығы талдауға дайын</span></div>'
UPLOAD_HEADING = f'<div class="ss-section-top"><h3>{icon("upload")}Аудио жүктеу</h3><span>01</span></div>'
UPLOAD_NOTE = '<p class="ss-upload-note">WAV, MP3, M4A, AAC, FLAC, OGG, OPUS, AIFF, WMA, WEBM, MP4, CAF, AU және балама кеңейтімдер.</p>'
LOCAL_NOTE = f'<div class="ss-private-note">{icon("lock")}Аудио осы компьютерде өңделеді.</div>'
PROCESSING_STATUS = '<div class="ss-processing-status" data-processing-status role="status" aria-live="polite" hidden><span class="ss-status-indicator"></span><span data-processing-message></span></div>'
RESULT_HEADING = f'<div class="ss-section-top"><h3>{icon("wave")}Талдау нәтижесі</h3><span class="ss-soft-chip">ДЫБЫСТЫ ТАНУ</span></div>'
RESULT_CAPTION = '<p class="ss-result-caption">Ең ықтимал дыбыс санаты</p>'
RESULT_SUBTITLE = '<p class="ss-result-subtitle">Үздік үш болжам</p>'
PROBABILITY_NOTE = '<p class="ss-probability-note">Ықтималдықтар — модель болжамы. Олар нәтиженің дұрыстығына кепілдік бермейді.</p>'
VIEW_LINK = f'<a class="ss-view-link" href="/visualizations"><span>{icon("wave")}Дыбыстың графиктерін көру</span>{icon("arrow")}</a>'
LAST_ANALYSIS = '<div class="ss-last-analysis" data-last-analysis hidden><span>Соңғы талдау</span><strong data-last-label></strong><a href="/visualizations">Көрінісін ашу ↗</a></div>'


def plot_heading(kind):
    title, text = ('Дыбыстың толқын пішіні', 'Амплитуданың уақыт бойынша өзгеруі') if kind == 'wave' else ('Модельдің Mel спектрограммасы', 'График атауында қолданылған модель көрсетілген')
    return f'<div class="ss-plot-heading"><span class="ss-plot-icon">{icon("wave" if kind == "wave" else "grid")}</span><div><h3>{title}</h3><p>{text}</p></div><span class="ss-round-arrow">↗</span></div>'


VISUALIZATIONS = f'''<section class="ss-visualization-page">
  <div class="ss-board-heading"><div><p class="ss-eyebrow">БІР ДЫБЫСТЫҢ ЕКІ КӨРІНІСІ</p><h2>Дыбыстың ішіне үңіліңіз</h2></div><button type="button" class="ss-clear-button" data-clear-analysis>Нәтижені тазарту</button></div>
  <div class="ss-visual-summary" data-visual-summary hidden><span class="ss-tiny-dot"></span><span>Соңғы талдау:</span><strong data-visual-label></strong><span class="ss-soft-chip">АЛҒАШҚЫ 5 СЕКУНД</span></div>
  <div class="ss-visual-grid">
    <article class="ss-plot-card">{plot_heading('wave')}<div class="ss-plot-surface"><img data-saved-plot="wave" alt="Талданған дыбыстың толқын пішіні" hidden/><div class="ss-plot-empty" data-plot-empty="wave">{icon('wave')}<h4>Әзірге толқын пішіні жоқ</h4><p>Графикті көру үшін алдымен дыбысты талдаңыз.</p><a href="/">Дыбысты тануға өту ↗</a></div></div><p class="ss-plot-explainer">Толқынның биіктігі дыбыс амплитудасын, көлденең ось уақытты көрсетеді.</p></article>
    <article class="ss-plot-card">{plot_heading('mel')}<div class="ss-plot-surface"><img data-saved-plot="mel" alt="Талданған дыбыстың Мел-спектрограммасы" hidden/><div class="ss-plot-empty" data-plot-empty="mel">{icon('grid')}<h4>Әзірге спектрограмма жоқ</h4><p>Дыбыстың жиілік көрінісі талдаудан кейін пайда болады.</p><a href="/">Аудиофайл жүктеу ↗</a></div></div><p class="ss-plot-explainer">Ашық реңк — қуаттырақ дыбыс. Көрініс тек жүктелген нақты үзіндіден алынады.</p></article>
  </div>
  <div class="ss-note-card">{icon('lock')}<p>Бұл бетте осы қойындыдағы соңғы талдаудың графиктері көрсетіледі. Жаңа дыбысты талдау үшін «Дыбысты тану» бетіне оралыңыз.</p><a href="/">Жаңа талдау ↗</a></div>
</section>'''

GUIDE = f'''<section class="ss-guide-page"><div class="ss-board-heading"><div><p class="ss-eyebrow">БІРІНШІ НӘТИЖЕГЕ ҮШ ҚАДАМ</p><h2>SoundSense қалай жұмыс істейді?</h2></div><span>Бастау үшін арнайы білім қажет емес</span></div>
  <div class="ss-steps">
    <article class="ss-step"><div class="ss-step-top"><span>01</span>{icon('upload')}</div><h3>Аудионы жүктеңіз</h3><p>«Дыбысты тану» бетінде файлды таңдаңыз немесе оны жүктеу аймағына сүйреп әкеліңіз.</p><span class="ss-step-tag">16 кеңейтім · 100 МБ-қа дейін</span></article>
    <article class="ss-step"><div class="ss-step-top"><span>02</span>{icon('wave')}</div><h3>Дыбысты талдаңыз</h3><p>YAMNet немесе EfficientNet-B0 моделін таңдап, «Дыбысты талдау» батырмасын басыңыз. Болжанған санат пен үздік үш ықтималдықты көресіз.</p><span class="ss-step-tag">Алғашқы 5 секунд өңделеді</span></article>
    <article class="ss-step"><div class="ss-step-top"><span>03</span>{icon('grid')}</div><h3>Тыңдаңыз және зерттеңіз</h3><p>Өңделген аудионы тыңдаңыз. «Дыбыс көрінісі» бетінде толқын пішіні мен Мел-спектрограмманы салыстырыңыз.</p><a class="ss-step-tag" href="/visualizations">Дыбыс көрінісін ашу ↗</a></article>
  </div>
  <div class="ss-guide-details"><article class="ss-note-card"><span class="ss-note-icon">{icon('clock')}</span><div><h3>Екі модельді қалай салыстырамын?</h3><p>Бір аудионы алдымен YAMNet, кейін EfficientNet-B0 арқылы талдаңыз. Санаттарды, ықтималдықтарды және әр модельдің Mel графигін салыстырыңыз.</p></div></article><article class="ss-note-card"><span class="ss-note-icon">{icon('lock')}</span><div><h3>Дыбысыңыз осы компьютерде қалады</h3><p>Файл жергілікті өңделеді. Ұзақ жазбаның алғашқы бес секунды алынады, қысқа жазба нөлмен толықтырылады.</p></div></article></div>
  <div class="ss-guide-cta"><div><h3>Алғашқы дыбысыңызды танып көріңіз</h3><p>Жаңбыр, иттің үруі немесе сағаттың тықылы — неден бастаймыз?</p></div><a href="/">Дыбысты тануға өту {icon('arrow')}</a></div>
</section>'''

NOTES = '''**Танылатын дыбыстар:** иттің үруі, әтеш, жаңбыр, теңіз толқындары, оттың шытырлауы, сәбидің жылауы, түшкіру, сағат тықылы, тікұшақ және моторлы ара.

Модель тек осы **10 санатты** таниды. Басқа дыбысты қате жіктеуі мүмкін. Алғашқы **5 секунд** өңделеді. MP4 және WEBM файлдарының алғашқы аудиожолы алынады. Ең үлкен файл көлемі — **100 МБ**. Бүлінген немесе оқылмайтын файлды қайта жүктеп көріңіз.'''

BOTTOM = '<footer class="ss-bottom"><span>SoundSense <i></i> Қоршаған орта дыбыстарын тану</span><span>Тыңда. Таны. Түсін.</span></footer>'

# Cache only presentation data in this browser tab, not the uploaded audio.
# The existing prediction endpoint still returns its original five outputs.
CACHE_RESULT_JS = '''(wave, mel, label) => {
  const record = {wave: wave?.plot || null, mel: mel?.plot || null, label, time: Date.now()};
  try { sessionStorage.setItem('soundsense:analysis:two-models:v1', JSON.stringify(record)); } catch (_) {}
  window.soundSenseView?.hydrate();
  window.soundSenseView?.finish();
  return [];
}'''

START_ANALYSIS_JS = '''(path, model) => {
  window.soundSenseView?.start(model);
  return [path, model];
}'''

FAILED_ANALYSIS_JS = '''() => {
  window.soundSenseView?.failed();
  return [];
}'''
