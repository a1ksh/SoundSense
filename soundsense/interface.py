"""Presentation assets for the Kazakh SoundSense dashboard."""
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
    }
    return f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths[name]}</svg>'


HEADER = f'''<header class="ss-header">
  <a class="ss-brand" href="#top" aria-label="SoundSense басты беті"><span class="ss-mark" style="color:#8867e5">{icon('wave')}</span><span><strong>SoundSense</strong><small>Дыбысты таны. Ортаны түсін.</small></span></a>
  <nav class="ss-nav" aria-label="Негізгі мәзір"><a class="active" href="#analysis">Дыбысты тану</a><a href="#visualizations">Дыбыс көрінісі</a><a href="#guide">Нұсқаулық</a></nav>
  <div class="ss-header-right"><span class="ss-ready"><i class="ss-dot"></i>Жүйе дайын</span><span class="ss-language" aria-label="Интерфейс тілі — қазақша">ҚАЗ</span></div>
</header>'''

INTRO = '''<div class="ss-title-row" id="top"><div><div class="ss-eyebrow">ДЫБЫСТАР ӘЛЕМІНЕ ҚОШ КЕЛДІҢІЗ</div><h1>Әр дыбыстың өз тарихы бар.</h1><p>Аудионы жүктеңіз. Қоршаған орта дыбыстарын бірге танып көрейік.</p></div><a class="ss-help-link" href="#guide">Қалай жұмыс істейді?<span aria-hidden="true">↗</span></a></div>'''

STATS = f'''<section class="ss-stats" aria-label="Талдау мүмкіндіктері">
  <article class="ss-stat"><div class="ss-stat-head">Танылатын санаттар{icon('grid')}</div><div class="ss-stat-value">10 <small>санат</small></div><div class="ss-stat-note">Күнделікті ортадағы таныс дыбыстар</div><div class="ss-category-orbs" aria-hidden="true"><span>🌧</span><span>🐕</span><span>🔥</span><span>🐓</span><span>🚁</span></div></article>
  <article class="ss-stat"><div class="ss-stat-head">Қолдау көрсетілетін кеңейтімдер{icon('file')}</div><div class="ss-stat-value">16</div><div class="ss-stat-note">Телефоннан да, компьютерден де</div><div class="ss-stat-foot"><span>WAV</span><span>MP3</span><span>FLAC</span><span class="accent">+13</span></div></article>
  <article class="ss-stat"><div class="ss-stat-head">Өңделетін үзінді{icon('clock')}</div><div class="ss-stat-value">5 <small>секунд</small></div><div class="ss-stat-note">Жазбаның алғашқы бес секунды</div><div class="ss-time-line"><span>Жүктеу</span><i></i><span>Талдау</span><i></i><span>Нәтиже</span></div></article>
  <article class="ss-stat tint"><div class="ss-stat-head">Файл көлемінің шегі{icon('upload')}</div><div class="ss-stat-value">100 <small>МБ</small></div><div class="ss-stat-note">Аудио және дыбысы бар бейне</div><div class="ss-stat-foot"><span>M4A</span><span>MP4</span><span>WEBM</span></div></article>
</section>'''

BOARD_HEADING = '<div class="ss-board-heading"><h2>Дыбысыңызды зерттеңіз</h2><span>Бір файл · бір талдау · жаңа түсінік</span></div>'
UPLOAD_HEADING = '<div class="ss-section-top"><h3>Аудио жүктеу</h3><span>01 / БІРІНШІ ҚАДАМ</span></div>'
UPLOAD_NOTE = '<p class="ss-upload-note">WAV, MP3, M4A, AAC, FLAC, OGG, OPUS, AIFF, WMA, WEBM, MP4, CAF, AU және олардың балама кеңейтімдері.</p>'
LOCAL_NOTE = f'<div class="ss-private-note">{icon("lock")}Аудио осы компьютерде өңделеді.</div>'
RESULT_HEADING = '<div class="ss-result-heading"><span>ТАЛДАУ НӘТИЖЕСІ</span><span class="ss-result-chip">02 / ДЫБЫСТЫ ТАНУ</span></div>'
RESULT_CAPTION = '<p class="ss-result-caption">Ең ықтимал дыбыс санаты</p>'
RESULT_SUBTITLE = '<p class="ss-result-subtitle">Үздік үш болжам</p>'
PROBABILITY_NOTE = '<p class="ss-probability-note">Көрсетілген ықтималдықтар — модель болжамы. Олар нәтиженің дұрыстығына кепілдік бермейді.</p>'


def plot_heading(kind):
    if kind == 'wave':
        title, text = 'Дыбыстың толқын пішіні', 'Дыбыс амплитудасының уақыт бойынша өзгеруі'
    else:
        title, text = 'Мел-спектрограмма', 'Дыбыстың уақыт пен жиілік бойынша көрінісі'
    return f'<div class="ss-plot-heading"><span class="ss-plot-icon">{icon("wave" if kind == "wave" else "grid")}</span><div><h3>{title}</h3><p>{text}</p></div></div>'


GUIDE = '''<section id="guide"><div class="ss-guide-heading"><h2>Үш қадам — бір нәтиже</h2><span>Бәрі бір бетте. Артық әрекетсіз.</span></div><div class="ss-steps"><article class="ss-step"><span>01</span><div><h3>Аудионы жүктеңіз</h3><p>Файлды таңдап немесе жүктеу аймағына сүйреп әкеліңіз.</p></div></article><article class="ss-step"><span>02</span><div><h3>Дыбысты талдаңыз</h3><p>«Дыбысты талдау» батырмасын басып, үш ықтимал санатты көріңіз.</p></div></article><article class="ss-step"><span>03</span><div><h3>Тыңдаңыз және зерттеңіз</h3><p>Өңделген үзіндіні тыңдап, дыбыстың графиктерін салыстырыңыз.</p></div></article></div></section>'''

NOTES = '''**Танылатын дыбыстар:** иттің үруі, әтеш, жаңбыр, теңіз толқындары, оттың шытырлауы, сәбидің жылауы, түшкіру, сағат тықылы, тікұшақ және моторлы ара.

Модель тек осы **10 санатты** таниды. Басқа дыбысты қате жіктеуі мүмкін. Ұзақ жазбаның алғашқы **5 секунды** алынады, қысқа жазба нөлмен толықтырылады. MP4 және WEBM файлдарының алғашқы аудиожолы өңделеді. Ең үлкен файл көлемі — **100 МБ**. Бүлінген немесе оқылмайтын файл болса, қайта жүктеп көріңіз.'''

BOTTOM = '<div class="ss-bottom"><span>SoundSense · Қоршаған орта дыбыстарын тану</span><span>Қазақша. Түсінікті. Қолжетімді.</span></div>'
