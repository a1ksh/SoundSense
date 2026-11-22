# Аудио форматтарын қолдау

Қолданба 16 файл кеңейтімін қабылдайды:

| Кеңейтім | Формат |
|---|---|
| `.wav`, `.wave` | WAV |
| `.mp3` | MP3 |
| `.m4a`, `.aac` | AAC, M4A контейнеріндегі қолдау көрсетілетін аудио |
| `.flac` | FLAC |
| `.ogg`, `.oga` | Ogg/Vorbis |
| `.opus` | Opus |
| `.aiff`, `.aif` | AIFF |
| `.wma` | Windows Media Audio |
| `.webm` | WebM ішіндегі аудио |
| `.mp4` | MP4 ішіндегі аудио |
| `.caf` | Core Audio Format |
| `.au` | AU |

Кеңейтімнің үлкен/кіші әріптері, қазақша атаулар және бос орындар қолдау табады. Контейнерде бірнеше аудиожол болса, алғашқысы алынады. Дыбыссыз бейне, бүлінген, шифрланған немесе қолдау көрсетілмейтін кодегі бар файл оқылмаса, түсінікті қате шығады. Кеңейтімді өзгерту файлды басқа форматқа түрлендірмейді.

Барлық форматқа ортақ әрекет: дыбысты декодтау → моно → 16 кГц → 5 секундқа қиып/толықтыру → frozen YAMNet embedding → сақталған train scaler → он санатты head. Тек алғашқы бес секунд жіктеледі. Өңделген үзіндіні интерфейстен тыңдауға болады. Көлем шегі — 100 МБ; FFmpeg декодтауына 30 секунд беріледі.

SoundFile қолдайтын шағын файлдар тікелей оқылады. Басқа форматтар мен үлкен жазбалардың бастапқы бөлігі FFmpeg арқылы уақытша float WAV-қа декодталады; уақытша файл жұмыс аяқталған соң жойылады. Бастапқы WAV өңдеуі, checkpoint, оқыту және тест нәтижелері сақталады. Lossy MP3/AAC/Opus/WMA қысуы дыбысты өзгертуі мүмкін, сондықтан WAV-пен бірдей болжамға кепілдік жоқ.

## Орнату және іске қосу

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

`imageio-ffmpeg==0.6.0` Windows wheel-і FFmpeg орындалатын файлын `.venv` ішіне орнатады. Жүйелік PATH немесе бөлек CUDA орнату қажет емес.

## Тексеру

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_audio_formats.py
```

Бұл пәрмен 16 кеңейтімнің нақты кодталған үлгілерін, mono/resampling/padding, қазақша атауларды, бүлінген файлды және decoder timeout-ты тексереді. Lossless FLAC пен WAV үшін YAMNet болжамдары салыстырылады. Оқыту қайталанбайды.

Толық жергілікті HTTP API және worker тексеруі: `python -m experiments.accuracy_v2.check`. Соңғы нәтиже [quality_checks.json](../reports/accuracy_v2/quality_checks.json), тест журналы [pytest.txt](../reports/accuracy_v2/pytest.txt) ішінде.

Декодер туралы бастапқы құжаттама: [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg), [FFmpeg](https://ffmpeg.org/ffmpeg.html).
