() => {
  document.documentElement.lang = 'kk';
  const root = document.querySelector('.gradio-container');
  if (!root || root.dataset.kazakhReady) return;
  root.dataset.kazakhReady = 'true';
  const cacheKey = 'soundsense:analysis:two-models:v1';
  const readAnalysis = () => {
    try {
      const record = JSON.parse(sessionStorage.getItem(cacheKey) || 'null');
      return record && typeof record.label === 'string' ? record : null;
    } catch (_) { return null; }
  };
  const hydrate = () => {
    const record = readAnalysis();
    root.querySelectorAll('[data-saved-plot]').forEach(img => {
      const kind = img.dataset.savedPlot;
      const source = record?.[kind];
      const valid = typeof source === 'string' && /^data:image\/(png|webp|jpeg|svg\+xml)[;,]/.test(source);
      if (valid && img.getAttribute('src') !== source) img.setAttribute('src', source);
      if (!valid && img.hasAttribute('src')) img.removeAttribute('src');
      img.hidden = !valid;
      const empty = root.querySelector('[data-plot-empty="' + kind + '"]');
      if (empty) empty.hidden = valid;
    });
    root.querySelectorAll('[data-visual-summary], [data-last-analysis]').forEach(el => { el.hidden = !record; });
    root.querySelectorAll('[data-visual-label], [data-last-label]').forEach(el => {
      const value = record?.label || '';
      if (el.textContent !== value) el.textContent = value;
    });
  };
  let processingTimer = null;
  const statusNodes = () => root.querySelectorAll('[data-processing-status]');
  const setStatus = (message, state) => {
    statusNodes().forEach(el => {
      el.hidden = false;
      el.dataset.state = state;
      el.setAttribute('aria-busy', state === 'busy' ? 'true' : 'false');
      const text = el.querySelector('[data-processing-message]');
      if (text) text.textContent = message;
    });
  };
  const stopTimer = () => {
    if (processingTimer !== null) clearInterval(processingTimer);
    processingTimer = null;
  };
  const start = model => {
    stopTimer();
    const started = Date.now();
    const message = () => {
      const seconds = Math.floor((Date.now() - started) / 1000);
      const name = model === 'EfficientNet-B0' ? 'EfficientNet-B0' : 'YAMNet';
      const hint = name + ' дыбысты талдап жатыр. Нәтиже сәлден кейін көрсетіледі.';
      setStatus(hint + (seconds ? ' Күту уақыты: ' + seconds + ' с.' : ''), 'busy');
    };
    message();
    processingTimer = setInterval(message, 1000);
  };
  const finish = () => {
    stopTimer();
    setStatus('Талдау аяқталды. Нәтижені тыңдап, графиктерін ашуға болады.', 'done');
  };
  const failed = () => {
    stopTimer();
    setStatus('Талдау орындалмады. Экрандағы қате хабарламасын тексеріп, қайта басыңыз.', 'error');
  };
  window.soundSenseView = { hydrate, start, finish, failed };
  root.addEventListener('click', event => {
    if (event.target.closest?.('[data-clear-analysis]')) {
      try { sessionStorage.removeItem(cacheKey); } catch (_) {}
      hydrate();
    }
  });
  // Localize Gradio's built-in controls too, regardless of browser language.
  const words = new Map(Object.entries({
    'Drop File Here': 'Файлды осында сүйреп әкеліңіз',
    'Drop files here': 'Файлды осында сүйреп әкеліңіз',
    'Click to Upload': 'Файлды таңдау', 'or': 'немесе',
    'Upload': 'Жүктеу', 'Upload file': 'Файл жүктеу',
    'Download': 'Жүктеп алу', 'Download file': 'Файлды жүктеп алу',
    'Clear': 'Тазарту', 'Clear value': 'Тазарту', 'Remove': 'Алып тастау',
    'Delete': 'Жою', 'Delete file': 'Файлды жою', 'Remove file': 'Файлды алып тастау',
    'Play': 'Ойнату', 'Pause': 'Кідірту', 'Stop': 'Тоқтату',
    'Replay': 'Қайта ойнату', 'Playback speed': 'Ойнату жылдамдығы',
    'Playback Speed': 'Ойнату жылдамдығы', 'Speed': 'Жылдамдық',
    'Loop': 'Қайталап ойнату', 'Repeat': 'Қайталау',
    'Mute': 'Дыбысты өшіру', 'Unmute': 'Дыбысты қосу', 'Volume': 'Дыбыс деңгейі',
    'Skip to start': 'Басына өту', 'Skip to end': 'Соңына өту',
    'Fullscreen': 'Толық экран', 'View fullscreen': 'Толық экранда көру',
    'Exit fullscreen': 'Толық экраннан шығу', 'Close': 'Жабу',
    'Error': 'Қате', 'Warning': 'Ескерту', 'Loading...': 'Жүктелуде…',
    'Processing...': 'Өңделуде…', 'Processing': 'Өңделуде',
    'No data': 'Әзірге дерек жоқ', 'No file selected': 'Файл таңдалмаған',
    'No audio': 'Әзірге дыбыс жоқ', 'Audio': 'Дыбыс',
    'Label': 'Санат', 'Confidence': 'Ықтималдық', 'Show more': 'Толығырақ',
    'Home': 'Дыбысты тану',
    'Search...': 'Іздеу…', 'No options': 'Нұсқа табылмады',
    'Show less': 'Жинау', 'More': 'Қосымша', 'Settings': 'Баптаулар',
    'Download plot': 'Графикті жүктеп алу', 'Share': 'Бөлісу',
    'Copy': 'Көшіру', 'Copied': 'Көшірілді', 'Retry': 'Қайталау',
    'Connection errored out.': 'Байланыс үзілді. Қайта көріңіз.',
    'Waiting...': 'Күтілуде…', 'waiting': 'Күтілуде', 'processing': 'Өңделуде',
    'Перетащите файл сюда': 'Файлды осында сүйреп әкеліңіз',
    'Нажмите для загрузки': 'Файлды таңдау', 'или': 'немесе',
    'Загрузка...': 'Жүктелуде…', 'Загрузка': 'Жүктелуде',
    'Очистить': 'Тазарту', 'Скачать': 'Жүктеп алу', 'Удалить': 'Алып тастау',
    'Воспроизвести': 'Ойнату', 'Пауза': 'Кідірту', 'Стоп': 'Тоқтату',
    'Продолжить': 'Жалғастыру', 'Ожидание': 'Күтілуде', 'Ошибка': 'Қате',
    'Аудио': 'Дыбыс', 'Метка': 'Санат', 'Пусто': 'Әзірге дерек жоқ',
    'Настройки': 'Баптаулар', 'Поделиться': 'Бөлісу', 'Отменить': 'Болдырмау',
    'Перетащите аудиофайл сюда для загрузки': 'Аудиофайлды осында сүйреп әкеліңіз',
    'Ожидание завершения загрузки файла(ов), пожалуйста, повторите попытку.': 'Файлдың жүктелуін күтіңіз де, қайта көріңіз.'
  }));
  const translate = () => {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      if (['SCRIPT', 'STYLE'].includes(node.parentElement?.tagName)) continue;
      const text = node.nodeValue.trim();
      if (words.has(text)) node.nodeValue = node.nodeValue.replace(text, words.get(text));
    }
    root.querySelectorAll('[aria-label], [title], [data-tooltip]').forEach(el => {
      ['aria-label', 'title', 'data-tooltip'].forEach(attribute => {
        const value = el.getAttribute(attribute);
        if (words.has(value)) el.setAttribute(attribute, words.get(value));
      });
    });
  };
  let scheduled = false;
  const observer = new MutationObserver(() => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      observer.disconnect(); translate();
      hydrate();
      observer.observe(root, {childList: true, subtree: true, characterData: true});
      scheduled = false;
    });
  });
  translate();
  hydrate();
  observer.observe(root, {childList: true, subtree: true, characterData: true});
}
