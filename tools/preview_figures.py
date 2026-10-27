"""Create a disposable contact sheet for visual review of scientific figures."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
paths = sorted((ROOT / 'results/figures').glob('*.png'))
columns, cell_w, cell_h = 4, 560, 440
sheet = Image.new('RGB', (columns * cell_w, ((len(paths) + columns - 1) // columns) * cell_h), 'white')
draw = ImageDraw.Draw(sheet)
for i, path in enumerate(paths):
    x, y = (i % columns) * cell_w, (i // columns) * cell_h
    with Image.open(path) as image:
        thumbnail = ImageOps.contain(image.convert('RGB'), (cell_w - 20, cell_h - 40))
        sheet.paste(thumbnail, (x + (cell_w - thumbnail.width) // 2, y + 24))
    draw.text((x + 10, y + 6), path.name, fill='#222222')
out = ROOT / 'results/qa/contact_sheet.jpg'
out.parent.mkdir(exist_ok=True)
sheet.save(out, quality=92)
print(out)
