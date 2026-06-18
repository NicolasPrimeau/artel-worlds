# Capture a world's opening scene into a looping animated-WebP hub thumbnail.
#   uv pip install playwright   (dev-only; not a project dependency)
#   run the world locally, then:
#   python tools/capture_thumb.py <url> automata/static/thumbs/<world>.webp ['<pin-js>']
# The optional pin-JS forces a held frame (e.g. Verglas pins phase='intro'); omit for live scenes.
# Uses a cached chromium (Playwright won't install fresh on this OS) — set CHROME below if it moves.
import io
import os
import sys

from PIL import Image
from playwright.sync_api import sync_playwright

CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome")

URL = sys.argv[1]
OUT = sys.argv[2]
PIN = sys.argv[3] if len(sys.argv) > 3 else ""
CAPW, CAPH = 1000, 625
FRAMES, MS, QUALITY = 18, 100, 52

# keep only the canvas's top-level container, then drop footer/nav chrome inside it too
HIDE = """
const cv=document.querySelector('canvas');
let top=cv; while(top.parentElement && top.parentElement!==document.body) top=top.parentElement;
[...document.body.children].forEach(e=>{ if(e!==top) e.style.display='none'; });
document.querySelectorAll('#foot, footer, nav').forEach(e=>e.style.display='none');
"""

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox", "--disable-gpu"])
    pg = b.new_page(viewport={"width": CAPW, "height": CAPH}, device_scale_factor=1)
    pg.goto(URL, wait_until="load")
    pg.wait_for_timeout(1800)
    pg.evaluate(HIDE)
    if PIN:
        pg.evaluate(PIN)
    pg.wait_for_timeout(700)
    frames = []
    for _ in range(FRAMES):
        frames.append(Image.open(io.BytesIO(pg.screenshot())).convert("RGB"))
        pg.wait_for_timeout(MS)
    b.close()

frames = [f.resize((800, 500), Image.LANCZOS) for f in frames]
frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=MS, loop=0,
               format="WEBP", quality=QUALITY, method=6)
print(f"saved {OUT}: {os.path.getsize(OUT)} bytes, {len(frames)} frames")
