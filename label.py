#!/usr/bin/env python3
"""
Ground-truth labeling tool for the cartridge eval set.

Draw a rectangle around every readable cartridge number panel, type the number,
press Enter. Saves one JSON file per image. Autosaves on every change.

Usage:
    python label.py --images eval/images --labels eval/labels
    then open http://localhost:8000

Stdlib only. No dependencies.
"""

import argparse
import json
import mimetypes
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>cartridge labeler</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; height: 100vh; display: grid;
    grid-template-columns: 240px 1fr;
    font: 13px/1.4 ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    background: #14161a; color: #d6dae0;
  }
  #side {
    border-right: 1px solid #2a2f37; display: flex; flex-direction: column;
    min-height: 0;
  }
  #side h1 { font-size: 12px; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: #7e8794; margin: 0; padding: 12px 12px 8px; }
  #stats { padding: 0 12px 10px; color: #7e8794; border-bottom: 1px solid #2a2f37; }
  #stats b { color: #d6dae0; font-weight: 600; }
  #list { overflow-y: auto; flex: 1; min-height: 0; }
  .row {
    padding: 5px 12px; cursor: pointer; display: flex; justify-content: space-between;
    gap: 8px; border-left: 2px solid transparent; white-space: nowrap;
  }
  .row:hover { background: #1b1f26; }
  .row.on { background: #1f2630; border-left-color: #5b9dd9; color: #fff; }
  .row .n { color: #5f6b7a; }
  .row.done .n { color: #5aa06f; }
  .name { overflow: hidden; text-overflow: ellipsis; }

  #main { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
  #bar {
    display: flex; align-items: center; gap: 8px; padding: 8px 12px;
    border-bottom: 1px solid #2a2f37; flex-wrap: wrap;
  }
  button {
    font: inherit; background: #232830; color: #d6dae0; border: 1px solid #363d47;
    border-radius: 3px; padding: 3px 9px; cursor: pointer;
  }
  button:hover { background: #2c323c; }
  #hint { color: #7e8794; margin-left: auto; }
  kbd { background: #232830; border: 1px solid #363d47; border-radius: 3px;
    padding: 0 4px; color: #aeb6c1; }

  #stage { flex: 1; overflow: auto; min-height: 0; background: #0e1013; }
  #wrap { position: relative; transform-origin: 0 0; user-select: none; }
  #wrap img { display: block; width: 100%; height: 100%; }

  .bx { position: absolute; border: 2px solid #f2c14e; background: rgba(242,193,78,.10); }
  .bx.hard { border-style: dashed; border-color: #d98b5b; }
  .bx .tag {
    position: absolute; left: -2px; bottom: 100%; background: #f2c14e; color: #14161a;
    font-weight: 700; padding: 1px 5px; white-space: nowrap; cursor: pointer;
  }
  .bx.hard .tag { background: #d98b5b; }
  #draft { position: absolute; border: 2px dashed #5b9dd9; background: rgba(91,157,217,.12); }
  #entry {
    position: absolute; background: #14161a; border: 1px solid #5b9dd9; padding: 3px;
    z-index: 20;
  }
  #entry input {
    font: inherit; background: #0e1013; color: #fff; border: 0; outline: 0;
    width: 110px; padding: 3px 5px;
  }
  #empty { padding: 40px; color: #7e8794; }
</style>

<div id="side">
  <h1>Images</h1>
  <div id="stats"></div>
  <div id="list"></div>
</div>

<div id="main">
  <div id="bar">
    <button onclick="go(-1)">&larr; prev</button>
    <button onclick="go(1)">next &rarr;</button>
    <button onclick="zoom(1/1.25)">&minus;</button>
    <button onclick="zoom(1.25)">+</button>
    <button onclick="fit()">fit</button>
    <span id="hint">drag a box &middot; type <kbd>67xl</kbd> &middot; <kbd>?</kbd> suffix = hard to read &middot; <kbd>,</kbd> <kbd>.</kbd> to page</span>
  </div>
  <div id="stage"><div id="wrap"><img id="img"></div><div id="empty">No images found.</div></div>
</div>

<script>
let files = [], idx = 0, scale = 1, boxes = [], natW = 0, natH = 0;
let dragging = null, draft = null, entry = null;
const wrap = document.getElementById('wrap');
const img = document.getElementById('img');
const stage = document.getElementById('stage');

async function boot() {
  files = await (await fetch('/api/images')).json();
  if (!files.length) { wrap.style.display = 'none'; return; }
  document.getElementById('empty').style.display = 'none';
  const last = localStorage.getItem('lastImage');
  const i = files.findIndex(f => f.name === last);
  load(i >= 0 ? i : 0);
}

function sidebar() {
  const done = files.filter(f => f.count > 0).length;
  const total = files.reduce((a, f) => a + f.count, 0);
  document.getElementById('stats').innerHTML =
    `<b>${done}</b> / ${files.length} labeled<br><b>${total}</b> boxes total`;
  document.getElementById('list').innerHTML = files.map((f, i) =>
    `<div class="row ${i === idx ? 'on' : ''} ${f.count > 0 ? 'done' : ''}" onclick="load(${i})">
       <span class="name">${f.name}</span><span class="n">${f.count || ''}</span></div>`).join('');
  const on = document.querySelector('.row.on');
  if (on) on.scrollIntoView({ block: 'nearest' });
}

async function load(i) {
  killEntry();
  idx = i;
  const f = files[idx];
  localStorage.setItem('lastImage', f.name);
  const data = await (await fetch('/api/labels/' + encodeURIComponent(f.name))).json();
  boxes = data.boxes || [];
  img.onload = () => {
    natW = img.naturalWidth; natH = img.naturalHeight;
    fit();
  };
  img.src = '/img/' + encodeURIComponent(f.name);
  sidebar();
}

function go(d) {
  const n = idx + d;
  if (n >= 0 && n < files.length) load(n);
}

function fit() {
  if (!natW) return;
  zoomTo((stage.clientWidth - 24) / natW);
}
function zoom(m) { zoomTo(scale * m); }
function zoomTo(s) {
  scale = Math.max(0.03, Math.min(3, s));
  wrap.style.width = (natW * scale) + 'px';
  wrap.style.height = (natH * scale) + 'px';
  render();
}

function render() {
  wrap.querySelectorAll('.bx').forEach(e => e.remove());
  boxes.forEach((b, i) => {
    const [l, t, r, bt] = b.box;
    const d = document.createElement('div');
    d.className = 'bx' + (b.hard ? ' hard' : '');
    d.style.left = (l * scale) + 'px';
    d.style.top = (t * scale) + 'px';
    d.style.width = ((r - l) * scale) + 'px';
    d.style.height = ((bt - t) * scale) + 'px';
    const tag = document.createElement('div');
    tag.className = 'tag';
    tag.textContent = b.number + (b.xl ? 'XL' : '') + (b.hard ? '?' : '') + '  \u00d7';
    tag.title = 'click to delete';
    tag.onmousedown = e => { e.stopPropagation(); boxes.splice(i, 1); render(); save(); };
    d.appendChild(tag);
    wrap.appendChild(d);
  });
}

function pt(e) {
  const r = wrap.getBoundingClientRect();
  return {
    x: Math.round(Math.max(0, Math.min(natW, (e.clientX - r.left) / scale))),
    y: Math.round(Math.max(0, Math.min(natH, (e.clientY - r.top) / scale)))
  };
}

wrap.onmousedown = e => {
  if (e.button !== 0 || !natW) return;
  killEntry();
  dragging = pt(e);
  draft = document.createElement('div');
  draft.id = 'draft';
  wrap.appendChild(draft);
};
window.onmousemove = e => {
  if (!dragging) return;
  const p = pt(e);
  Object.assign(draft.style, {
    left: (Math.min(dragging.x, p.x) * scale) + 'px',
    top: (Math.min(dragging.y, p.y) * scale) + 'px',
    width: (Math.abs(p.x - dragging.x) * scale) + 'px',
    height: (Math.abs(p.y - dragging.y) * scale) + 'px'
  });
};
window.onmouseup = e => {
  if (!dragging) return;
  const p = pt(e), s = dragging;
  dragging = null;
  const box = [Math.min(s.x, p.x), Math.min(s.y, p.y), Math.max(s.x, p.x), Math.max(s.y, p.y)];
  if (box[2] - box[0] < 8 || box[3] - box[1] < 8) { draft.remove(); draft = null; return; }
  ask(box);
};

function ask(box) {
  entry = document.createElement('div');
  entry.id = 'entry';
  entry.style.left = (box[0] * scale) + 'px';
  entry.style.top = (box[3] * scale + 4) + 'px';
  const inp = document.createElement('input');
  inp.placeholder = '67xl';
  entry.appendChild(inp);
  wrap.appendChild(entry);
  inp.focus();
  inp.onkeydown = ev => {
    if (ev.key === 'Escape') { killEntry(); return; }
    if (ev.key !== 'Enter') return;
    const m = /^\s*(\d{2,4})\s*(xl)?\s*(\?)?\s*$/i.exec(inp.value);
    if (!m) { inp.style.background = '#3a1d1d'; return; }
    boxes.push({ number: m[1], xl: !!m[2], hard: !!m[3], box: box });
    killEntry(); render(); save();
  };
}

function killEntry() {
  if (entry) { entry.remove(); entry = null; }
  if (draft) { draft.remove(); draft = null; }
}

async function save() {
  const f = files[idx];
  f.count = boxes.length;
  sidebar();
  await fetch('/api/labels/' + encodeURIComponent(f.name), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: f.name, width: natW, height: natH, boxes: boxes })
  });
}

window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === ',') go(-1);
  if (e.key === '.') go(1);
});

boot();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    images_dir = ""
    labels_dir = ""

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _label_path(self, name):
        stem = os.path.splitext(os.path.basename(name))[0]
        return os.path.join(self.labels_dir, stem + ".json")

    def do_GET(self):
        path = unquote(urlparse(self.path).path)

        if path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")

        if path == "/api/images":
            names = sorted(
                f for f in os.listdir(self.images_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS
            )
            out = []
            for n in names:
                count = 0
                lp = self._label_path(n)
                if os.path.exists(lp):
                    try:
                        with open(lp) as fh:
                            count = len(json.load(fh).get("boxes", []))
                    except Exception:
                        count = 0
                out.append({"name": n, "count": count})
            return self._send(200, json.dumps(out))

        if path.startswith("/api/labels/"):
            lp = self._label_path(path[len("/api/labels/"):])
            if not os.path.exists(lp):
                return self._send(200, json.dumps({"boxes": []}))
            with open(lp) as fh:
                return self._send(200, fh.read())

        if path.startswith("/img/"):
            name = os.path.basename(path[len("/img/"):])
            fp = os.path.join(self.images_dir, name)
            if not os.path.isfile(fp):
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
            with open(fp, "rb") as fh:
                return self._send(200, fh.read(), ctype)

        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        if not path.startswith("/api/labels/"):
            return self._send(404, b"not found", "text/plain")
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        lp = self._label_path(path[len("/api/labels/"):])
        tmp = lp + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=1)
        os.replace(tmp, lp)
        return self._send(200, json.dumps({"ok": True}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    Handler.images_dir = os.path.abspath(os.path.expanduser(args.images))
    Handler.labels_dir = os.path.abspath(os.path.expanduser(args.labels))
    os.makedirs(Handler.labels_dir, exist_ok=True)

    n = len([f for f in os.listdir(Handler.images_dir)
             if os.path.splitext(f)[1].lower() in IMAGE_EXTS])
    print(f"{n} images in {Handler.images_dir}")
    print(f"labels -> {Handler.labels_dir}")
    print(f"open http://localhost:{args.port}   (ctrl-c to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()