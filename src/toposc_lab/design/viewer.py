"""Read-only loopback viewer. No training, external assets, or arbitrary file serving."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

HTML = """<!doctype html>
<html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Toposc · Graphbauer</title>
<style>
:root{color-scheme:dark;font:16px system-ui;background:#101722;color:#edf2f8}
body{max-width:1150px;margin:auto;padding:28px}h1{font-size:28px;margin-bottom:8px}
.muted{color:#aebed0}main{display:grid;grid-template-columns:2fr 1fr;gap:20px}
section{background:#192435;border:1px solid #304159;border-radius:12px;padding:18px}
canvas{width:100%;height:auto;background:#121c2a;border-radius:8px}
button,input{padding:8px;background:#24344b;color:white;border:1px solid #536982;border-radius:5px}
#notice{padding:12px;border-left:3px solid #efb960;margin:18px 0;color:#efd7af}
dt{color:#aebed0;margin-top:12px}dd{margin:3px 0;font-size:19px}#error{color:#ffacac}
@media(max-width:760px){main{grid-template-columns:1fr}}
</style>
<h1>Toposc · Graphbauer</h1><div class="muted">Punktplatzierung und Verbindungen live verfolgen</div>
<div id="notice">Aufbau ist keine Physikbewertung. Die mitgelieferte Demo optimiert nur Verbindungslängen.</div>
<p><button id="pause">Anzeige pausieren</button> <button id="live">Live</button>
<input id="episode" type="number" min="0" value="0" style="width:80px" aria-label="Episode">
<button id="replay">Episode abspielen</button></p>
<p id="error" role="status"></p>
<main><section><canvas id="graph" width="760" height="600" aria-label="Aktueller Graph"></canvas>
<p class="muted">Orange: zuletzt gesetzter Punkt oder letzte Kante. Koordinatenmaßstab bleibt fest.</p>
</section><section><dl id="stats"></dl><h3>Belohnung je Bauversuch</h3>
<canvas id="curve" width="350" height="210" aria-label="Belohnungskurve"></canvas>
<p class="muted">Fehlende Bewertungen bleiben Lücken. Geometrische Ablehnung: −1.
Keine Aussage über Topologie oder Supraleitung.</p></section></main>
<script>
let paused=false,replaying=false,frames=[],index=0,busy=false;
const $=id=>document.getElementById(id);
function draw(d){
 const c=$('graph'),g=c.getContext('2d'),s=d.state,r=d.rules,pad=35;
 g.clearRect(0,0,c.width,c.height);
 const scale=Math.min((c.width-2*pad)/r.width,(c.height-2*pad)/r.height);
 const xy=p=>[pad+p[0]*scale,c.height-pad-p[1]*scale];
 g.strokeStyle='#405773';g.lineWidth=1;g.strokeRect(pad,c.height-pad-r.height*scale,r.width*scale,r.height*scale);
 s.edges.forEach((e,i)=>{const a=xy(s.points[e[0]]),b=xy(s.points[e[1]]);
  g.strokeStyle=i===s.edges.length-1?'#ffbf69':'#63c8db';g.lineWidth=i===s.edges.length-1?3:1.5;
  g.beginPath();g.moveTo(...a);g.lineTo(...b);g.stroke();});
 s.points.forEach((p,i)=>{const q=xy(p);g.fillStyle=i===s.points.length-1&&s.edges.length===0?'#ffbf69':'#e6eef9';
  g.beginPath();g.arc(...q,4,0,2*Math.PI);g.fill();g.font='11px system-ui';g.fillText(String(i),q[0]+6,q[1]-5);});
 const values=[['Vergleichsarm',d.benchmark_arm??'Einzellauf'],['Seed',d.benchmark_seed??'siehe Manifest'],
 ['Episode',d.episode+1],['Ausführung / Schritt',`${d.attempt+1} / ${d.step}`],
 ['Zustand',d.stage],['Punkte',`${s.points.length} / ${r.site_count}`],
 ['Kanten',`${s.edges.length} / ${r.edge_count}`],['Bewertung',d.evaluation?.reward??'noch nicht bewertet'],
 ['Evaluator',d.evaluator],['Status',d.evaluation?.status??s.failure??'Aufbau'],
 ['Laufzeit',`${Math.round(d.elapsed)} s`]];
 $('stats').replaceChildren();values.forEach(([k,v])=>{const dt=document.createElement('dt'),dd=document.createElement('dd');
 dt.textContent=k;dd.textContent=v;$('stats').append(dt,dd);});
 const h=d.history,q=$('curve'),ctx=q.getContext('2d');ctx.clearRect(0,0,q.width,q.height);
 const finite=h.filter(x=>Number.isFinite(x.reward));if(!finite.length)return;
 const lo=Math.min(-1,...finite.map(x=>x.reward)),hi=Math.max(1,...finite.map(x=>x.reward));
 ctx.fillStyle='#aebed0';ctx.font='12px system-ui';ctx.fillText(hi.toFixed(2),2,15);ctx.fillText(lo.toFixed(2),2,200);
 ctx.strokeStyle='#63c8db';ctx.beginPath();let pen=false;
 h.forEach((v,i)=>{if(!Number.isFinite(v.reward)){pen=false;return;}
 const x=35+i*300/Math.max(1,h.length-1),y=190-(v.reward-lo)*170/(hi-lo);
 if(pen)ctx.lineTo(x,y);else ctx.moveTo(x,y);pen=true;});ctx.stroke();
}
$('pause').onclick=()=>{paused=!paused;$('pause').textContent=paused?'Anzeige fortsetzen':'Anzeige pausieren';};
$('live').onclick=()=>{replaying=false;paused=false;$('pause').textContent='Anzeige pausieren';};
$('replay').onclick=async()=>{try{const n=Number($('episode').value);
 if(!Number.isInteger(n)||n<0)throw Error('Ungültige Episodennummer');
 const r=await fetch('/episode/'+n);if(!r.ok)throw Error('Episode noch nicht versiegelt');
 frames=await r.json();index=0;replaying=true;paused=false;$('error').textContent='';
 }catch(e){$('error').textContent=e.message;}};
setInterval(async()=>{if(paused||busy)return;busy=true;try{
 if(replaying){if(index<frames.length)draw(frames[index++]);return;}
 const r=await fetch('/state',{cache:'no-store'});if(!r.ok)throw Error('Warte auf Fortschrittsdaten …');
 draw(await r.json());$('error').textContent='';
 }catch(e){$('error').textContent=e.message;}finally{busy=false;}},250);
</script></html>"""


def make_server(output: Path, port: int = 8766) -> ThreadingHTTPServer:
    root = output.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            try:
                if path == "/":
                    payload, mime = HTML.encode("utf-8"), "text/html; charset=utf-8"
                elif path == "/state":
                    payload, mime = (root / "live.json").read_bytes(), "application/json"
                elif path.startswith("/episode/") and path[9:].isascii() and path[9:].isdigit():
                    episode = int(path[9:])
                    from .training import load_sealed

                    record = load_sealed(root / f"episode_{episode:06d}.json")
                    trace = (root / record["execution"]).resolve()
                    if not trace.is_relative_to(root):
                        raise ValueError("invalid trace path")
                    import hashlib

                    data = trace.read_bytes()
                    if hashlib.sha256(data).hexdigest() != record["trace_sha256"]:
                        raise ValueError("invalid trace checksum")
                    payload = json.dumps([json.loads(line) for line in data.splitlines()]).encode()
                    mime = "application/json"
                else:
                    self.send_error(404)
                    return
            except (OSError, ValueError, KeyError):
                self.send_error(404, "No readable snapshot")
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
