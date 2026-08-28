"""Browser-native skipping-orbit material for the IQHE edge-mode lab."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

import numpy as np

from toposc_lab.quantum_hall.integer_quantum_hall import (
    IQHEParameters,
    JOULE_PER_MEV,
    edge_mode_spectrum,
)
from toposc_lab.quantum_hall.landau_levels import ELEMENTARY_CHARGE, HBAR


def integer_quantum_hall_edge_configuration(
    parameters: IQHEParameters,
) -> dict[str, Any]:
    """Return a finite, JSON-safe configuration for the edge canvas."""
    spectrum = edge_mode_spectrum(parameters, grid_points=401)
    magnetic_length = np.sqrt(
        HBAR / (ELEMENTARY_CHARGE * parameters.magnetic_field_tesla)
    )
    sample_indices = np.linspace(0, spectrum.x_over_l_b.size - 1, 161, dtype=int)
    left_velocities = spectrum.crossing_velocity_m_s[spectrum.crossing_sides < 0]
    right_velocities = spectrum.crossing_velocity_m_s[spectrum.crossing_sides > 0]
    configuration = asdict(parameters)
    configuration.update(
        {
            "magnetic_length_nm": float(magnetic_length * 1.0e9),
            "sample_width_nm": float(
                parameters.edge_sample_width_l_b * magnetic_length * 1.0e9
            ),
            "mode_count_per_edge": spectrum.mode_count_per_edge,
            "edge_current_na": float(spectrum.edge_current_ampere * 1.0e9),
            "left_velocity_km_s": float(
                np.mean(left_velocities) / 1.0e3 if left_velocities.size else 0.0
            ),
            "right_velocity_km_s": float(
                np.mean(right_velocities) / 1.0e3 if right_velocities.size else 0.0
            ),
            "x_over_l_b": spectrum.x_over_l_b[sample_indices].tolist(),
            "k_l_b": spectrum.k_l_b[sample_indices].tolist(),
            "potential_mev": (
                spectrum.potential_joule[sample_indices] / JOULE_PER_MEV
            ).tolist(),
            "energies_mev": (
                spectrum.energies_joule[:, sample_indices] / JOULE_PER_MEV
            ).tolist(),
            "chemical_potential_mev": float(
                spectrum.chemical_potential_joule / JOULE_PER_MEV
            ),
            "crossing_k_l_b": spectrum.crossing_k_l_b.tolist(),
            "crossing_sides": spectrum.crossing_sides.tolist(),
        }
    )
    return configuration


def integer_quantum_hall_edge_live_html(parameters: IQHEParameters) -> str:
    """Build a self-contained edge-mode animation with no external assets."""
    serialized_configuration = json.dumps(
        integer_quantum_hall_edge_configuration(parameters),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    template = r"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root { color-scheme:dark; --ink:#edf8ff; --muted:#8eabc0; --cyan:#62e7ff; --blue:#4895ff; --green:#55e5a5; --pink:#e78bd4; --orange:#ff9d4d; --panel:rgba(5,19,32,.82); --line:rgba(126,194,224,.18); }
  * { box-sizing:border-box; }
  html,body { margin:0; background:transparent; color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,sans-serif; }
  #edge-live { overflow:hidden; border:1px solid rgba(98,231,255,.24); border-radius:22px; padding:17px; background:radial-gradient(circle at 12% 0%,rgba(30,145,203,.22),transparent 31%),radial-gradient(circle at 91% 8%,rgba(207,84,194,.14),transparent 27%),linear-gradient(145deg,#06111e,#071b2b 52%,#050d17); box-shadow:0 24px 75px rgba(0,7,16,.34); }
  .head { display:flex; justify-content:space-between; align-items:flex-start; gap:14px; margin-bottom:11px; }
  .eyebrow { color:var(--cyan); font-size:10px; font-weight:850; letter-spacing:.15em; text-transform:uppercase; }
  h1 { margin:4px 0 3px; font-size:clamp(21px,3vw,31px); letter-spacing:-.035em; }
  .subtitle { color:var(--muted); max-width:760px; font-size:12px; line-height:1.45; }
  .live-pill { display:flex; gap:7px; align-items:center; color:#9df4cd; border:1px solid rgba(85,229,165,.25); border-radius:999px; padding:7px 10px; background:rgba(85,229,165,.07); font-size:10px; font-weight:800; white-space:nowrap; }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--green); box-shadow:0 0 13px var(--green); animation:pulse 1.7s ease-in-out infinite; }
  @keyframes pulse { 50% { opacity:.35; transform:scale(.78); } }
  .metrics { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:7px; margin-bottom:10px; }
  .metric { min-width:0; border:1px solid var(--line); border-radius:11px; padding:8px 9px; background:rgba(2,13,24,.64); }
  .metric span { display:block; color:var(--muted); font-size:8px; letter-spacing:.08em; text-transform:uppercase; margin-bottom:3px; }
  .metric strong { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:13px; }
  .tutorial { border:1px solid rgba(98,231,255,.18); border-radius:14px; padding:10px; margin-bottom:10px; background:linear-gradient(135deg,rgba(7,31,49,.9),rgba(34,20,58,.75)); }
  .tutorial-head { display:flex; justify-content:space-between; gap:10px; align-items:center; margin-bottom:7px; }
  .tutorial h2 { margin:0; font-size:15px; }
  .tutorial small { color:var(--cyan); font-weight:800; letter-spacing:.09em; text-transform:uppercase; }
  .tutorial-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:6px; }
  .card { border:1px solid rgba(126,194,224,.12); border-radius:9px; padding:7px 8px; background:rgba(2,12,22,.6); }
  .card b { display:block; margin-bottom:3px; font-size:8px; letter-spacing:.08em; text-transform:uppercase; }
  .card p { margin:0; color:#bfd3df; font-size:9px; line-height:1.35; }
  .action b{color:var(--orange)} .observe b{color:var(--cyan)} .physics b{color:#c4a5ff} .expect b{color:var(--green)}
  .tutorial-nav { display:flex; gap:7px; margin-top:7px; align-items:center; }
  button { border:1px solid rgba(126,194,224,.22); border-radius:9px; padding:7px 10px; background:rgba(8,28,44,.9); color:var(--ink); font:inherit; font-size:10px; font-weight:750; cursor:pointer; }
  button:hover { border-color:rgba(98,231,255,.55); }
  .progress { flex:1; text-align:center; color:var(--muted); font-size:9px; }
  .stage { height:570px; border:1px solid var(--line); border-radius:16px; overflow:hidden; background:rgba(1,8,15,.72); position:relative; }
  canvas { display:block; width:100%; height:100%; }
  .legend { position:absolute; left:12px; bottom:10px; display:flex; flex-wrap:wrap; gap:6px; pointer-events:none; }
  .tag { border:1px solid var(--line); border-radius:999px; padding:5px 8px; background:rgba(2,13,23,.82); color:#bdd5e4; font-size:9px; }
  .tag i { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:5px; }
  .controls { display:grid; grid-template-columns:auto auto minmax(160px,1fr) minmax(160px,1fr); gap:8px; align-items:center; margin-top:10px; }
  .control { border:1px solid var(--line); border-radius:10px; padding:7px 9px; background:rgba(3,14,25,.62); }
  .control label { display:flex; justify-content:space-between; gap:8px; color:var(--muted); font-size:9px; margin-bottom:4px; }
  input[type=range] { width:100%; accent-color:#62e7ff; }
  .note { color:var(--muted); font-size:9px; line-height:1.45; margin:8px 2px 0; }
  @media(max-width:850px){.metrics{grid-template-columns:repeat(3,1fr)}.tutorial-grid{grid-template-columns:repeat(2,1fr)}.controls{grid-template-columns:1fr 1fr}.stage{height:650px}}
</style>
</head>
<body>
<section id="edge-live">
  <div class="head">
    <div><div class="eyebrow">2.1.1 Edge Modes / Live-Material</div><h1>Chirale Randkanäle</h1><div class="subtitle">Links laufen Zustände in +y, rechts in -y. Die Teilchenpunkte zeigen eine semiklassische Skipping-Orbit-Interpretation; ein stationärer Quantenzustand selbst besitzt keine verborgene klassische Bahn.</div></div>
    <div class="live-pill"><i class="dot"></i>browser-native</div>
  </div>
  <div class="metrics">
    <div class="metric"><span>Kanäle je Rand</span><strong id="m-modes">-</strong></div>
    <div class="metric"><span>Magnetische Länge</span><strong id="m-lb">-</strong></div>
    <div class="metric"><span>Probenbreite</span><strong id="m-width">-</strong></div>
    <div class="metric"><span>linker Rand</span><strong id="m-left">-</strong></div>
    <div class="metric"><span>rechter Rand</span><strong id="m-right">-</strong></div>
    <div class="metric"><span>Netto-Randstrom</span><strong id="m-current">-</strong></div>
  </div>
  <div class="tutorial">
    <div class="tutorial-head"><div><small>Geführtes Experiment</small><h2 id="t-title">-</h2></div><button id="t-setup">Aufbau anwenden</button></div>
    <div class="tutorial-grid">
      <div class="card action"><b>Aktion</b><p id="t-action"></p></div>
      <div class="card observe"><b>Beobachten</b><p id="t-observe"></p></div>
      <div class="card physics"><b>Physik</b><p id="t-physics"></p></div>
      <div class="card expect"><b>Erwartung</b><p id="t-expect"></p></div>
    </div>
    <div class="tutorial-nav"><button id="t-prev">Zurück</button><div class="progress" id="t-progress"></div><button id="t-next">Weiter</button></div>
  </div>
  <div class="stage"><canvas id="edge-canvas" aria-label="Live-Simulation chiraler Quanten-Hall-Randmoden"></canvas><div class="legend"><span class="tag"><i style="background:#55e5a5"></i>linker Rand: +y</span><span class="tag"><i style="background:#e78bd4"></i>rechter Rand: -y</span><span class="tag"><i style="background:#ff9d4d"></i>Fermi-Schnittpunkt</span></div></div>
  <div class="controls">
    <button id="play">Pause</button><button id="bias">Hall-Bias aus</button>
    <div class="control"><label><span>Bahnradius R/l_B</span><b id="radius-value"></b></label><input id="radius" type="range" min="0.4" max="4" step="0.1"></div>
    <div class="control"><label><span>Zeitlupe</span><b id="speed-value"></b></label><input id="speed" type="range" min="0.2" max="3" step="0.1"></div>
  </div>
  <p class="note">Die Dispersion rechts stammt aus E_n(k)=hbar omega_B(n+1/2)+V(-k l_B^2). Die Animation links ist bewusst semiklassisch; die quantisierte Stromsteigung wird aus N e^2/h berechnet.</p>
</section>
<script>
(() => {
  const cfg=__CONFIG__;
  const $=id=>document.getElementById(id), canvas=$("edge-canvas"), ctx=canvas.getContext("2d");
  const state={running:true,bias:true,time:0,last:performance.now(),radius:cfg.skipping_orbit_radius_l_b,speed:cfg.edge_animation_speed,tutorial:0,visible:true};
  const tutorial=[
    {title:"Zwei Ränder, zwei Chiralitäten",action:"Lassen Sie die Animation laufen.",observe:"Links wandern Punkte nach oben, rechts nach unten.",physics:"Für B>0 ist v_y=-(eB)^-1 dV/dX. Die Gradienten der beiden Wände haben entgegengesetzte Vorzeichen.",expect:"Im Gleichgewicht heben sich die beiden Randströme auf.",setup:()=>{state.running=true;state.bias=false;}},
    {title:"Skipping-Orbits",action:"Vergrößern Sie R/l_B.",observe:"Die Bögen greifen weiter in die Probe hinein.",physics:"Klassische Zyklotronbahnen werden an der Wand spiegelnd reflektiert und ergeben eine gerichtete Folge von Bögen.",expect:"Die Ausbreitungsrichtung an einem gegebenen Rand ändert sich nicht.",setup:()=>{state.radius=2.6;state.running=true;}},
    {title:"Quantenmechanische Randdispersion",action:"Vergleichen Sie die Punkte an den Rändern mit den Fermi-Schnittpunkten rechts.",observe:"Jeder Schnitt einer ansteigenden Randbranch mit mu liefert einen leitenden Kanal.",physics:"X=-k l_B^2 macht den Ortsgradienten des Potentials zu einer Dispersion in k.",expect:"Die Zahl der Kanäle je Rand stimmt mit der Zahl gefüllter Landau-Zweige überein.",setup:()=>{state.radius=1.4;state.bias=false;}},
    {title:"Quantisierter Hall-Strom",action:"Schalten Sie den Hall-Bias ein.",observe:"Eine Randrichtung wird hervorgehoben; die Stromanzeige wird ungleich null.",physics:"Eine chemische Potentialdifferenz erzeugt I_y=N(e^2/h)V_H.",expect:"Das Vorzeichen kehrt sich mit V_H um, die Steigung bleibt durch N quantisiert.",setup:()=>{state.bias=true;state.running=true;}}
  ];
  function mod(v,m){return((v%m)+m)%m}
  function resize(){const r=canvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d);ctx.setTransform(d,0,0,d,0,0)}
  function panel(x,y,w,h,title,sub){ctx.fillStyle="rgba(4,18,31,.86)";ctx.strokeStyle="rgba(126,194,224,.18)";ctx.lineWidth=1;ctx.beginPath();ctx.roundRect(x,y,w,h,13);ctx.fill();ctx.stroke();ctx.fillStyle="#edf8ff";ctx.font="700 13px system-ui";ctx.fillText(title,x+12,y+20);ctx.fillStyle="#8eabc0";ctx.font="9px system-ui";ctx.fillText(sub,x+12,y+35)}
  function arrow(x,y,dy,color){ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,y-dy*.5);ctx.lineTo(x,y+dy*.5);ctx.stroke();ctx.beginPath();ctx.moveTo(x,y+dy*.5);ctx.lineTo(x-4,y+dy*.5-Math.sign(dy)*7);ctx.lineTo(x+4,y+dy*.5-Math.sign(dy)*7);ctx.closePath();ctx.fill()}
  function skippingPoint(side,progress,p){const phi=mod(progress,Math.PI),seg=Math.floor(progress/Math.PI),R=state.radius*p.scaleX;return{x:side<0?p.left+R*Math.sin(phi):p.right-R*Math.sin(phi),y:p.top+mod(p.h*.45+(side<0?-1:1)*(2*R*seg+R*(1-Math.cos(phi))),p.h)}}
  function drawSample(p){panel(p.x,p.y,p.w,p.h,"Semiklassische Skipping-Orbits","B zeigt aus der Ebene; Gegenränder besitzen entgegengesetzte Chiralität");const s={left:p.x+38,right:p.x+p.w-38,top:p.y+52,h:p.h-75,scaleX:(p.w-76)/cfg.edge_sample_width_l_b};ctx.fillStyle="rgba(34,111,154,.09)";ctx.fillRect(s.left,s.top,s.right-s.left,s.h);ctx.strokeStyle="rgba(98,231,255,.25)";ctx.strokeRect(s.left,s.top,s.right-s.left,s.h);
    ctx.fillStyle="rgba(98,231,255,.34)";ctx.font="9px system-ui";for(let x=s.left+22;x<s.right;x+=42)for(let y=s.top+20;y<s.top+s.h;y+=42){ctx.beginPath();ctx.arc(x,y,1.7,0,Math.PI*2);ctx.fill();ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.strokeStyle="rgba(98,231,255,.10)";ctx.stroke()}
    for(let y=s.top+45;y<s.top+s.h-25;y+=82){arrow(s.left+9,y,-22,"#55e5a5");arrow(s.right-9,y,22,"#e78bd4")}
    const modes=Math.max(1,Math.min(6,cfg.mode_count_per_edge));for(const side of [-1,1])for(let m=0;m<modes;m++){const color=side<0?"#55e5a5":"#e78bd4",emphasis=!state.bias||((cfg.hall_voltage_microvolt>=0?side<0:side>0));for(let j=28;j>=0;j--){const q=skippingPoint(side,state.time*2.2+m*1.37-j*.055,s);ctx.globalAlpha=(emphasis?.7:.24)*(1-j/30)*.65;ctx.fillStyle=color;ctx.beginPath();ctx.arc(q.x,q.y,1.2,0,Math.PI*2);ctx.fill()}const q=skippingPoint(side,state.time*2.2+m*1.37,s);ctx.globalAlpha=emphasis?1:.42;ctx.fillStyle=color;ctx.shadowColor=color;ctx.shadowBlur=12;ctx.beginPath();ctx.arc(q.x,q.y,4.2,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0}ctx.globalAlpha=1;ctx.fillStyle="#8eabc0";ctx.font="9px system-ui";ctx.fillText("x",s.right+14,s.top+s.h/2);ctx.save();ctx.translate(s.left-18,s.top+18);ctx.rotate(-Math.PI/2);ctx.fillText("y",0,0);ctx.restore()}
  function drawDispersion(p){panel(p.x,p.y,p.w,p.h,"Energie und Randdispersion","Fermi-Schnittpunkte markieren leitende eindimensionale Kanäle");const a={l:p.x+44,r:p.x+p.w-15,t:p.y+52,b:p.y+p.h-35};const all=cfg.energies_mev.flat(),emin=Math.min(...all,cfg.chemical_potential_mev),emax=Math.max(...all,cfg.chemical_potential_mev),kmin=Math.min(...cfg.k_l_b),kmax=Math.max(...cfg.k_l_b);const mx=k=>a.l+(k-kmin)/(kmax-kmin)*(a.r-a.l),my=e=>a.b-(e-emin)/(emax-emin)*(a.b-a.t);ctx.strokeStyle="rgba(142,171,192,.35)";ctx.beginPath();ctx.moveTo(a.l,a.t);ctx.lineTo(a.l,a.b);ctx.lineTo(a.r,a.b);ctx.stroke();for(let n=0;n<cfg.energies_mev.length;n++){ctx.strokeStyle=["#4895ff","#55e5a5","#c4a5ff","#ff9d4d"][n%4];ctx.lineWidth=1.5;ctx.beginPath();cfg.k_l_b.forEach((k,i)=>{const x=mx(k),y=my(cfg.energies_mev[n][i]);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}ctx.strokeStyle="#ff9d4d";ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(a.l,my(cfg.chemical_potential_mev));ctx.lineTo(a.r,my(cfg.chemical_potential_mev));ctx.stroke();ctx.setLineDash([]);cfg.crossing_k_l_b.forEach((k,i)=>{ctx.fillStyle=cfg.crossing_sides[i]<0?"#55e5a5":"#e78bd4";ctx.beginPath();ctx.arc(mx(k),my(cfg.chemical_potential_mev),4,0,Math.PI*2);ctx.fill()});ctx.fillStyle="#8eabc0";ctx.font="9px system-ui";ctx.fillText("k l_B",a.r-24,a.b+19);ctx.fillText("E",a.l-18,a.t+4);ctx.fillStyle="#ffbd88";ctx.fillText("mu",a.r-20,my(cfg.chemical_potential_mev)-5)}
  function draw(){const w=canvas.clientWidth,h=canvas.clientHeight;ctx.clearRect(0,0,w,h);const gap=10;if(w<760){drawSample({x:8,y:8,w:w-16,h:h*.58-12});drawDispersion({x:8,y:h*.58+3,w:w-16,h:h*.42-11})}else{const left=(w-gap-16)*.64;drawSample({x:8,y:8,w:left,h:h-16});drawDispersion({x:8+left+gap,y:8,w:w-left-gap-16,h:h-16})}}
  function update(){$("m-modes").textContent=cfg.mode_count_per_edge;$("m-lb").textContent=cfg.magnetic_length_nm.toFixed(2)+" nm";$("m-width").textContent=cfg.sample_width_nm.toFixed(0)+" nm";$("m-left").textContent=(cfg.left_velocity_km_s>=0?"+":"")+cfg.left_velocity_km_s.toFixed(1)+" km/s";$("m-right").textContent=cfg.right_velocity_km_s.toFixed(1)+" km/s";$("m-current").textContent=state.bias?cfg.edge_current_na.toFixed(3)+" nA":"0 nA";$("radius").value=state.radius;$("speed").value=state.speed;$("radius-value").textContent=state.radius.toFixed(1);$("speed-value").textContent=state.speed.toFixed(1)+"x";$("play").textContent=state.running?"Pause":"Start";$("bias").textContent=state.bias?"Hall-Bias aus":"Hall-Bias ein"}
  function renderTutorial(){const t=tutorial[state.tutorial];$("t-title").textContent=t.title;$("t-action").textContent=t.action;$("t-observe").textContent=t.observe;$("t-physics").textContent=t.physics;$("t-expect").textContent=t.expect;$("t-progress").textContent=`${state.tutorial+1} / ${tutorial.length}`}
  function tick(now){const dt=Math.min(.05,(now-state.last)/1000);state.last=now;if(state.running&&state.visible)state.time+=dt*state.speed;draw();update();requestAnimationFrame(tick)}
  $("play").onclick=()=>state.running=!state.running;$("bias").onclick=()=>state.bias=!state.bias;$("radius").oninput=e=>state.radius=+e.target.value;$("speed").oninput=e=>state.speed=+e.target.value;$("t-prev").onclick=()=>{state.tutorial=mod(state.tutorial-1,tutorial.length);renderTutorial()};$("t-next").onclick=()=>{state.tutorial=(state.tutorial+1)%tutorial.length;renderTutorial()};$("t-setup").onclick=()=>{tutorial[state.tutorial].setup();renderTutorial()};new ResizeObserver(resize).observe(canvas);new IntersectionObserver(e=>state.visible=e[0].isIntersecting).observe(canvas);resize();renderTutorial();update();requestAnimationFrame(tick);
})();
</script>
</body>
</html>"""
    return template.replace("__CONFIG__", serialized_configuration)


def render_integer_quantum_hall_edge_live(
    streamlit: Any,
    parameters: IQHEParameters,
) -> None:
    """Embed the self-contained edge-mode canvas in Streamlit."""
    streamlit.iframe(
        integer_quantum_hall_edge_live_html(parameters),
        width="stretch",
        height=1_130,
    )


__all__ = [
    "integer_quantum_hall_edge_configuration",
    "integer_quantum_hall_edge_live_html",
    "render_integer_quantum_hall_edge_live",
]
