"""Smooth browser-native live material for the Landau-level laboratory.

The component deliberately distinguishes stationary energy eigenstates from a
moving coherent wavepacket.  Animation runs inside an HTML5 canvas so changing
time does not trigger a Streamlit rerun for every frame.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from toposc_lab.quantum_hall.landau_levels import LandauLevelParameters


def landau_live_configuration(parameters: LandauLevelParameters) -> dict[str, Any]:
    """Return the validated, JSON-safe configuration consumed by the canvas."""
    configuration = asdict(parameters)
    configuration.update(
        {
            "field_slider_max_tesla": max(
                10.0,
                2.0 * parameters.magnetic_field_tesla,
            ),
            "electric_slider_max_v_per_m": max(
                5_000.0,
                2.0 * abs(parameters.electric_field_v_per_m),
            ),
            "initial_electron_count": min(
                12,
                max(1, parameters.maximum_level + 1),
            ),
        }
    )
    return configuration


def landau_live_material_html(parameters: LandauLevelParameters) -> str:
    """Build a self-contained live visualization with no external assets."""
    serialized_configuration = json.dumps(
        landau_live_configuration(parameters),
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
  :root {
    color-scheme: dark;
    --ink: #eaf6ff;
    --muted: #8da9bd;
    --cyan: #64e8ff;
    --blue: #3b82f6;
    --violet: #b794f6;
    --orange: #ff9f43;
    --green: #58e6a9;
    --panel: rgba(8, 23, 38, .88);
    --line: rgba(133, 190, 219, .18);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; background: transparent; font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: var(--ink); }
  #ll-live {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(103, 207, 255, .22);
    border-radius: 22px;
    padding: 18px;
    background:
      radial-gradient(circle at 18% 0%, rgba(38, 139, 210, .22), transparent 32%),
      radial-gradient(circle at 90% 18%, rgba(144, 88, 255, .16), transparent 28%),
      linear-gradient(145deg, #06111e 0%, #071827 48%, #050d18 100%);
    box-shadow: 0 24px 80px rgba(0, 8, 18, .34), inset 0 1px rgba(255,255,255,.04);
  }
  #ll-live::before {
    content: "";
    position: absolute; inset: 0; pointer-events: none; opacity: .24;
    background-image: linear-gradient(rgba(93,182,225,.025) 1px, transparent 1px), linear-gradient(90deg, rgba(93,182,225,.025) 1px, transparent 1px);
    background-size: 26px 26px;
  }
  .head { position: relative; display: flex; gap: 16px; align-items: flex-start; justify-content: space-between; margin-bottom: 13px; }
  .eyebrow { color: var(--cyan); font-size: 11px; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
  h1 { margin: 4px 0 3px; font-size: clamp(20px, 3vw, 31px); letter-spacing: -.035em; }
  .subtitle { color: var(--muted); font-size: 13px; line-height: 1.45; max-width: 720px; }
  .live-pill { display: flex; align-items: center; gap: 8px; border: 1px solid rgba(88,230,169,.26); background: rgba(88,230,169,.08); border-radius: 999px; padding: 7px 11px; color: #9ff6ce; font-size: 11px; font-weight: 750; white-space: nowrap; }
  .live-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 14px var(--green); animation: pulse 1.8s ease-in-out infinite; }
  @keyframes pulse { 50% { opacity: .35; transform: scale(.78); } }
  .metrics { position: relative; display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 8px; margin-bottom: 10px; }
  .metric { background: rgba(3,14,25,.62); border: 1px solid var(--line); border-radius: 12px; padding: 8px 10px; min-width: 0; }
  .metric span { display: block; color: var(--muted); font-size: 9px; text-transform: uppercase; letter-spacing: .09em; margin-bottom: 3px; }
  .metric strong { display: block; color: var(--ink); font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .tutorial { position: relative; margin-bottom: 11px; padding: 12px; border: 1px solid rgba(100,232,255,.2); border-radius: 16px; background: linear-gradient(135deg,rgba(7,29,47,.94),rgba(32,22,62,.82)); }
  .tutorial-head { display:flex; align-items:flex-start; justify-content:space-between; gap:14px; margin-bottom:9px; }
  .tutorial-kicker { color:var(--cyan); font-size:9px; font-weight:850; letter-spacing:.14em; text-transform:uppercase; }
  .tutorial h2 { margin:2px 0 0; font-size:16px; letter-spacing:-.015em; }
  .tutorial-status { flex:0 0 auto; color:#bcefdc; background:rgba(88,230,169,.08); border:1px solid rgba(88,230,169,.2); border-radius:999px; padding:6px 9px; font-size:9px; font-weight:750; }
  .tutorial-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:7px; }
  .tutorial-card { min-width:0; padding:8px 9px; border-radius:10px; background:rgba(2,12,22,.62); border:1px solid rgba(133,190,219,.13); }
  .tutorial-card span { display:block; margin-bottom:4px; font-size:8px; font-weight:850; letter-spacing:.08em; text-transform:uppercase; }
  .tutorial-card p { margin:0; color:#bfd2df; font-size:9.5px; line-height:1.4; }
  .tutorial-card.action span { color:#ffb66f; }
  .tutorial-card.observe span { color:#68e8ff; }
  .tutorial-card.physics span { color:#c6a8ff; }
  .tutorial-card.expect span { color:#74edb5; }
  .tutorial-nav { display:flex; align-items:center; gap:7px; margin-top:9px; }
  .tutorial-progress { display:flex; flex:1; justify-content:center; gap:5px; }
  .tutorial-dot { width:17px; height:5px; padding:0; border:0; border-radius:999px; background:rgba(141,169,189,.25); }
  .tutorial-dot.active { width:28px; background:linear-gradient(90deg,var(--cyan),var(--violet)); }
  .tutorial-target { position:relative; z-index:2; border-color:rgba(100,232,255,.74) !important; box-shadow:0 0 0 2px rgba(100,232,255,.14),0 0 24px rgba(59,130,246,.22); animation:tutorGlow 1.8s ease-in-out infinite; }
  @keyframes tutorGlow { 50% { box-shadow:0 0 0 3px rgba(100,232,255,.07),0 0 32px rgba(183,148,246,.25); } }
  .stage { position: relative; height: 560px; border: 1px solid var(--line); border-radius: 17px; overflow: hidden; background: rgba(1,8,16,.72); }
  canvas { display: block; width: 100%; height: 100%; }
  .stage-legend { position: absolute; left: 13px; bottom: 12px; display: flex; flex-wrap: wrap; gap: 7px; pointer-events: none; }
  .tag { padding: 5px 8px; border-radius: 999px; background: rgba(3,13,23,.78); border: 1px solid var(--line); color: #b9d4e5; font-size: 9px; backdrop-filter: blur(8px); }
  .tag i { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:5px; vertical-align:0; }
  .controls { position: relative; display: grid; gap: 10px; margin-top: 11px; }
  .control-row { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 9px; }
  .control { min-width: 0; padding: 9px 10px; border: 1px solid var(--line); border-radius: 12px; background: rgba(5,18,31,.72); }
  .control label { display:flex; justify-content:space-between; gap:8px; color:#b9d2e2; font-size:10px; font-weight:700; margin-bottom:7px; }
  .control output { color: var(--cyan); font-variant-numeric: tabular-nums; white-space: nowrap; }
  input[type=range] { width:100%; accent-color:#52dff8; cursor:pointer; }
  select { width:100%; color:var(--ink); background:#071a2b; border:1px solid rgba(118,190,222,.25); border-radius:8px; padding:7px; outline:none; }
  .buttons { display:flex; flex-wrap:wrap; gap:7px; align-items:center; }
  button { border:1px solid rgba(106,202,240,.24); border-radius:10px; padding:8px 11px; background:rgba(13,43,66,.9); color:#dff6ff; font-weight:750; font-size:10px; cursor:pointer; transition:.18s ease; }
  button:hover { transform:translateY(-1px); border-color:rgba(99,232,255,.65); }
  button.primary { color:#03131d; background:linear-gradient(135deg,#58e6ff,#8fffd0); border:0; }
  button.field-on { color:#07131b; background:var(--orange); border-color:transparent; }
  .checks { display:flex; flex-wrap:wrap; gap:10px 14px; color:#abc4d4; font-size:10px; align-items:center; }
  .checks label { display:flex; gap:6px; align-items:center; cursor:pointer; }
  .checks input { accent-color:#63e6ff; }
  .explain { display:grid; grid-template-columns:auto 1fr; gap:14px; align-items:start; margin-top:10px; padding:10px 12px; border-radius:12px; border:1px solid rgba(183,148,246,.18); background:rgba(102,64,166,.08); color:#b9cfe0; font-size:11px; line-height:1.45; }
  .explain b { color:#d9c5ff; }
  @media (max-width: 820px) {
    .metrics { grid-template-columns: repeat(2,minmax(0,1fr)); }
    .tutorial-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .stage { height: 500px; }
    .control-row { grid-template-columns: repeat(2,minmax(0,1fr)); }
  }
  @media (max-width: 520px) {
    #ll-live { padding: 11px; border-radius: 15px; }
    .head { display:block; }
    .live-pill { display:inline-flex; margin-top:8px; }
    .tutorial-head { display:block; }
    .tutorial-status { display:inline-block; margin-top:7px; }
    .tutorial-grid { grid-template-columns:1fr; }
    .tutorial-nav { flex-wrap:wrap; }
    .tutorial-progress { order:3; flex-basis:100%; }
    .stage { height: 540px; }
    .control-row { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<main id="ll-live">
  <section class="head">
    <div>
      <div class="eyebrow">Quantum motion laboratory</div>
      <h1>Elektronen im Landau-Feld – live</h1>
      <div class="subtitle">Kontinuierliche Browser-Simulation von Wellenpaket, Wahrscheinlichkeitsdichte, Leitzentrum und besetzten Energieniveaus.</div>
    </div>
    <div class="live-pill"><span class="live-dot"></span><span id="fps-label">LIVE · 60 FPS</span></div>
  </section>

  <section class="metrics">
    <div class="metric"><span>Aktuelles B</span><strong id="metric-b">–</strong></div>
    <div class="metric"><span>Magnetische Länge</span><strong id="metric-lb">–</strong></div>
    <div class="metric"><span>Landau-Abstand</span><strong id="metric-gap">–</strong></div>
    <div class="metric"><span>E×B-Drift</span><strong id="metric-drift">–</strong></div>
    <div class="metric"><span>Animationszeit</span><strong id="metric-time">–</strong></div>
  </section>

  <section class="tutorial" aria-live="polite">
    <div class="tutorial-head">
      <div>
        <div class="tutorial-kicker">Geführtes Physik-Tutorial · <span id="tutorial-count">1 / 9</span></div>
        <h2 id="tutorial-title">Orientierung: Was ist hier überhaupt real?</h2>
      </div>
      <div id="tutorial-status" class="tutorial-status">Schritt lesen und vorbereiten</div>
    </div>
    <div class="tutorial-grid">
      <article class="tutorial-card action"><span>1 · Das machst du</span><p id="tutorial-action"></p></article>
      <article class="tutorial-card observe"><span>2 · Darauf achtest du</span><p id="tutorial-observe"></p></article>
      <article class="tutorial-card physics"><span>3 · Physikalische Vorstellung</span><p id="tutorial-physics"></p></article>
      <article class="tutorial-card expect"><span>4 · Das sollte passieren</span><p id="tutorial-expect"></p></article>
    </div>
    <div class="tutorial-nav">
      <button id="tutorial-prev">← Zurück</button>
      <div id="tutorial-progress" class="tutorial-progress" aria-label="Tutorial-Fortschritt"></div>
      <button id="tutorial-setup" class="primary">Versuch vorbereiten</button>
      <button id="tutorial-next">Weiter →</button>
    </div>
  </section>

  <section id="live-stage" class="stage">
    <canvas id="landau-canvas" aria-label="Live-Simulation von Elektronen in Landau-Niveaus"></canvas>
    <div class="stage-legend">
      <span class="tag"><i style="background:#64e8ff"></i>Wahrscheinlichkeitsdichte</span>
      <span class="tag"><i style="background:#ff9f43"></i>Messprobe / Paketzentrum</span>
      <span class="tag"><i style="background:#b794f6"></i>Leitzentrum</span>
    </div>
  </section>

  <section class="controls">
    <div class="buttons">
      <button id="play" class="primary">❚❚ Pause</button>
      <button id="reset">↺ Experiment zurücksetzen</button>
      <button id="toggle-b" class="field-on">B-Feld AN</button>
      <button id="toggle-e">E-Feld AUS</button>
    </div>

    <div class="control-row">
      <div class="control">
        <label for="mode">Darstellung</label>
        <select id="mode">
          <option value="packet">Bewegtes Wellenpaket</option>
          <option value="landau">Landau-Gauge-Eigenzustand</option>
          <option value="symmetric">Symmetrischer LLL-Eigenzustand</option>
        </select>
      </div>
      <div class="control">
        <label for="electrons">Elektronen / Messproben <output id="electrons-out"></output></label>
        <input id="electrons" type="range" min="1" max="24" step="1">
      </div>
      <div class="control">
        <label for="b-target">Ziel-Magnetfeld <output id="b-out"></output></label>
        <input id="b-target" type="range" min="0.05" step="0.05">
      </div>
      <div class="control">
        <label for="e-target">Ziel-E-Feld <output id="e-out"></output></label>
        <input id="e-target" type="range" step="50">
      </div>
    </div>

    <div class="control-row">
      <div class="control">
        <label for="ramp">Feld-Rampenzeit <output id="ramp-out"></output></label>
        <input id="ramp" type="range" min="0.15" max="5" value="1.5" step="0.05">
      </div>
      <div class="control">
        <label for="speed">Animationsgeschwindigkeit <output id="speed-out"></output></label>
        <input id="speed" type="range" min="0.2" max="3" value="1" step="0.1">
      </div>
      <div class="control" style="grid-column:span 2">
        <label>Sichtbare Ebenen</label>
        <div class="checks">
          <label><input id="show-density" type="checkbox" checked> Dichte</label>
          <label><input id="show-samples" type="checkbox" checked> Messproben</label>
          <label><input id="show-current" type="checkbox" checked> Strompfeile</label>
          <label><input id="show-classical" type="checkbox" checked> klassische Vergleichsbahn</label>
        </div>
      </div>
    </div>
  </section>

  <section class="explain"><b id="state-title">Wellenpaket</b><span id="state-explanation"></span></section>
</main>

<script>
(() => {
  "use strict";
  const cfg = __CONFIG__;
  const C = { hbar: 1.054571817e-34, e: 1.602176634e-19, me: 9.1093837139e-31, mev: 1000 / 1.602176634e-19 };
  const root = document.getElementById("ll-live");
  const canvas = document.getElementById("landau-canvas");
  const ctx = canvas.getContext("2d", { alpha: false });
  const el = id => document.getElementById(id);
  const ui = {
    play: el("play"), reset: el("reset"), toggleB: el("toggle-b"), toggleE: el("toggle-e"), mode: el("mode"),
    electrons: el("electrons"), b: el("b-target"), e: el("e-target"), ramp: el("ramp"), speed: el("speed"),
    density: el("show-density"), samples: el("show-samples"), current: el("show-current"), classical: el("show-classical")
  };
  ui.electrons.value = cfg.initial_electron_count;
  ui.b.max = cfg.field_slider_max_tesla;
  ui.b.value = cfg.magnetic_field_tesla;
  ui.e.min = -cfg.electric_slider_max_v_per_m;
  ui.e.max = cfg.electric_slider_max_v_per_m;
  ui.e.value = cfg.electric_field_v_per_m;

  const state = {
    running: true, visible: true, bOn: true, eOn: Math.abs(cfg.electric_field_v_per_m) > 1e-12,
    B: cfg.magnetic_field_tesla, E: Math.abs(cfg.electric_field_v_per_m) > 1e-12 ? cfg.electric_field_v_per_m : 0,
    phase: 0, freeTime: 0, last: performance.now(), fps: 60, frameCount: 0, fpsClock: performance.now()
  };

  const tutorialSteps = Object.freeze([
    {
      title: "Orientierung: vier Bilder, nicht vier verschiedene Elektronen",
      action: "Bereite den Versuch vor, beobachte eine Runde und drücke dann Pause.",
      observe: "Blaue Wolke, oranger Punkt, violettes Leitzentrum und gestrichelte Vergleichsbahn.",
      physics: "Die Wolke ist |ψ|². Der Punkt ist Paketzentrum bzw. Messprobe; nur die gestrichelte Linie ist ein klassischer Vergleich.",
      expect: "Das Paket kreist um das ruhende Leitzentrum, während rechts diskrete Energieniveaus stehen.",
      targets: ["#live-stage", "#play"], setup: "orientation"
    },
    {
      title: "Nullfeld: Warum es ohne B keine Landau-Niveaus gibt",
      action: "Bereite B = 0 vor und vergleiche sofort Realraum und Energiepanel.",
      observe: "Die Feldpunkte verschwinden; oben stehen l_B = ∞ und beim Abstand ‚Kontinuum‘.",
      physics: "Ohne Lorentzkraft fehlt die quantisierte Zyklotronbewegung. Ein freies Teilchen besitzt E ∝ k².",
      expect: "Das Paket läuft frei; rechts ersetzt eine Parabel die getrennten Landau-Linien.",
      targets: ["#reset", "#metric-b", "#metric-gap", "#live-stage"], setup: "zero"
    },
    {
      title: "Magnetfeld einschalten: Entstehung der Quantisierung",
      action: "Bereite den Schritt vor und beobachte während der Feldrampe das aktuelle B.",
      observe: "Feldsymbole erscheinen, die Bahn krümmt sich und rechts wachsen waagerechte Niveaus auseinander.",
      physics: "ω_c = eB/m* quantisiert die Kreisbewegung; ΔE = ℏω_c und l_B = √(ℏ/eB).",
      expect: "B steigt weich auf 1 T, l_B wird endlich und das Paket geht in eine Zyklotronbewegung über.",
      targets: ["#toggle-b", "#b-target", "#metric-b", "#metric-lb"], setup: "field"
    },
    {
      title: "Dichte, Messung und Strom richtig auseinanderhalten",
      action: "Pausiere und schalte die vier sichtbaren Ebenen einzeln aus und wieder an.",
      observe: "Jedes Häkchen entfernt genau eine Information, der zugrunde liegende Zustand bleibt derselbe.",
      physics: "Messpunkte sind Born-Proben aus |ψ|², keine verborgenen Bahnen. Strompfeile zeigen Wahrscheinlichkeitsfluss.",
      expect: "Ohne Dichte bleibt der Messpunkt; ohne Messproben bleibt die Wolke; die Vergleichsbahn ist nur eine Hilfslinie.",
      targets: ["#show-density", "#show-samples", "#show-current", "#show-classical"], setup: "layers"
    },
    {
      title: "B verändern: Länge und Energie reagieren gegensinnig",
      action: "Starte bei 0,5 T und ziehe das Ziel-Magnetfeld langsam auf 3 T.",
      observe: "Die magnetische Länge sinkt, während der Landau-Abstand zunimmt.",
      physics: "Stärkeres B lokalisiert Zustände stärker und erhöht die Zyklotronenergie. Die Zeichenfläche nutzt l_B als Einheit.",
      expect: "Der Radius kann im normierten Bild ähnlich aussehen, obwohl die reale Ausdehnung in Nanometern schrumpft.",
      targets: ["#b-target", "#metric-lb", "#metric-gap"], setup: "strength"
    },
    {
      title: "E×B-Drift: Kreisbewegung mit wanderndem Zentrum",
      action: "Bereite E = 1500 V/m vor; ändere danach das Vorzeichen des E-Feldes.",
      observe: "Das violette Leitzentrum driftet quer zum E-Pfeil und die Energielinien werden schräg.",
      physics: "Zur Zyklotronbewegung kommt die Drift v_D = E×B/B². Ihr Betrag ist hier |E|/B.",
      expect: "Bei umgekehrtem E kehrt die Drift um; bei größerem B wird sie für dasselbe E langsamer.",
      targets: ["#toggle-e", "#e-target", "#metric-drift", "#live-stage"], setup: "drift"
    },
    {
      title: "Energie & Besetzung: So liest du das rechte Panel",
      action: "Beobachte n = 0, 1, …; schalte anschließend E kurz an und wieder aus.",
      observe: "Cyan markiert das gewählte Niveau, Orange den gewählten Zustand, Violett weitere dargestellte Besetzungen.",
      physics: "Bei E = 0 ist E_n unabhängig vom Leitzentrum. Ein E-Feld macht die Energie vom Impuls k abhängig.",
      expect: "Ohne E sind die Linien horizontal; mit E kippen sie. Ein aktivierter Zeeman-Term spaltet sie zusätzlich schwach.",
      targets: ["#live-stage", "#toggle-e"], setup: "energy"
    },
    {
      title: "Landau-Gauge: stationär heißt nicht punktförmig",
      action: "Bereite den Eigenzustand vor, pausiere und vergleiche Dichte mit den orangefarbenen Messproben.",
      observe: "Ein ortsfestes Dichteband liegt um x_c; einzelne Born-Proben erscheinen an verschiedenen Positionen.",
      physics: "|n,k⟩ ist ein Energieeigenzustand. Seine Dichte ist stationär; k bestimmt das Leitzentrum x_c.",
      expect: "Pause ändert die Dichte nicht. Mit E verschiebt sich das Band und es entsteht ein Driftstrom.",
      targets: ["#mode", "#live-stage"], setup: "landau"
    },
    {
      title: "Symmetrische Gauge: ein stationärer LLL-Ring",
      action: "Bereite den LLL-Zustand vor und trenne Ringdichte, Strompfeile und Messproben mit den Häkchen.",
      observe: "Die Dichte bildet einen Ring; grüne Tangenten zeigen Strom, orange Punkte mögliche Messorte.",
      physics: "Im Zustand |0,m⟩ liegt das Maximum ungefähr bei r = √(2m) l_B. Strom kann fließen, obwohl |ψ|² stationär ist.",
      expect: "Der Ring rotiert nicht als feste Materie. Größeres m in der Seitenleiste verschiebt sein Maximum nach außen.",
      targets: ["#mode", "#show-density", "#show-current", "#show-samples", "#live-stage"], setup: "symmetric"
    }
  ]);
  let tutorialIndex = 0;

  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const mix = (a, b, t) => a + (b - a) * t;
  const wrap = (x, extent) => ((x + extent) % (2 * extent) + 2 * extent) % (2 * extent) - extent;
  const mass = () => cfg.effective_mass_ratio * C.me;
  const lB = B => B > 1e-8 ? Math.sqrt(C.hbar / (C.e * B)) : Infinity;
  const omega = B => C.e * B / mass();
  const gapMeV = B => C.hbar * omega(B) * C.mev;
  const drift = (B, E) => B > 1e-8 ? E / B : 0;
  const fieldShiftLB = (B, E) => {
    if (B < 1e-8) return 0;
    return mass() * E / (C.e * B * B * lB(B));
  };
  const driftPerCycleLB = (B, E) => {
    if (B < 1e-8) return 0;
    return drift(B, E) * (2 * Math.PI / omega(B)) / lB(B);
  };

  function highlightTutorialTargets(selectors) {
    for (const node of document.querySelectorAll(".tutorial-target")) node.classList.remove("tutorial-target");
    const highlighted = new Set();
    for (const selector of selectors) {
      const node = document.querySelector(selector);
      if (!node) continue;
      let target = node;
      if (node.matches("input, select")) target = node.closest(".control") || node;
      else if (node.matches("strong")) target = node.closest(".metric") || node;
      if (!highlighted.has(target)) {
        target.classList.add("tutorial-target");
        highlighted.add(target);
      }
    }
  }

  function renderTutorial() {
    const step = tutorialSteps[tutorialIndex];
    el("tutorial-count").textContent = `${tutorialIndex + 1} / ${tutorialSteps.length}`;
    el("tutorial-title").textContent = step.title;
    el("tutorial-action").textContent = step.action;
    el("tutorial-observe").textContent = step.observe;
    el("tutorial-physics").textContent = step.physics;
    el("tutorial-expect").textContent = step.expect;
    el("tutorial-status").textContent = "Schritt lesen und vorbereiten";
    el("tutorial-prev").disabled = tutorialIndex === 0;
    el("tutorial-next").disabled = tutorialIndex === tutorialSteps.length - 1;
    const progress = el("tutorial-progress");
    progress.replaceChildren();
    tutorialSteps.forEach((item, index) => {
      const dot = document.createElement("button");
      dot.className = `tutorial-dot${index === tutorialIndex ? " active" : ""}`;
      dot.title = `${index + 1}: ${item.title}`;
      dot.setAttribute("aria-label", dot.title);
      dot.onclick = () => { tutorialIndex = index; renderTutorial(); };
      progress.appendChild(dot);
    });
    highlightTutorialTargets(step.targets);
  }

  function prepareTutorialStep() {
    const setup = tutorialSteps[tutorialIndex].setup;
    const bMax = Number(ui.b.max);
    const baseB = clamp(1, .05, bMax);
    const driftE = clamp(1500, Number(ui.e.min), Number(ui.e.max));
    ui.density.checked = true;
    ui.samples.checked = true;
    ui.current.checked = true;
    ui.classical.checked = true;
    ui.ramp.value = 1.5;
    ui.speed.value = 1;
    state.running = true;
    state.phase = 0;
    state.freeTime = 0;
    state.eOn = false;
    state.E = 0;
    ui.e.value = 0;

    if (setup === "zero") {
      ui.mode.value = "packet";
      state.bOn = false;
      state.B = 0;
    } else if (setup === "field") {
      ui.mode.value = "packet";
      ui.b.value = baseB;
      state.bOn = true;
      state.B = 0;
    } else if (setup === "strength") {
      ui.mode.value = "packet";
      ui.b.value = clamp(.5, .05, bMax);
      state.bOn = true;
      state.B = Number(ui.b.value);
    } else if (setup === "drift") {
      ui.mode.value = "packet";
      ui.b.value = baseB;
      ui.e.value = driftE;
      state.bOn = true;
      state.B = baseB;
      state.eOn = true;
    } else {
      ui.mode.value = setup === "landau" ? "landau" : setup === "symmetric" ? "symmetric" : "packet";
      ui.b.value = baseB;
      state.bOn = true;
      state.B = baseB;
    }
    updateUI();
    el("tutorial-status").textContent = "Versuch vorbereitet · jetzt beobachten";
  }

  function hermite(n, x) {
    if (n === 0) return 1;
    if (n === 1) return 2 * x;
    let hm2 = 1, hm1 = 2 * x, h = hm1;
    for (let j = 2; j <= n; j++) { h = 2 * x * hm1 - 2 * (j - 1) * hm2; hm2 = hm1; hm1 = h; }
    return h;
  }
  function hashNoise(i, t) {
    const x = Math.sin(i * 127.1 + Math.floor(t * 2.5) * 311.7) * 43758.5453;
    return x - Math.floor(x);
  }
  function gaussianPair(i, t) {
    const u = Math.max(1e-6, hashNoise(i * 2 + 1, t));
    const v = hashNoise(i * 2 + 2, t);
    const r = Math.sqrt(-2 * Math.log(u));
    return [r * Math.cos(2 * Math.PI * v), r * Math.sin(2 * Math.PI * v)];
  }

  let cssW = 900, cssH = 560, dpr = 1;
  function resize() {
    const rect = canvas.getBoundingClientRect();
    cssW = Math.max(320, rect.width); cssH = Math.max(420, rect.height); dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  new ResizeObserver(resize).observe(canvas);
  new IntersectionObserver(entries => { state.visible = entries[0].isIntersecting; }, { threshold: .02 }).observe(root);

  function roundRect(x, y, w, h, r, fill, stroke) {
    ctx.beginPath(); ctx.roundRect(x, y, w, h, r);
    if (fill) ctx.fill(); if (stroke) ctx.stroke();
  }
  function arrow(x1, y1, x2, y2, color, alpha = 1, width = 1.3) {
    const angle = Math.atan2(y2 - y1, x2 - x1), size = 5;
    ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - size * Math.cos(angle - .55), y2 - size * Math.sin(angle - .55));
    ctx.lineTo(x2 - size * Math.cos(angle + .55), y2 - size * Math.sin(angle + .55)); ctx.closePath(); ctx.fill(); ctx.restore();
  }
  function glowDot(x, y, radius, color, alpha = 1) {
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius * 3.1);
    gradient.addColorStop(0, color); gradient.addColorStop(.22, color); gradient.addColorStop(1, "rgba(0,0,0,0)");
    ctx.save(); ctx.globalAlpha = alpha; ctx.fillStyle = gradient; ctx.beginPath(); ctx.arc(x, y, radius * 3.1, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = "#f7fdff"; ctx.beginPath(); ctx.arc(x, y, Math.max(1.2, radius * .42), 0, 2 * Math.PI); ctx.fill(); ctx.restore();
  }

  function panels() {
    const gap = 12, pad = 15;
    const stacked = cssW < 620;
    if (stacked) {
      const upper = Math.max(270, cssH * .62);
      return { main: { x: pad, y: pad, w: cssW - 2 * pad, h: upper - pad }, energy: { x: pad, y: upper + gap, w: cssW - 2 * pad, h: cssH - upper - gap - pad } };
    }
    const mainW = cssW * .68;
    return { main: { x: pad, y: pad, w: mainW - pad, h: cssH - 2 * pad }, energy: { x: mainW + gap, y: pad, w: cssW - mainW - gap - pad, h: cssH - 2 * pad } };
  }
  function drawPanel(p, title, subtitle) {
    ctx.fillStyle = "rgba(3,13,23,.88)"; ctx.strokeStyle = "rgba(112,190,224,.17)"; ctx.lineWidth = 1;
    roundRect(p.x, p.y, p.w, p.h, 14, true, true);
    ctx.fillStyle = "#dff5ff"; ctx.font = "700 12px Inter,system-ui"; ctx.fillText(title, p.x + 14, p.y + 21);
    ctx.fillStyle = "#7896aa"; ctx.font = "10px Inter,system-ui"; ctx.fillText(subtitle, p.x + 14, p.y + 37);
  }
  function drawGrid(p, extent) {
    const cx = p.x + p.w / 2, cy = p.y + p.h / 2 + 7, scale = Math.min(p.w, p.h - 44) / (2 * extent * 1.08);
    ctx.save(); ctx.strokeStyle = "rgba(106,176,208,.08)"; ctx.lineWidth = 1;
    for (let j = -Math.floor(extent); j <= Math.floor(extent); j++) {
      ctx.beginPath(); ctx.moveTo(cx + j * scale, p.y + 44); ctx.lineTo(cx + j * scale, p.y + p.h - 10); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(p.x + 10, cy + j * scale); ctx.lineTo(p.x + p.w - 10, cy + j * scale); ctx.stroke();
    }
    ctx.strokeStyle = "rgba(118,204,238,.22)";
    ctx.beginPath(); ctx.moveTo(p.x + 10, cy); ctx.lineTo(p.x + p.w - 10, cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, p.y + 44); ctx.lineTo(cx, p.y + p.h - 10); ctx.stroke(); ctx.restore();
    return { cx, cy, scale, X: x => cx + x * scale, Y: y => cy - y * scale };
  }
  function drawFields(p, map) {
    const bStrength = clamp(state.B / Math.max(.1, Number(ui.b.value)), 0, 1);
    if (state.B > 1e-5) {
      ctx.save(); ctx.strokeStyle = `rgba(100,232,255,${.12 + .28 * bStrength})`; ctx.fillStyle = `rgba(100,232,255,${.18 + .45 * bStrength})`;
      const nx = 8, ny = 6;
      for (let ix = 0; ix < nx; ix++) for (let iy = 0; iy < ny; iy++) {
        const x = mix(p.x + 28, p.x + p.w - 28, ix / (nx - 1)); const y = mix(p.y + 60, p.y + p.h - 22, iy / (ny - 1));
        ctx.beginPath(); ctx.arc(x, y, 3.1, 0, 2 * Math.PI); ctx.stroke(); ctx.beginPath(); ctx.arc(x, y, 1.05, 0, 2 * Math.PI); ctx.fill();
      }
      ctx.restore();
    }
    if (Math.abs(state.E) > 1e-4) {
      const direction = Math.sign(state.E); const y = p.y + 55;
      arrow(map.cx - direction * 42, y, map.cx + direction * 42, y, "#ff9f43", .9, 2);
      ctx.fillStyle = "#ffbd7a"; ctx.font = "700 10px Inter,system-ui"; ctx.fillText("Eₓ", map.cx + direction * 50 - 7, y + 3);
    }
  }

  function packetPosition(i, phaseOffset = 0) {
    const extent = cfg.view_extent_l_b;
    if (state.B < 1e-5) {
      const x = wrap(-extent + state.freeTime * 1.2 + i * .36, extent);
      const y = (i - (Number(ui.electrons.value) - 1) / 2) * .18;
      return { x, y, gx: x, gy: y, angle: 0 };
    }
    const columns = Math.max(1, Math.ceil(Math.sqrt(Number(ui.electrons.value))));
    const row = Math.floor(i / columns), col = i % columns;
    const gx = (col - (columns - 1) / 2) * .34;
    const gy0 = (row - 1) * .28;
    const cycles = state.phase / (2 * Math.PI);
    const gy = gy0 + driftPerCycleLB(state.B, state.E) * cycles;
    const angle = state.phase + cfg.orbit_phase_radians + i * 2.399963 + phaseOffset;
    const radius = cfg.orbit_radius_l_b * (.78 + .07 * (i % 5));
    return { x: gx - radius * Math.sin(angle), y: wrap(gy + radius * Math.cos(angle), extent), gx, gy: wrap(gy, extent), angle };
  }
  function drawWavepacket(p, map) {
    const count = Number(ui.electrons.value), extent = cfg.view_extent_l_b;
    if (state.B > 1e-5 && ui.classical.checked) {
      const center = packetPosition(0);
      ctx.save(); ctx.setLineDash([5, 5]); ctx.strokeStyle = "rgba(255,159,67,.42)"; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.arc(map.X(center.gx), map.Y(center.gy), cfg.orbit_radius_l_b * map.scale, 0, 2 * Math.PI); ctx.stroke(); ctx.restore();
      ctx.fillStyle = "rgba(255,183,112,.72)"; ctx.font = "9px Inter,system-ui"; ctx.fillText("⟨r(t)⟩ / klassische Vergleichsbahn", p.x + 14, p.y + p.h - 32);
    }
    for (let i = count - 1; i >= 0; i--) {
      const pos = packetPosition(i), selected = i === 0;
      if (ui.density.checked) {
        const widthLB = .42 * Math.sqrt(2 * (i % (cfg.maximum_level + 1)) + 1);
        const radius = Math.max(10, widthLB * map.scale * 2.7);
        const g = ctx.createRadialGradient(map.X(pos.x), map.Y(pos.y), 0, map.X(pos.x), map.Y(pos.y), radius);
        g.addColorStop(0, selected ? "rgba(100,232,255,.72)" : "rgba(89,143,255,.34)");
        g.addColorStop(.32, selected ? "rgba(74,153,255,.30)" : "rgba(120,101,255,.13)"); g.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(map.X(pos.x), map.Y(pos.y), radius, 0, 2 * Math.PI); ctx.fill();
      }
      if (ui.samples.checked) glowDot(map.X(pos.x), map.Y(pos.y), selected ? 3.2 : 2.1, selected ? "#ff9f43" : "#64e8ff", selected ? 1 : .65);
      if (ui.current.checked && state.B > 1e-5 && (selected || i % 4 === 0)) {
        const tangentX = -Math.cos(pos.angle), tangentY = Math.sin(pos.angle);
        arrow(map.X(pos.x), map.Y(pos.y), map.X(pos.x) + tangentX * 18, map.Y(pos.y) - tangentY * 18, "#58e6a9", .72, 1.4);
      }
      if (selected && state.B > 1e-5) {
        ctx.strokeStyle = "rgba(183,148,246,.72)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(map.X(pos.gx), map.Y(pos.gy), 5, 0, 2 * Math.PI); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(map.X(pos.gx)-7,map.Y(pos.gy));ctx.lineTo(map.X(pos.gx)+7,map.Y(pos.gy));ctx.moveTo(map.X(pos.gx),map.Y(pos.gy)-7);ctx.lineTo(map.X(pos.gx),map.Y(pos.gy)+7);ctx.stroke();
      }
    }
    if (state.B < 1e-5 && ui.density.checked) {
      ctx.fillStyle = "#9eb8c8"; ctx.font = "10px Inter,system-ui"; ctx.fillText("B = 0: freies Paket breitet sich aus", p.x + 14, p.y + p.h - 32);
    }
  }

  function drawNoFieldEigenstate(p, map) {
    const extent = cfg.view_extent_l_b;
    const x = wrap(-extent + state.freeTime * .75, extent);
    const width = .85 + .08 * state.freeTime;
    if (ui.density.checked) {
      const radius = Math.min(3.2, width) * map.scale;
      const g = ctx.createRadialGradient(map.X(x), map.cy, 0, map.X(x), map.cy, radius);
      g.addColorStop(0, "rgba(100,232,255,.52)");
      g.addColorStop(.38, "rgba(74,153,255,.2)");
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(map.X(x), map.cy, radius, 0, 2 * Math.PI); ctx.fill();
    }
    if (ui.samples.checked) glowDot(map.X(x), map.cy, 2.8, "#ff9f43", .9);
    ctx.fillStyle = "#ffbd7a"; ctx.font = "700 10px Inter,system-ui";
    ctx.fillText("B = 0: noch kein Landau-Eigenzustand – freies Kontinuum", p.x + 14, p.y + p.h - 32);
  }

  function drawLandauGauge(p, map) {
    if (state.B < 1e-5) { drawNoFieldEigenstate(p, map); return; }
    const extent = cfg.view_extent_l_b, n = cfg.selected_level;
    const xc = -cfg.wave_number_l_b + fieldShiftLB(state.B, state.E);
    if (ui.density.checked) {
      const samples = 180, values = [], hValues = []; let maxD = 1e-12;
      for (let j=0;j<samples;j++) { const x = mix(-extent, extent, j/(samples-1)); const xi = x - xc; const h = hermite(n, xi); const d = h*h*Math.exp(-xi*xi); values.push(x); hValues.push(d); maxD=Math.max(maxD,d); }
      for (let j=0;j<samples-1;j++) { const alpha = .04 + .78*hValues[j]/maxD; ctx.fillStyle=`rgba(80,206,255,${alpha})`; const x1=map.X(values[j]),x2=map.X(values[j+1]);ctx.fillRect(x1,p.y+44,Math.max(1,x2-x1+1),p.h-56); }
    }
    const phaseVelocity = state.B > 1e-5 ? drift(state.B,state.E)/Math.max(1,Math.abs(drift(state.B,state.E))) : 0;
    ctx.save(); ctx.strokeStyle="rgba(216,244,255,.22)"; ctx.lineWidth=1;
    for(let j=-8;j<=8;j++){ const y=wrap(j*1.1 + state.phase*.15*phaseVelocity,extent);ctx.beginPath();ctx.moveTo(map.X(xc-2.5),map.Y(y));ctx.lineTo(map.X(xc+2.5),map.Y(y));ctx.stroke(); } ctx.restore();
    ctx.save(); ctx.strokeStyle="#b794f6";ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(map.X(xc),p.y+44);ctx.lineTo(map.X(xc),p.y+p.h-12);ctx.stroke();ctx.restore();
    if(ui.samples.checked){ const count=Number(ui.electrons.value); for(let i=0;i<count;i++){const g=gaussianPair(i,state.freeTime);const x=xc+g[0]*Math.sqrt(n+.5)*.52;const y=mix(-extent*.82,extent*.82,hashNoise(i+99,state.freeTime));glowDot(map.X(x),map.Y(y),2.2,"#ff9f43",.8);} }
    if(ui.current.checked && Math.abs(state.E)>1e-4){for(let j=-3;j<=3;j++){const y=j*1.35;arrow(map.X(xc),map.Y(y),map.X(xc),map.Y(y)-Math.sign(state.E)*22,"#58e6a9",.72,1.4);}}
    ctx.fillStyle="#a9c4d5";ctx.font="10px Inter,system-ui";ctx.fillText(`stationär: |n=${n}, k l_B=${cfg.wave_number_l_b.toFixed(1)}⟩ · x_c/l_B=${xc.toFixed(2)}`,p.x+14,p.y+p.h-32);
  }

  function drawSymmetric(p,map){
    if(state.B<1e-5){drawNoFieldEigenstate(p,map);return;}
    const m=cfg.angular_momentum,extent=cfg.view_extent_l_b,peak=Math.sqrt(2*m),maxLog=m===0?0:2*m*Math.log(Math.max(1e-6,peak))-.5*peak*peak;
    if(ui.density.checked){for(let r=0;r<extent;r+=.065){const logD=m===0?-.5*r*r:2*m*Math.log(Math.max(1e-6,r))-.5*r*r;const a=.015+.62*Math.exp(logD-maxLog);ctx.strokeStyle=`rgba(100,232,255,${clamp(a,0,.68)})`;ctx.lineWidth=Math.max(1,.09*map.scale);ctx.beginPath();ctx.arc(map.cx,map.cy,r*map.scale,0,2*Math.PI);ctx.stroke();}}
    if(peak>0){ctx.save();ctx.setLineDash([5,5]);ctx.strokeStyle="rgba(183,148,246,.66)";ctx.beginPath();ctx.arc(map.cx,map.cy,peak*map.scale,0,2*Math.PI);ctx.stroke();ctx.restore();}
    if(ui.current.checked && peak>.1){for(let j=0;j<12;j++){const a=2*Math.PI*j/12-state.phase*.18;const x=map.cx+peak*map.scale*Math.cos(a),y=map.cy-peak*map.scale*Math.sin(a);arrow(x,y,x-13*Math.sin(a),y-13*Math.cos(a),"#58e6a9",.68,1.2);}}
    if(ui.samples.checked){const count=Number(ui.electrons.value);for(let i=0;i<count;i++){const noise=gaussianPair(i,state.freeTime);const r=Math.max(0,peak+noise[0]*.5);const a=2*Math.PI*hashNoise(i+201,state.freeTime);glowDot(map.cx+r*map.scale*Math.cos(a),map.cy-r*map.scale*Math.sin(a),2.3,"#ff9f43",.82);}}
    ctx.fillStyle="#a9c4d5";ctx.font="10px Inter,system-ui";ctx.fillText(`stationär: |0,m=${m}⟩ · Dichtemaximum r=${peak.toFixed(2)} l_B`,p.x+14,p.y+p.h-32);
  }

  function drawEnergy(p){
    drawPanel(p,"Energie & Besetzung",state.B<1e-5?"freies Kontinuum":"Landau-Niveaus Eₙ(k)");
    const x0=p.x+38,x1=p.x+p.w-15,y0=p.y+p.h-30,y1=p.y+52,w=x1-x0,h=y0-y1;
    ctx.strokeStyle="rgba(135,193,219,.24)";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x0,y1);ctx.lineTo(x0,y0);ctx.lineTo(x1,y0);ctx.stroke();
    ctx.fillStyle="#6f8ea2";ctx.font="9px Inter,system-ui";ctx.fillText("E",x0-15,y1+3);ctx.fillText("k l_B",x1-22,y0+18);
    if(state.B<1e-5){ctx.strokeStyle="#64e8ff";ctx.lineWidth=2;ctx.beginPath();for(let j=0;j<=100;j++){const q=-1+2*j/100;const x=mix(x0,x1,j/100),y=y0-h*(.08+.72*q*q);j?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();ctx.fillStyle="#a8c4d5";ctx.fillText("B=0: E∝k²",x0+10,y1+16);return;}
    const nmax=cfg.maximum_level,targetB=Math.max(.05,Number(ui.b.value),state.B),baseMax=gapMeV(targetB)*(nmax+1.2),lb=lB(state.B);
    const energyAt=(n,k)=>gapMeV(state.B)*(n+.5)+C.e*state.E*lb*k*C.mev;
    for(let n=0;n<=nmax;n++){
      const selected=n===cfg.selected_level;ctx.strokeStyle=selected?"#64e8ff":`rgba(111,155,255,${.26+.45*n/Math.max(1,nmax)})`;ctx.lineWidth=selected?2.6:1.25;
      const eL=energyAt(n,-4),eR=energyAt(n,4),ya=y0-h*clamp(eL/baseMax,-.08,1.08),yb=y0-h*clamp(eR/baseMax,-.08,1.08);ctx.beginPath();ctx.moveTo(x0,ya);ctx.lineTo(x1,yb);ctx.stroke();
      if(cfg.include_zeeman){const dz=Math.abs(cfg.g_factor)*5.7883818e-2*state.B/2;for(const s of [-1,1]){const off=h*s*dz/baseMax;ctx.strokeStyle="rgba(183,148,246,.34)";ctx.lineWidth=.8;ctx.beginPath();ctx.moveTo(x0,ya-off);ctx.lineTo(x1,yb-off);ctx.stroke();}}
      ctx.fillStyle=selected?"#bdf5ff":"#758fa3";ctx.font=selected?"700 9px Inter,system-ui":"9px Inter,system-ui";ctx.fillText(`n=${n}`,x0+4,mix(ya,yb,.5)-3);
    }
    const count=Number(ui.electrons.value);for(let i=0;i<count;i++){const n=i===0?cfg.selected_level:(i-1)%(nmax+1),k=i===0?cfg.wave_number_l_b:-3.2+6.4*((i*.6180339)%1),e=energyAt(n,k),x=mix(x0,x1,(k+4)/8),y=y0-h*clamp(e/baseMax,-.08,1.08);glowDot(x,y,i===0?2.8:1.7,i===0?"#ff9f43":"#b794f6",i===0?1:.68);}
  }

  function updateUI(){
    el("electrons-out").textContent=ui.electrons.value;el("b-out").textContent=`${Number(ui.b.value).toFixed(2)} T`;el("e-out").textContent=`${Number(ui.e.value).toFixed(0)} V/m`;el("ramp-out").textContent=`${Number(ui.ramp.value).toFixed(2)} s`;el("speed-out").textContent=`${Number(ui.speed.value).toFixed(1)}×`;
    el("metric-b").textContent=`${state.B.toFixed(3)} T`;el("metric-lb").textContent=state.B>1e-5?`${(lB(state.B)*1e9).toFixed(2)} nm`:"∞";el("metric-gap").textContent=state.B>1e-5?`${gapMeV(state.B).toFixed(3)} meV`:"Kontinuum";el("metric-drift").textContent=state.B>1e-5?`${drift(state.B,state.E).toFixed(1)} m/s`:"–";el("metric-time").textContent=`${(state.phase/(2*Math.PI)).toFixed(2)} T_B`;
    ui.play.textContent=state.running?"❚❚ Pause":"▶ Start";ui.toggleB.textContent=state.bOn?"B-Feld AN":"B-Feld AUS";ui.toggleE.textContent=state.eOn?"E-Feld AN":"E-Feld AUS";ui.toggleB.classList.toggle("field-on",state.bOn);ui.toggleE.classList.toggle("field-on",state.eOn);
    ui.classical.disabled=ui.mode.value!=="packet";ui.classical.parentElement.style.opacity=ui.classical.disabled ? .38 : 1;
    const texts={packet:["Wellenpaket","Ein kohärentes Paket ist eine Überlagerung mehrerer Landau-Eigenzustände. Sein Erwartungswert kann eine Zyklotronbahn durchlaufen; die leuchtende Fläche zeigt seine endliche quantenmechanische Ausdehnung."],landau:["Landau-Gauge-Eigenzustand","Die Dichte eines Energieeigenzustands ist stationär. Orange Punkte sind unabhängige Born-Messproben – keine verborgenen klassischen Bahnen. Mit Eₓ driftet das gesamte Leitzentrum."],symmetric:["Symmetrischer LLL-Eigenzustand","Der Ring ist eine stationäre Wahrscheinlichkeitsdichte des Zustands |0,m⟩. Grüne Tangenten zeigen die Wahrscheinlichkeitsströmung; die Messpunkte werden aus derselben Dichte gezogen."]};
    el("state-title").textContent=texts[ui.mode.value][0];el("state-explanation").textContent=texts[ui.mode.value][1];
  }

  function draw(){
    const gradient=ctx.createLinearGradient(0,0,cssW,cssH);gradient.addColorStop(0,"#020a13");gradient.addColorStop(.55,"#041421");gradient.addColorStop(1,"#020913");ctx.fillStyle=gradient;ctx.fillRect(0,0,cssW,cssH);
    const ps=panels();drawPanel(ps.main,"Realraum",ui.mode.options[ui.mode.selectedIndex].text);const map=drawGrid(ps.main,cfg.view_extent_l_b);drawFields(ps.main,map);
    if(ui.mode.value==="packet")drawWavepacket(ps.main,map);else if(ui.mode.value==="landau")drawLandauGauge(ps.main,map);else drawSymmetric(ps.main,map);drawEnergy(ps.energy);updateUI();
  }

  function tick(now){
    const dt=clamp((now-state.last)/1000,0,.05);state.last=now;
    const ramp=Math.max(.15,Number(ui.ramp.value));const bTarget=state.bOn?Number(ui.b.value):0,eTarget=state.eOn?Number(ui.e.value):0;state.B += clamp(bTarget-state.B,-cfg.field_slider_max_tesla*dt/ramp,cfg.field_slider_max_tesla*dt/ramp);state.E += clamp(eTarget-state.E,-cfg.electric_slider_max_v_per_m*dt/ramp,cfg.electric_slider_max_v_per_m*dt/ramp);
    if(state.running){const normalizedB=state.B/Math.max(.05,Number(ui.b.value));state.phase+=dt*Number(ui.speed.value)*2*Math.PI*.36*clamp(normalizedB,0,2.5);state.freeTime+=dt*Number(ui.speed.value);}
    state.frameCount++;if(now-state.fpsClock>700){state.fps=Math.round(state.frameCount*1000/(now-state.fpsClock));state.frameCount=0;state.fpsClock=now;el("fps-label").textContent=`LIVE · ${state.fps} FPS`;}
    if(state.visible)draw();requestAnimationFrame(tick);
  }

  ui.play.onclick=()=>{state.running=!state.running;updateUI();};ui.reset.onclick=()=>{state.running=true;state.bOn=false;state.eOn=false;state.B=0;state.E=0;state.phase=0;state.freeTime=0;updateUI();};ui.toggleB.onclick=()=>{state.bOn=!state.bOn;};ui.toggleE.onclick=()=>{state.eOn=!state.eOn;};
  el("tutorial-prev").onclick=()=>{if(tutorialIndex>0){tutorialIndex--;renderTutorial();}};
  el("tutorial-next").onclick=()=>{if(tutorialIndex<tutorialSteps.length-1){tutorialIndex++;renderTutorial();}};
  el("tutorial-setup").onclick=prepareTutorialStep;
  for(const input of document.querySelectorAll("input,select"))input.addEventListener("input",updateUI);
  resize();updateUI();renderTutorial();requestAnimationFrame(tick);
})();
</script>
</body>
</html>"""
    return template.replace("__CONFIG__", serialized_configuration)


def render_landau_live_material(
    streamlit: Any,
    parameters: LandauLevelParameters,
) -> None:
    """Embed the live canvas through Streamlit's native iframe surface."""
    streamlit.iframe(
        landau_live_material_html(parameters),
        width="stretch",
        height=1_240,
    )


__all__ = [
    "landau_live_configuration",
    "landau_live_material_html",
    "render_landau_live_material",
]
