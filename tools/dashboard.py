"""ATM Sentinel live dashboard. Reads the SQLite file that mqtt_to_sqlite.py writes.

    pip install flask
    python tools/dashboard.py                      # http://localhost:5000, db atm_sentinel.db
    python tools/dashboard.py --db demo.db --port 8080

One page, polls /api/state every second. No JS libraries; charts are inline SVG.
"""
import argparse, json, sqlite3
from flask import Flask, jsonify, Response

app = Flask(__name__)
DB = "atm_sentinel.db"
STAGE1_THRESHOLD = 0.7615026
STAGE2 = ["COOLING_FAILURE", "HIGH_HUMIDITY", "POWER_FAILURE", "TAMPER", "SENSOR_FAULT"]


def rows(sql, *args):
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in db.execute(sql, args)]
    except sqlite3.OperationalError:  # tables not created yet
        return []
    finally:
        db.close()


@app.get("/api/state")
def state():
    # rowid order, not timestamp: pre-NTP rows carry seconds-since-boot and would sort wrong
    tel = rows("SELECT * FROM telemetry ORDER BY rowid DESC LIMIT 300")[::-1]
    inf = rows("SELECT * FROM inference ORDER BY rowid DESC LIMIT 30")[::-1]
    for r in inf:
        for k in ("stage2", "topz", "features"):
            r[k] = json.loads(r[k])
    st = rows("SELECT json FROM status ORDER BY rowid DESC LIMIT 1")
    return jsonify(telemetry=tel, inference=inf, status=json.loads(st[0]["json"]) if st else None,
                   threshold=STAGE1_THRESHOLD, stage2_labels=STAGE2)


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ATM Sentinel</title>
<style>
:root{color-scheme:light;--surface:#fcfcfb;--card:#ffffff;--line:#e4e3df;--grid:#ecebe7;--text:#0b0b0b;--text2:#52514e;--muted:#8a8985;
  --series:#2a78d6;--pos:#2a78d6;--neg:#e34948;--mid:#f0efec;--good:#0ca30c;--warn:#fab219;--serious:#ec835a;--critical:#d03b3b}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;--surface:#1a1a19;--card:#232322;--line:#383835;--grid:#2c2c2a;--text:#fff;--text2:#c3c2b7;--muted:#8a8985;--series:#3987e5;--pos:#3987e5;--neg:#e66767;--mid:#383835}}
:root[data-theme=dark]{color-scheme:dark;--surface:#1a1a19;--card:#232322;--line:#383835;--grid:#2c2c2a;--text:#fff;--text2:#c3c2b7;--muted:#8a8985;--series:#3987e5;--pos:#3987e5;--neg:#e66767;--mid:#383835}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--surface);color:var(--text);font:14px/1.4 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px}h2{font-size:13px;font-weight:600;color:var(--text2);margin:0 0 6px;text-transform:uppercase;letter-spacing:.04em}
.sub{color:var(--text2);margin-bottom:14px}.grid{display:grid;gap:12px}.kpi{grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}
.sm{grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}.two{grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}
.hero{font-size:44px;font-weight:600;line-height:1.1;letter-spacing:-.01em}.k{color:var(--text2);font-size:12px;margin-top:4px}
.badge{display:inline-flex;align-items:center;gap:8px;font-size:22px;font-weight:600}.dot{width:14px;height:14px;border-radius:50%;flex:none}
.meter{height:10px;background:var(--grid);border-radius:5px;position:relative;margin:10px 0 4px;overflow:hidden}.meter>i{position:absolute;left:0;top:0;bottom:0;background:var(--series);border-radius:5px}
.meter>b{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--text)}
svg{display:block;width:100%;height:auto}.ax{stroke:var(--line);stroke-width:1}.gl{stroke:var(--grid);stroke-width:1}
.ln{fill:none;stroke:var(--series);stroke-width:2;stroke-linejoin:round}.ar{fill:var(--series);opacity:.12}
text{fill:var(--text2);font-size:11px}.lbl{fill:var(--text);font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;color:var(--text2);font-weight:600;border-bottom:1px solid var(--line);padding:6px 8px}td{padding:6px 8px;border-bottom:1px solid var(--grid);white-space:nowrap}
.mono{font-variant-numeric:tabular-nums}.tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:12px;display:none;z-index:9}
section{margin-top:14px}.empty{color:var(--muted);padding:24px;text-align:center}
</style></head><body>
<h1>ATM Sentinel <span id="atm" style="color:var(--text2);font-weight:400"></span></h1>
<div class="sub" id="sub">waiting for data…</div>
<div class="grid kpi">
  <div class="card"><h2>Current state</h2><div class="badge"><span class="dot" id="dot"></span><span id="cls">—</span></div><div class="k" id="clsk"></div></div>
  <div class="card"><h2>P(abnormal)</h2><div class="hero mono" id="p">—</div><div class="meter"><i id="pbar" style="width:0"></i><b id="pthr"></b></div><div class="k">threshold <span id="thr" class="mono"></span> · Stage 1 sigmoid</div></div>
  <div class="card"><h2>Persistence</h2><div class="hero mono" id="persist">—</div><div class="k" id="persistk">consecutive abnormal windows</div></div>
  <div class="card"><h2>Node</h2><div id="node" class="k" style="margin-top:0;font-size:13px;line-height:1.6">—</div></div>
</div>
<section><h2>Last 300 s — the inference window</h2><div class="grid sm" id="sm"></div></section>
<div class="grid two" style="margin-top:14px">
  <div class="card"><h2>Stage 2 — fault probabilities</h2><div id="s2"></div></div>
  <div class="card"><h2>Furthest from training data — top |z|</h2><div id="tz"></div><div class="k">standardised feature value in σ, clipped at ±8</div></div>
</div>
<section><h2>Inference history</h2><div class="card" style="padding:0;overflow:auto"><table><thead><tr><th>time</th><th>p_abnormal</th><th>class</th><th>persist</th><th>alert</th><th>top |z|</th></tr></thead><tbody id="hist"></tbody></table></div></section>
<div class="tip" id="tip"></div>
<script>
const $=id=>document.getElementById(id);
const CH=[["temperature","°C",2],["humidity","% RH",1],["pressure","hPa",1],["light","lux",1],["vibration","edges/s",0],["current","A",3]];
const STATUS={NORMAL:["good","✓"],SENSOR_FAULT:["warn","⚠"],COOLING_FAILURE:["serious","●"],HIGH_HUMIDITY:["serious","●"],POWER_FAILURE:["serious","●"],TAMPER:["critical","■"]};
const fmt=(v,d)=>v==null?"null":Number(v).toFixed(d);
const tsf=t=>t>1e9?new Date(t*1000).toLocaleTimeString():`+${t}s`;
const tip=$("tip");
function showTip(e,html){tip.innerHTML=html;tip.style.display="block";tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY+12)+"px"}
function hideTip(){tip.style.display="none"}

function lineChart(name,unit,dec,data){ // single series: no legend, title names it
  const W=300,H=110,L=40,R=8,T=18,B=18,vals=data.map(r=>r[name]);
  const ok=vals.filter(v=>v!=null);const n=data.length;
  if(!ok.length)return `<div class="card"><h2>${name}</h2><div class="empty">all null</div></div>`;
  let lo=Math.min(...ok),hi=Math.max(...ok);if(hi-lo<1e-9){hi=lo+1;lo=lo-1}
  const x=i=>L+(i/(Math.max(n,300)-1))*(W-L-R),y=v=>T+(1-(v-lo)/(hi-lo))*(H-T-B);
  let path="",cur="";vals.forEach((v,i)=>{if(v==null){if(cur)path+=cur;cur="";return}cur+=(cur?"L":"M")+x(i).toFixed(1)+","+y(v).toFixed(1)+" "});path+=cur;
  const last=ok[ok.length-1],nulls=vals.filter(v=>v==null).length;
  return `<div class="card" data-ch="${name}"><h2>${name} <span style="float:right;color:var(--text);font-weight:600" class="mono">${fmt(last,dec)} ${unit}</span></h2>
  <svg viewBox="0 0 ${W} ${H}" data-lo="${lo}" data-hi="${hi}" data-n="${n}">
   <line class="gl" x1="${L}" x2="${W-R}" y1="${y(hi)}" y2="${y(hi)}"/><line class="gl" x1="${L}" x2="${W-R}" y1="${y((hi+lo)/2)}" y2="${y((hi+lo)/2)}"/>
   <line class="ax" x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}"/>
   <text x="${L-4}" y="${y(hi)+4}" text-anchor="end" class="mono">${fmt(hi,dec)}</text><text x="${L-4}" y="${y(lo)+4}" text-anchor="end" class="mono">${fmt(lo,dec)}</text>
   <text x="${L}" y="${H-4}">-${n}s</text><text x="${W-R}" y="${H-4}" text-anchor="end">now</text>
   <path class="ln" d="${path}"/><line class="cross" x1="0" x2="0" y1="${T}" y2="${H-B}" stroke="var(--text2)" stroke-dasharray="2 2" style="display:none"/>
   <circle r="4" fill="var(--series)" stroke="var(--card)" stroke-width="2" style="display:none"/>
  </svg>${nulls?`<div class="k">${nulls} null sample${nulls>1?"s":""}</div>`:""}</div>`;
}
function hbar(items,dec,signed){ // horizontal bars, thin, rounded data-end, 2px gap
  const W=320,rowH=22,L=signed?130:120,R=44,H=items.length*rowH+4,max=signed?8:1;
  const mid=signed?L+(W-L-R)/2:L,scale=(W-L-R)/(signed?2*max:max);
  return `<svg viewBox="0 0 ${W} ${H}">${signed?`<line class="ax" x1="${mid}" x2="${mid}" y1="0" y2="${H}"/>`:""}${items.map(([k,v],i)=>{
    const y=i*rowH+3,w=Math.abs(v)*scale,x=signed?(v<0?mid-w:mid):L,col=signed?(v<0?"var(--neg)":"var(--pos)"):"var(--series)";
    return `<text x="${L-6}" y="${y+12}" text-anchor="end" class="${signed?"":"lbl"}">${k}</text><rect x="${x}" y="${y+2}" width="${Math.max(w,0)}" height="12" rx="3" fill="${col}"/>
    <text x="${signed?(v<0?x-4:x+w+4):x+w+4}" y="${y+12}" text-anchor="${signed&&v<0?"end":"start"}" class="mono">${signed?(v>0?"+":"")+v.toFixed(1):(v*100).toFixed(0)+"%"}</text>`}).join("")}</svg>`;
}

async function tick(){
  let s;try{s=await (await fetch("/api/state")).json()}catch(e){return}
  const tel=s.telemetry,inf=s.inference,last=inf[inf.length-1],lt=tel[tel.length-1];window._tel=tel;
  $("thr").textContent=s.threshold.toFixed(4);$("pthr").style.left=(s.threshold*100)+"%";
  if(lt){$("atm").textContent=lt.atm_id;$("sub").textContent=`last sample ${tsf(lt.timestamp)} · ${tel.length} samples in window · ${inf.length} inferences stored`}
  if(last){
    const [st,ic]=STATUS[last.class]||["warn","?"];
    $("dot").style.background=`var(--${st})`;$("cls").textContent=`${ic} ${last.class}`;
    $("clsk").textContent=last.fault_mask?`deterministic sensor check fired (mask ${last.fault_mask})`:last.p_abnormal>=s.threshold?"Stage 1 abnormal → Stage 2":"Stage 1 below threshold";
    $("p").textContent=last.p_abnormal==null?"n/a":last.p_abnormal.toFixed(4);$("pbar").style.width=((last.p_abnormal||0)*100)+"%";
    $("persist").innerHTML=`${last.persist}${last.alert?` <span style="color:var(--critical);font-size:22px">■ ALERT</span>`:""}`;
    const s2=last.stage2.every(v=>v==null)?`<div class="empty">Stage 2 not run — ${last.fault_mask?"deterministic fault":"Stage 1 said normal"}</div>`:hbar(s.stage2_labels.map((k,i)=>[k,last.stage2[i]]),2,false);
    $("s2").innerHTML=s2;$("tz").innerHTML=hbar(Object.entries(last.topz).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])),1,true);
  }
  if(s.status){const st=s.status,sn=st.sensors,pv=st.provisional;
    $("node").innerHTML=`${st.ip||""}<br>BME280 ${sn.bme280?"✓ "+sn.bme280_addr:"✗"} · VEML7700 ${sn.veml7700?"✓":"✗"} · INA219 ${sn.ina219?"✓":"✗"}<br>voltage simulated · current ${pv.current_simulated?"simulated":"measured"} · P +${pv.pressure_offset_hpa} hPa<br>arena ${st.models.arena1_used}+${st.models.arena2_used} B`}
  $("sm").innerHTML=CH.map(([n,u,d])=>lineChart(n,u,d,tel)).join("")+motionChart(tel);
  $("hist").innerHTML=inf.slice().reverse().map(r=>{const [st,ic]=STATUS[r.class]||["warn","?"];
    return `<tr><td class="mono">${tsf(r.timestamp)}</td><td class="mono">${r.p_abnormal==null?"n/a":r.p_abnormal.toFixed(4)}</td><td><span class="dot" style="display:inline-block;width:10px;height:10px;vertical-align:-1px;background:var(--${st})"></span> ${r.class}</td><td class="mono">${r.persist}</td><td>${r.alert?"■ ALERT":""}</td><td class="mono">${Object.entries(r.topz).slice(0,3).map(([k,v])=>`${k} ${v>0?"+":""}${v.toFixed(1)}`).join(", ")}</td></tr>`}).join("")||`<tr><td colspan="6" class="empty">first inference arrives 300 s after boot</td></tr>`;
}
function motionChart(tel){ // binary: strip, not a line
  const W=300,H=110,L=40,R=8,n=tel.length,on=tel.filter(r=>r.motion).length;
  const x=i=>L+(i/(Math.max(n,300)-1))*(W-L-R);
  return `<div class="card"><h2>motion <span style="float:right;color:var(--text);font-weight:600" class="mono">${n?Math.round(100*on/n):0}% on</span></h2><svg viewBox="0 0 ${W} ${H}">
  <rect x="${L}" y="40" width="${W-L-R}" height="30" fill="var(--grid)" rx="3"/>${tel.map((r,i)=>r.motion?`<rect x="${x(i)}" y="40" width="${(W-L-R)/300+0.5}" height="30" fill="var(--series)"/>`:"").join("")}
  <text x="${L-4}" y="59" text-anchor="end">PIR</text><text x="${L}" y="${H-4}">-${n}s</text><text x="${W-R}" y="${H-4}" text-anchor="end">now</text></svg></div>`;
}
// crosshair + tooltip on the small multiples
document.addEventListener("mousemove",e=>{const svg=e.target.closest("[data-ch] svg");if(!svg){hideTip();document.querySelectorAll(".cross,circle").forEach(el=>el.style.display="none");return}
  const card=svg.closest("[data-ch]"),name=card.dataset.ch,n=+svg.dataset.n,lo=+svg.dataset.lo,hi=+svg.dataset.hi,pt=svg.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;
  const p=pt.matrixTransform(svg.getScreenCTM().inverse()),i=Math.round((p.x-40)/(300-48)*(Math.max(n,300)-1));
  const tel=window._tel||[];const r=tel[i];if(!r||i<0){hideTip();return}
  const v=r[name],x=40+(i/(Math.max(n,300)-1))*(300-48),y=v==null?null:18+(1-(v-lo)/(hi-lo))*(110-36);
  const c=svg.querySelector(".cross"),d=svg.querySelector("circle");c.setAttribute("x1",x);c.setAttribute("x2",x);c.style.display="";
  if(y!=null){d.setAttribute("cx",x);d.setAttribute("cy",y);d.style.display=""}else d.style.display="none";
  showTip(e,`<b>${name}</b> ${v==null?"null":v}<br><span style="color:var(--text2)">${tsf(r.timestamp)}</span>`)});
tick();setInterval(tick,1000);
</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    DB = args.db
    app.run(host="0.0.0.0", port=args.port, debug=False)
