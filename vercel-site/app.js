const $=id=>document.getElementById(id),A=$('app');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const N=x=>{const n=parseFloat(x);return Number.isFinite(n)?n:null},f1=x=>x==null?'N/A':Number(x).toFixed(1),f2=x=>x==null?'N/A':Number(x).toFixed(2);
const COORDS={"2BKLG005":[3.232191,101.750134],"2BKLG006":[3.206971,101.755936],"2BKLG016":[3.016563,101.376878],"2BKLG022":[3.045897,101.446625],"2BKLG023":[3.042362,101.472604],"2BKLG024":[3.029567,101.550903],"2BKLG034":[3.020628,101.596586],"2BKLG050":[3.038944,101.534036],"2WKLG001":[3.082549,101.664382],"2WKLG002":[3.138222,101.694807],"2WKLG003":[3.155487,101.701133],"2WKLG004":[3.12083,101.676438],"2WKLG049":[3.164633,101.73497],"3JSKU001":[1.500322,103.683419],"3JSKU002":[1.519387,103.677122],"3JSKU003":[1.555154,103.660858],"3JSKU004":[1.684808,103.572752],"3JSKU005":[1.662678,103.604485],"3JSKU006":[1.63204,103.63651],"3JSKU007":[1.62168,103.655247],"3JSKU010":[1.668734,103.589723],"3JSKU011":[1.692786,103.514711]};
const cat=v=>v==null?'No data':v>=81?'Clean':v>=60?'Slightly polluted':'Polluted';
const SYM={'Clean':'●','Slightly polluted':'◆','Polluted':'▲','No data':'○'},COL={'Clean':'#1f6fb5','Slightly polluted':'#8a5d00','Polluted':'#b3261e','No data':'#6b7b82'};
const badge=c=>`<span class="st ${c.split(' ')[0]}">${SYM[c]} ${c}</span>`;
const PAR=[['DO','Dissolved oxygen (mg/L)'],['BOD','BOD (mg/L)'],['COD','COD (mg/L)'],['SS','Suspended solids (mg/L)'],['pH','pH'],['NH3N','NH3-N (mg/L)']];
const RIV={ALL:'Both rivers',KLANG:'Sungai Klang',SKUDAI:'Sungai Skudai'};
let H=[],F=[],M=[],ST=[],ready=false;
const S={river:'ALL',station:'',from:'',to:'',model:'Persistence'};   // applied selections, kept between pages
let D={...S};                                                       // draft values, only applied on Apply
const hist=(extra={})=>H.filter(r=>(S.river==='ALL'||r.river===S.river)&&(!S.from||r.date>=S.from)&&(!S.to||r.date<=S.to)&&(!extra.id||r.station_id===extra.id));
const latest=rows=>{const m={};rows.forEach(r=>{if(!m[r.station_id]||r.date>m[r.station_id].date)m[r.station_id]=r});return m};
const stName=id=>{const s=ST.find(x=>x.id===id);return s?id+' · '+s.location:id};
const table=(rows,cols,id)=>`<div class="tw" tabindex="0" role="region" aria-label="Data table, scrolls inside this region"><table><thead><tr>${cols.map(c=>`<th scope="col">${c[1]}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr ${id?`data-id="${esc(r[id])}"`:''}>${cols.map(c=>`<td>${c[2]?c[2](r[c[0]],r):esc(r[c[0]]??'')}</td>`).join('')}</tr>`).join('')||`<tr><td colspan="${cols.length}">No records in this period.</td></tr>`}</tbody></table></div>`;
const qual=v=>v===''||v==null?'—':esc(v);
function bar(opts={}){const st=opts.station;return `<form class="filters" id="fl" aria-label="Filters"><label>River<select id="fr">${Object.entries(RIV).map(([k,v])=>`<option value="${k}" ${D.river===k?'selected':''}>${v}</option>`).join('')}</select></label>
${st?`<label>Station<select id="fs">${ST.filter(s=>D.river==='ALL'||s.river===D.river).map(s=>`<option value="${s.id}" ${D.station===s.id?'selected':''}>${esc(s.id+' · '+s.location)}</option>`).join('')}</select></label>`:''}
<label>From<input type="date" id="ff" value="${D.from}" min="${ST.mind}" max="${ST.maxd}"></label><label>To<input type="date" id="ft" value="${D.to}" min="${ST.mind}" max="${ST.maxd}"></label>
<button type="submit">Apply</button><button type="button" class="sec" id="rs">Reset</button></form>`}
function wire(go){const fl=$('fl');if(!fl)return;
 $('fr').onchange=e=>{D.river=e.target.value;if($('fs')){go(false,true)}};
 fl.onsubmit=e=>{e.preventDefault();D.river=$('fr').value;D.from=$('ff').value;D.to=$('ft').value;if($('fs'))D.station=$('fs').value;Object.assign(S,D);go(true)};
 $('rs').onclick=()=>{D={...S,river:'ALL',from:'',to:''};Object.assign(S,D);go(true)}}
const LAY=(o={})=>({margin:{t:16,l:56,r:16,b:56},paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'Arial',size:14,color:'#173D45'},legend:{orientation:'h',y:-.25},xaxis:{type:'date',title:'Sampling date',gridcolor:'#e3eaed'},yaxis:{gridcolor:'#e3eaed'},...o});
const CFG={responsive:true,displayModeBar:false};
function map(el,pts,sel,onclick){const lat=pts.map(p=>COORDS[p.id][0]),lon=pts.map(p=>COORDS[p.id][1]);if(!lat.length)return;
 const sp=a=>Math.max(Math.max(...a)-Math.min(...a),.05),w=$(el).clientWidth||600,z=Math.min(11,Math.log2(Math.min(360*w/(256*sp(lon)*1.4),180*440/(256*sp(lat)*1.5))));
 Plotly.react(el,[{type:'scattermapbox',mode:'markers',lat,lon,customdata:pts.map(p=>p.id),hoverinfo:'text',text:pts.map(p=>`<b>${esc(p.name)}</b><br>${esc(p.tip)}`),marker:{size:pts.map(p=>p.id===sel?20:14),color:pts.map(p=>COL[p.cat]),opacity:1}}],
 {margin:{t:0,l:0,r:0,b:0},showlegend:false,mapbox:{style:'open-street-map',center:{lat:(Math.min(...lat)+Math.max(...lat))/2,lon:(Math.min(...lon)+Math.max(...lon))/2},zoom:z}},CFG);
 if(onclick)$(el).on('plotly_click',e=>onclick(e.points[0].customdata))}
const legend=`<p class="cap">Markers: ${Object.keys(SYM).map(k=>`<span style="color:${COL[k]}">${SYM[k]}</span> ${k}`).join(' · ')}. Colour is always paired with a label. The map shows station locations only; it does not infer river paths.</p>`;
const periodNote=()=>`<p class="cap">Applied: ${RIV[S.river]} · ${S.from||ST.mind} to ${S.to||ST.maxd}. Dates are actual irregular sampling dates.</p>`;

/* ---------- Overview ---------- */
function overview(){
 const L=latest(hist()),rows=Object.values(L).sort((a,b)=>a.station_id<b.station_id?-1:1),sel=S.station;
 const n=hist().length;
 A.innerHTML=`<h1>Overview</h1><p class="sub">Choose a river and period, then pick a station. Each station shows its latest record in the applied period.</p>${bar()}
 <div class="cards"><div class="card">Stations with records<b>${rows.length}</b></div><div class="card">Observations in period<b>${n}</b></div><div class="card">Data period<b style="font-size:20px">${ST.mind} to ${ST.maxd}</b></div><div class="card">Source<b style="font-size:20px">DOE Malaysia</b></div></div>
 ${n?'':'<div class="note">No observations in this period. <button class="sec" id="rs2">Reset</button></div>'}
 <div class="two"><section class="panel"><h2>Station map</h2><div class="map" id="mp" role="img" aria-label="Map of stations coloured by the status of each latest record"></div>${legend}</section>
 <section class="panel"><h2>Latest record per station</h2>${periodNote()}<div id="tb">${table(rows.map(r=>({...r,c:cat(N(r.wqi))})),[['station_id','Station'],['location','Location'],['date','Sampling date'],['wqi','WQI',v=>f1(N(v))],['c','Status',v=>badge(v)]],'station_id')}</div>
 <div class="row" style="margin-top:16px"><a class="btn sec" href="#stations">Explore stations</a></div><div id="sm"></div></section></div>`;
 const pts=rows.map(r=>({id:r.station_id,name:r.location,cat:cat(N(r.wqi)),tip:`${r.date} · WQI ${f1(N(r.wqi))} · ${cat(N(r.wqi))}`}));
 const pick=id=>{S.station=D.station=id;const r=L[id];document.querySelectorAll('#tb tr').forEach(t=>t.classList.toggle('sel',t.dataset.id===id));
  $('sm').innerHTML=`<div class="note"><b>${esc(stName(id))}</b><br>Sampling date ${r.date} · WQI ${f1(N(r.wqi))} · ${badge(cat(N(r.wqi)))}<br>Source status: ${esc(r.river_status||'—')} (DOE CLASS ${esc(r.class||'—')}).<br><a class="btn" style="margin-top:8px" href="#station:${id}">View station details</a></div>`;map('mp',pts,id,pick)};
 map('mp',pts,sel,pick);document.querySelectorAll('#tb tr[data-id]').forEach(t=>t.onclick=()=>pick(t.dataset.id));if(L[sel])pick(sel);
 wire(overview);if($('rs2'))$('rs2').onclick=()=>$('rs').click()}

/* ---------- Explore stations ---------- */
let CHK=new Set(),QRY='',VIEW='chart',PRM='wqi';
function stations(){
 const ids=ST.filter(s=>(S.river==='ALL'||s.river===S.river)&&(s.id+s.location).toLowerCase().includes(QRY.toLowerCase()));
 A.innerHTML=`<h1>Explore stations</h1><p class="sub">Find stations and compare their historical measurements (two to four stations).</p>${bar()}
 <div class="two"><section class="panel"><h2>Stations</h2><div class="row"><label>Search<input type="search" id="q" value="${esc(QRY)}"></label><button class="sec" id="cq">Clear search</button></div>
 <div class="tw" style="max-height:380px"><table><thead><tr><th>Select</th><th>Station</th><th>Location</th><th>Records</th></tr></thead><tbody>${ids.map(s=>`<tr><td><input type="checkbox" class="ck" value="${s.id}" aria-label="Select ${esc(s.id)}" ${CHK.has(s.id)?'checked':''}></td><td>${s.id}</td><td>${esc(s.location)}</td><td>${hist({id:s.id}).length}</td></tr>`).join('')||'<tr><td colspan="4">No stations match. Use Clear search.</td></tr>'}</tbody></table></div>
 <div class="row" style="margin-top:16px"><button id="cmp">Compare selected</button><span class="cap" id="cn"></span></div></section>
 <section class="panel"><h2>Comparison</h2><div class="row"><label>Measure<select id="pm"><option value="wqi">WQI</option>${PAR.map(p=>`<option value="${p[0]}" ${PRM===p[0]?'selected':''}>${p[1]}</option>`).join('')}</select></label>
 <span class="tabs"><button class="${VIEW==='chart'?'':'sec'}" data-v="chart">Chart</button> <button class="${VIEW==='table'?'':'sec'}" data-v="table">Table</button></span></div><div id="cmpv"></div></section></div>`;
 const upd=()=>{$('cn').textContent=CHK.size+' selected';$('cmp').disabled=CHK.size<2||CHK.size>4};upd();
 document.querySelectorAll('.ck').forEach(c=>c.onchange=()=>{c.checked?CHK.add(c.value):CHK.delete(c.value);upd()});
 $('q').oninput=e=>{QRY=e.target.value;stations()};$('cq').onclick=()=>{QRY='';stations()};
 document.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>{VIEW=b.dataset.v;stations();draw()});
 const draw=()=>{PRM=$('pm').value;const sel=[...CHK];const box=$('cmpv');
  if(sel.length<2){box.innerHTML='<p>Select at least two to compare.</p>';return}
  const un=PRM==='wqi'?'WQI':PAR.find(p=>p[0]===PRM)[1],key=PRM==='wqi'?'wqi':PRM;
  if(VIEW==='table'){const rows=sel.flatMap(id=>hist({id}).map(r=>({...r})));box.innerHTML=periodNote()+table(rows,[['station_id','Station'],['date','Date'],[key,un,qual]]);return}
  box.innerHTML=`<div class="chart" id="cc"></div>${periodNote()}<p class="cap">Counts: ${sel.map(id=>id+' = '+hist({id}).length).join(', ')} records. Missing values are gaps, not zeros.</p>`;
  Plotly.react('cc',sel.map(id=>{const h=hist({id}).filter(r=>N(r[key])!=null);return{type:'scatter',mode:'lines+markers',name:id,x:h.map(r=>r.date),y:h.map(r=>N(r[key]))}}),LAY({yaxis:{title:un,gridcolor:'#e3eaed'}}),CFG)};
 $('cmp').onclick=draw;$('pm').onchange=draw;if(CHK.size>=2)draw();else $('cmpv').innerHTML='<p>Select at least two to compare.</p>';wire(stations)}

/* ---------- Station details ---------- */
function station(id){S.station=D.station=id;const h=hist({id}),s=ST.find(x=>x.id===id);
 if(!s){A.innerHTML='<h1>Station details</h1><p>Station not found.</p>';return}
 const l=h[h.length-1];
 A.innerHTML=`<p><a href="#overview">← Back</a></p><h1>${esc(s.id)} · ${esc(s.location)}</h1><p class="sub">${RIV[s.river]}</p>${bar({station:1})}
 ${l?`<div class="cards"><div class="card">Latest record in period<b style="font-size:20px">${l.date}</b></div><div class="card">WQI<b>${f1(N(l.wqi))}</b></div><div class="card">Status<b style="font-size:20px">${badge(cat(N(l.wqi)))}</b></div><div class="card">Records in period<b>${h.length}</b></div></div>`:'<div class="note">No records for this station in the applied period.</div>'}
 <details><summary>Metadata and provenance</summary><p class="cap">Coordinates: ${COORDS[id].join(', ')} (placement and reference system pending verification). Source: DOE Malaysia workbook, Lampiran A – Klang &amp; Skudai. Supplied CLASS: ${esc(l?.class||'—')}; RIVER STATUS: ${esc(l?.river_status||'—')}.</p></details>
 <section class="panel" style="margin-top:24px"><h2>WQI over time</h2><div class="chart" id="cw"></div><p class="cap">Sampling is irregular; lines join actual sampling dates only.</p>
 <div class="row"><label>Add parameter chart<select id="pp"><option value="">None</option>${PAR.map(p=>`<option value="${p[0]}">${p[1]}</option>`).join('')}</select></label></div><div class="chart" id="cp" hidden></div></section>
 <section class="panel"><h2>Observations</h2>${table(h,[['date','Date'],['wqi','WQI',v=>f1(N(v))],...PAR.map(p=>[p[0],p[1],qual]),['river_status','Source status']])}<p class="cap">Missing values are not zero. Qualifiers such as &lt;1.000 are kept as supplied.</p></section>
 <a class="btn" href="#forecasts">View forecast</a>`;
 Plotly.react('cw',[{type:'scatter',mode:'lines+markers',name:'Observed WQI',x:h.map(r=>r.date),y:h.map(r=>N(r.wqi))}],LAY({yaxis:{title:'WQI',range:[0,105],gridcolor:'#e3eaed'}}),CFG);
 $('pp').onchange=e=>{const k=e.target.value,c=$('cp');c.hidden=!k;if(!k)return;Plotly.react('cp',[{type:'scatter',mode:'lines+markers',name:k,x:h.map(r=>r.date),y:h.map(r=>N(r[k]))}],LAY({yaxis:{title:PAR.find(p=>p[0]===k)[1],gridcolor:'#e3eaed'}}),CFG)};
 wire(()=>{const v=$('fs').value;location.hash='station:'+v;route()})}

/* ---------- Forecasts ---------- */
function forecasts(){
 const mods=[...new Set(F.map(r=>r.model))],id=S.station||ST[0].id;D.station=id;
 const f=F.find(r=>r.station_id===id&&r.model===S.model),h=hist({id});
 A.innerHTML=`<h1>Forecasts</h1><p class="sub">An issued next-sampling estimate is not an observed measurement.</p>${bar({station:1})}
 <div class="row"><label>Model<select id="fm">${mods.map(m=>`<option ${m===S.model?'selected':''}>${esc(m)}</option>`).join('')}</select></label></div>
 ${f?`<div class="cards"><div class="card">WQI estimate<b>${f1(N(f.predicted_wqi))}</b><span class="cap">Next-sampling horizon</span></div><div class="card">Derived category<b style="font-size:20px">${badge(cat(N(f.predicted_wqi)))}</b><span class="cap">Boundaries 81 / 60; rule pending verification</span></div><div class="card">Input cutoff<b style="font-size:20px">${f.last_observation_date}</b></div><div class="card">Model<b style="font-size:20px">${esc(f.model)}</b></div></div>`:'<div class="note">Forecast unavailable for this station and model.</div>'}
 <section class="panel"><h2>${esc(stName(id))}</h2><div class="chart" id="fc"></div><p class="cap">Solid line: observed. Dashed line and diamond: estimated. The estimate is placed at the model's expected next sampling date, which is not a confirmed sample date. No confidence interval is available.</p>
 <details><summary>Model details</summary><p class="cap">Saved model outputs from the project pipeline; training cutoff and versions are not recorded in this static copy. <a href="#evaluation">View evaluation</a></p></details></section>`;
 if(f)Plotly.react('fc',[{type:'scatter',mode:'lines+markers',name:'Observed',x:h.map(r=>r.date),y:h.map(r=>N(r.wqi)),line:{color:'#173D45'}},
  {type:'scatter',mode:'lines+markers',name:'Estimated',x:[f.last_observation_date,f.forecast_date.slice(0,10)],y:[N(f.last_observed_wqi),N(f.predicted_wqi)],line:{dash:'dash',color:'#167E87'},marker:{symbol:'diamond',size:11}}],LAY({yaxis:{title:'WQI',range:[0,105],gridcolor:'#e3eaed'}}),CFG);
 $('fm').onchange=e=>{S.model=e.target.value;forecasts()};wire(forecasts)}

/* ---------- Model evaluation ---------- */
function evaluation(){
 A.innerHTML=`<h1>Model evaluation</h1><p class="sub">Saved metrics only. No model is declared a winner.</p>
 <div class="note">Models were scored on different numbers of test targets, so these results are not directly comparable. Treat them as indicative.</div>
 <section class="panel"><h2>Metrics by model</h2>${table(M,[['model','Model'],['n_samples','Targets (n)'],['MAE','MAE (WQI)',f2],['RMSE','RMSE (WQI)',f2],['R2','R²',f2],['accuracy','Category accuracy',f2],['balanced_accuracy','Balanced accuracy',f2],['macro_f1','Macro F1',f2]])}
 <p class="cap">N/A means not produced; it does not mean zero. Categories are derived from predicted WQI.</p></section>
 <section class="panel"><h2>Actual versus predicted</h2><div class="row"><label>Model<select id="em">${M.map(m=>`<option>${esc(m.model)}</option>`).join('')}</select></label></div><div class="chart" id="ev"></div><p class="cap">Historical out-of-sample records, not live readings.</p></section>
 <details><summary>Protocol</summary><p class="cap">Twelve observations per station is a limitation. Split cutoffs, feature policy and the final-test protection are being frozen; until then validation and final-test results are not separated in this view.</p></details>`;
 const P=window._P,draw=()=>{const m=$('em').value,h=P.filter(r=>r.model===m);Plotly.react('ev',[{type:'scatter',mode:'markers',name:m,x:h.map(r=>N(r.actual_wqi)),y:h.map(r=>N(r.predicted_wqi)),text:h.map(r=>r.station_id)},{type:'scatter',mode:'lines',name:'Perfect',x:[0,100],y:[0,100],line:{dash:'dash',color:'#8aa0a8'}}],LAY({xaxis:{title:'Actual WQI',range:[0,105],gridcolor:'#e3eaed'},yaxis:{title:'Predicted WQI',range:[0,105],gridcolor:'#e3eaed'}}),CFG)};
 $('em').onchange=draw;draw()}

/* ---------- About ---------- */
function about(){A.innerHTML=`<h1>About the data</h1><p class="sub">Source, coverage, definitions and interpretation limits.</p>
 <section class="panel"><h2>Source and coverage</h2><p>Department of Environment Malaysia (DOE/JAS) sampling records, 2023–2024: ${H.length} observations at ${ST.length} stations (Klang ${ST.filter(s=>s.river==='KLANG').length}, Skudai ${ST.filter(s=>s.river==='SKUDAI').length}). Records are historical, not a live sensor feed. Sampling intervals vary.</p></section>
 <details open><summary>Definitions</summary><p>WQI is the DOE Water Quality Index. DO, BOD, COD, SS and NH3-N are in mg/L. A value such as &lt;1.000 is below the detection limit, not zero. CLASS and RIVER STATUS are kept as supplied.</p></details>
 <details><summary>Data limitations</summary><p>Twelve observations per station and irregular sampling limit forecasting. Station coordinates and map placement still need verification.</p></details>
 <details><summary>Observations, forecasts and interpretation</summary><p>Observations are measured. Forecasts are model estimates for the next sampling, not confirmed readings or official warnings. Use only within the permitted purpose agreed with DOE.</p></details>`}

function route(){const h=location.hash.slice(1)||'overview',[r,id]=h.split(':');
 document.querySelectorAll('.topnav a').forEach(a=>a.classList.toggle('on',a.dataset.r===(r==='station'?'stations':r)));
 if(!ready)return;({overview,stations,forecasts,evaluation,about,station:()=>station(id)}[r]||overview)();scrollTo(0,0)}
addEventListener('hashchange',route);
addEventListener('load',async()=>{try{const g=async f=>Papa.parse(await(await fetch('/data/'+f)).text(),{header:true,skipEmptyLines:true}).data;
 [H,F,M,window._P]=await Promise.all(['station_history.csv','all_models_latest_forecasts.csv','all_models_metrics.csv','all_models_wqi_predictions.csv'].map(g));
 const m={};H.forEach(r=>m[r.station_id]??={id:r.station_id,river:r.river,location:r.location});ST=Object.values(m).sort((a,b)=>a.id<b.id?-1:1);
 const ds=H.map(r=>r.date).sort();ST.mind=ds[0];ST.maxd=ds.at(-1);ready=true;route()}catch(e){A.innerHTML='<div class="note">Data could not be loaded ('+esc(e.message)+'). Your selections are kept. <button onclick="location.reload()">Retry</button></div>'}});
