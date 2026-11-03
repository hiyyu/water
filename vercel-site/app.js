const $ = id => document.getElementById(id);
let forecasts=[], metrics=[], history=[], observations=[];
const safe = x => String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = x => {const n=Number(x);return Number.isFinite(n)?n:null};
const fmt = x => x===null||x===undefined||x===''?'—':Number(x).toFixed(2);
const dateOnly = s => String(s||'').slice(0,10);
const unique = arr => [...new Set(arr.filter(Boolean))].sort();
const statusColor = s => ({Clean:'#1f78c8','Slightly Polluted':'#b7820a',Polluted:'#c23b32'}[s]||'#94a3b8');
const wqiStatus = v => v===null?null:v>=81?'Clean':v>=60?'Slightly Polluted':'Polluted';
const wqiColor = v => statusColor(wqiStatus(v));
const bandShapes = () => [[0,60,'rgba(194,59,50,.09)'],[60,81,'rgba(183,130,10,.11)'],[81,105,'rgba(31,120,200,.09)']].map(b=>({type:'rect',xref:'paper',x0:0,x1:1,yref:'y',y0:b[0],y1:b[1],fillcolor:b[2],line:{width:0},layer:'below'}));
const graphLayout = (title='') => ({title:{text:title,font:{size:13}},margin:{t:35,l:50,r:20,b:75},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:'Instrument Sans, system-ui',color:'#5b7482'},legend:{orientation:'h',y:-.28},xaxis:{gridcolor:'rgba(15,46,61,.08)',zeroline:false},yaxis:{gridcolor:'rgba(15,46,61,.08)',zeroline:false,range:[0,105]}});
async function csv(file){const r=await fetch('/data/'+file);if(!r.ok)throw Error(file+' returned '+r.status);const t=await r.text();return Papa.parse(t,{header:true,skipEmptyLines:true}).data}
function table(id,rows,cols){$(id).innerHTML='<thead><tr>'+cols.map(c=>'<th>'+safe(c[1])+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(c=>'<td>'+safe(c[2]?c[2](r[c[0]],r):r[c[0]])+'</td>').join('')+'</tr>').join('')+'</tbody>';pills(id)}
function pills(id){document.querySelectorAll('#'+id+' td').forEach(td=>{const k={'Clean':'clean','Slightly Polluted':'slight','Polluted':'polluted'}[td.textContent.trim()];if(k){td.innerHTML='<span class="pill '+k+'">'+td.textContent+'</span>'}})}
function options(id,items,first){const previous=$(id).value;$(id).innerHTML=(first||'')+items.map(x=>'<option value="'+safe(x)+'">'+safe(x)+'</option>').join('');if(items.includes(previous)||previous==='All')$(id).value=previous}
function selected(){return {river:$('river').value,station:$('station').value,model:$('model').value}}
function filtered(data,model=true){const s=selected();return data.filter(r=>(!model||r.model===s.model)&&(s.river==='All'||r.river===s.river)&&(s.station==='All'||r.station_id===s.station))}
function refreshStations(){const r=$('river').value;options('station',unique(forecasts.filter(x=>r==='All'||x.river===r).map(x=>x.station_id)),'<option value="All">All stations</option>')}
function render(){const s=selected(),f=filtered(forecasts),o=filtered(observations,false);const mean=(rows,key)=>{const a=rows.map(x=>num(x[key])).filter(x=>x!==null);return a.length?a.reduce((x,y)=>x+y,0)/a.length:null};
$('today').textContent=new Date().toLocaleDateString('en-MY',{day:'numeric',month:'short',year:'numeric'});
$('observed').textContent=fmt(mean(f,'last_observed_wqi'));
$('predicted').textContent=fmt(mean(f,'predicted_wqi'));
const statuses=f.reduce((a,x)=>(a[x.predicted_status]=(a[x.predicted_status]||0)+1,a),{});const dominant=Object.entries(statuses).sort((a,b)=>b[1]-a[1])[0];$('status').textContent=dominant?dominant[0]:'—';$('status').style.color=statusColor(dominant?.[0]);
$('observed-date').textContent='Last observed: '+(f.map(x=>dateOnly(x.last_observation_date)).sort().at(-1)||'—');$('predicted-date').textContent='Next sampling estimate: '+(f.map(x=>dateOnly(x.forecast_date)).sort().at(-1)||'—');
$('freshness').textContent='Today is the display date, not the date of a new DOE reading. Last historical observations are from 2024. Forecasts are saved next-sampling outputs, not verified current water quality.';
const labels=f.map(x=>x.station_id+' ('+x.river+')');Plotly.react('compare-chart',[{type:'bar',name:'Last observed (faded)',x:labels,y:f.map(x=>num(x.last_observed_wqi)),text:f.map(x=>x.last_observed_status),hovertemplate:'%{x}<br>Observed WQI %{y:.1f} · %{text}<extra></extra>',marker:{color:f.map(x=>statusColor(x.last_observed_status)),opacity:.4}},{type:'bar',name:'Next-sampling estimate (solid)',x:labels,y:f.map(x=>num(x.predicted_wqi)),text:f.map(x=>x.predicted_status),hovertemplate:'%{x}<br>Forecast WQI %{y:.1f} · %{text}<extra></extra>',marker:{color:f.map(x=>statusColor(x.predicted_status))}}],{...graphLayout(),barmode:'group'},{responsive:true,displaylogo:false});
const baseline=mean(f,'predicted_wqi');const days=Array.from({length:7},(_,i)=>{const d=new Date();d.setDate(d.getDate()+i);return d.toISOString().slice(0,10)});Plotly.react('outlook-chart',[{type:'scatter',mode:'lines+markers',x:days,y:days.map(()=>baseline),name:'Carried-forward estimate',line:{color:wqiColor(baseline),dash:'dash',width:3},marker:{size:8,color:wqiColor(baseline)}}],{...graphLayout(),yaxis:{range:[0,105],title:'WQI'},annotations:[{text:'Illustrative carry-forward — not a daily ML forecast',xref:'paper',yref:'paper',x:.5,y:1.1,showarrow:false,font:{size:11,color:'#5b7482'}}]},{responsive:true,displaylogo:false});
table('forecast-table',f,[['river','River'],['station_id','Station'],['station_location','Location'],['last_observation_date','Last observation',dateOnly],['forecast_date','Forecast date',dateOnly],['last_observed_wqi','Observed WQI',fmt],['predicted_wqi','Forecast WQI',fmt],['predicted_status','Estimated status']]);
const m=metrics.filter(x=>s.model===x.model);table('metrics-table',m.length?m:metrics,[['model','Model'],['n_samples','Samples'],['MAE','MAE',fmt],['RMSE','RMSE',fmt],['R2','R²',fmt],['accuracy','Accuracy',fmt],['balanced_accuracy','Balanced accuracy',fmt],['macro_f1','Macro F1',fmt]]);
const h=filtered(history).slice(0,450);Plotly.react('evaluation-chart',[{type:'scatter',mode:'markers',x:h.map(x=>num(x.actual_wqi)),y:h.map(x=>num(x.predicted_wqi)),text:h.map(x=>x.station_id),name:s.model+' (coloured by actual status)',marker:{color:h.map(x=>wqiColor(num(x.actual_wqi))),size:8,opacity:.7}},{type:'scatter',mode:'lines',x:[0,100],y:[0,100],name:'Perfect prediction',line:{dash:'dash',color:'#94a3b8'}}],{...graphLayout(),xaxis:{title:'Actual WQI',range:[0,105]},yaxis:{title:'Predicted WQI',range:[0,105]}},{responsive:true,displaylogo:false});
}
async function init(){try{[forecasts,metrics,history,observations]=await Promise.all(['all_models_latest_forecasts.csv','all_models_metrics.csv','all_models_wqi_predictions.csv','latest_station_observations.csv'].map(csv));options('river',unique(forecasts.map(x=>x.river)),'<option value="All">Both rivers</option>');options('model',unique(forecasts.map(x=>x.model)));refreshStations();['river','station','model'].forEach(id=>$(id).addEventListener('change',()=>{if(id==='river')refreshStations();render()}));document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===b));const t=b.dataset.tab;['forecast','evaluation','live'].forEach(id=>$(id).hidden=id!==t);$('filters').style.display=t==='live'?'none':'';setTimeout(t==='live'?renderLive:render,20)}));render();bindLive();loadLive();setInterval(loadLive,LIVE_REFRESH_MS)}catch(e){$('freshness').textContent='Unable to load dashboard data: '+e.message;console.error(e)}}

/* ===== Live monitoring tab (DOE continuous stations, hourly) =====
   Data files are written by scripts/fetch_live_wqi.py via GitHub Actions. */
let liveLatest=[], liveHistory=[], liveMeta=null, liveMapBound=false;
const LIVE_REFRESH_MS=10*60*1000;   // re-read the data files every 10 minutes
const LIVE_STALE_HOURS=6;           // older than this = shown as stale
const liveStatuses=['Clean','Slightly Polluted','Polluted'];
async function liveFetch(file,asJson){
  const r=await fetch('/data/'+file+'?t='+Date.now(),{cache:'no-store'});
  if(!r.ok)throw Error(file+' returned '+r.status);
  return asJson?r.json():Papa.parse(await r.text(),{header:true,skipEmptyLines:true}).data;
}
// DOE timestamps are Malaysia local time (UTC+8)
const liveAgeHours=s=>(Date.now()-new Date(String(s)+'+08:00').getTime())/36e5;
const liveIsStale=s=>!(liveAgeHours(s)<=LIVE_STALE_HOURS);
const liveTime=d=>d.toLocaleString('en-MY',{timeZone:'Asia/Kuala_Lumpur',day:'numeric',month:'short',hour:'2-digit',minute:'2-digit',hour12:false});
function liveRows(){const st=$('live-state').value;return liveLatest.filter(r=>st==='All'||r.state===st)}
function liveSelectOptions(id,pairs,firstAll){
  const el=$(id),prev=el.value;
  el.innerHTML=(firstAll?'<option value="All">All states</option>':'')+pairs.map(p=>'<option value="'+safe(p[0])+'">'+safe(p[1])+'</option>').join('');
  if([...el.options].some(o=>o.value===prev))el.value=prev;
}
function refreshLiveStations(preferred){
  const rows=liveRows(),prev=$('live-station').value;
  liveSelectOptions('live-station',rows.map(r=>[r.station_id,r.station_id+' · '+r.river+' ('+r.state+')']),false);
  if(preferred&&rows.some(r=>r.station_id===preferred))$('live-station').value=preferred;
  else if(!prev||![...$('live-station').options].some(o=>o.value===prev)){
    const fav=rows.find(r=>/skudai|klang/i.test(r.river));  // your two study rivers, if present
    if(fav)$('live-station').value=fav.station_id;
  }
}
function bindLive(){
  $('live-state').addEventListener('change',()=>{refreshLiveStations();renderLive()});
  $('live-station').addEventListener('change',renderLive);
}
async function loadLive(){
  try{
    const [l,h,m]=await Promise.all([liveFetch('live_latest.csv'),liveFetch('live_history.csv'),liveFetch('live_meta.json',true)]);
    liveLatest=l;liveHistory=h;liveMeta=m;
    liveSelectOptions('live-state',unique(l.map(r=>r.state)).map(s=>[s,s]),true);
    refreshLiveStations();
    updateWaterMood();
    if(!$('live').hidden)renderLive();
  }catch(e){
    $('live-note').textContent='Live data unavailable: '+e.message+'. The rest of the dashboard is unaffected.';
    console.error(e);
  }
}
function updateWaterMood(){
  if(!window.WaterBG||!liveLatest.length)return;
  const rows=liveRows(),fresh=rows.filter(r=>!liveIsStale(r.datetime)),use=fresh.length?fresh:rows;
  const vals=use.map(r=>num(r.wqi)).filter(v=>v!==null);
  if(!vals.length){WaterBG.setScore(null,'');return}
  const avg=vals.reduce((a,b)=>a+b,0)/vals.length;
  const scope=$('live-state').value==='All'?'all '+unique(rows.map(r=>r.state)).length+' states':$('live-state').value;
  WaterBG.setScore(avg,scope+(fresh.length?'':' (stale data)'));
}
function renderLive(){
  if(!liveLatest.length)return;
  updateWaterMood();
  const rows=liveRows(),sel=$('live-station').value;
  const checked=liveMeta&&liveMeta.fetched_at?liveTime(new Date(liveMeta.fetched_at)):'—';
  const newest=liveMeta&&liveMeta.newest_reading?liveTime(new Date(liveMeta.newest_reading+'+08:00')):'—';
  $('live-note').textContent='Source: DOE continuous water quality monitoring (MyEQMS), hourly WQI from automatic river-intake stations. These are different stations from the sampling stations used by the forecasting models and are not used for training. Last checked: '+checked+' · Newest reading: '+newest+' (Malaysia time).';
  // map (OpenStreetMap tiles, no API token needed)
  const faded=(hex,al)=>{const n=parseInt(hex.slice(1),16);return 'rgba('+(n>>16)+','+((n>>8)&255)+','+(n&255)+','+al+')'};
  const pts=rows.filter(r=>num(r.lat)!==null&&num(r.lon)!==null);
  const lats=pts.map(r=>num(r.lat)),lons=pts.map(r=>num(r.lon));
  const mid=v=>v.length?(Math.min(...v)+Math.max(...v))/2:0;
  const mapEl=$('live-map'),mw=mapEl.clientWidth||800,mh=mapEl.clientHeight||460;
  const lonSpan=Math.max(lons.length?Math.max(...lons)-Math.min(...lons):0,.05),latSpan=Math.max(lats.length?Math.max(...lats)-Math.min(...lats):0,.05);
  const zoom=Math.max(5,Math.min(11.5,Math.min(Math.log2(360*mw/(256*lonSpan*1.25)),Math.log2(180*mh/(256*latSpan*1.35)))));
  const selRow=rows.find(r=>r.station_id===sel);
  const ring=selRow?[{type:'scattermapbox',mode:'markers',lon:[num(selRow.lon)],lat:[num(selRow.lat)],marker:{color:'#0f2e3d',size:24},hoverinfo:'skip',showlegend:false}]:[];
  const mk=(s,g,stale)=>({type:'scattermapbox',mode:'markers',name:s+' stations',legendgroup:s,showlegend:!stale,lon:g.map(r=>num(r.lon)),lat:g.map(r=>num(r.lat)),customdata:g.map(r=>r.station_id),
    text:g.map(r=>'<b>'+safe(r.river)+'</b><br>'+safe(r.location)+'<br>WQI '+safe(r.wqi)+' ('+safe(r.status)+')<br>'+safe(r.datetime.replace('T',' ').slice(0,16))+(stale?' · STALE':'')),hoverinfo:'text',
    marker:{color:statusColor(s),size:g.map(r=>r.station_id===sel?17:13),opacity:stale?.4:1}});
  const traces=ring.concat(liveStatuses.flatMap(s=>{const g=rows.filter(r=>r.status===s);return[mk(s,g.filter(r=>!liveIsStale(r.datetime)),false),mk(s,g.filter(r=>liveIsStale(r.datetime)),true)]}));
  Plotly.react('live-map',traces,{margin:{t:0,l:0,r:0,b:0},uirevision:$('live-state').value,paper_bgcolor:'rgba(0,0,0,0)',font:{family:'Instrument Sans, system-ui',color:'#0f2e3d'},
    legend:{orientation:'h',x:.01,y:.02,xanchor:'left',yanchor:'bottom',bgcolor:'rgba(255,255,255,.92)'},
    mapbox:{style:'open-street-map',center:{lat:mid(lats),lon:mid(lons)},zoom}},{responsive:true,displaylogo:false});
  if(!liveMapBound){liveMapBound=true;$('live-map').on('plotly_click',ev=>{const id=ev.points&&ev.points[0]&&ev.points[0].customdata;if(id){$('live-station').value=id;renderLive()}})}
  // trend for the selected station
  const info=liveLatest.find(r=>r.station_id===sel);
  const hist=liveHistory.filter(r=>r.station_id===sel).sort((a,b)=>a.datetime<b.datetime?-1:1);
  $('live-trend-sub').textContent=info?info.river+' · '+info.location+' · '+hist.length+' hourly readings stored':'No station selected';
  Plotly.react('live-chart',[{type:'scatter',mode:'lines+markers',x:hist.map(r=>r.datetime),y:hist.map(r=>num(r.wqi)),name:'WQI',line:{color:'#8aa0ad',width:1.5},marker:{size:6,color:hist.map(r=>wqiColor(num(r.wqi)))}}],
    {...graphLayout(),showlegend:false,yaxis:{title:'WQI',range:[0,105],gridcolor:'rgba(15,46,61,.08)'},xaxis:{gridcolor:'rgba(15,46,61,.08)'},
     shapes:bandShapes().concat([81,60].map(y=>({type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:y,y1:y,line:{color:'#94a3b8',dash:'dot',width:1}}))),
     annotations:[{xref:'paper',x:1,yref:'y',y:81,text:'Clean ≥ 81',showarrow:false,xanchor:'right',yanchor:'bottom',font:{size:10,color:'#64748b'}},{xref:'paper',x:1,yref:'y',y:60,text:'Slightly polluted 60–80',showarrow:false,xanchor:'right',yanchor:'bottom',font:{size:10,color:'#64748b'}}]},
    {responsive:true,displaylogo:false});
  // table
  table('live-table',rows,[['state','State'],['river','River'],['station_id','Station'],['location','Location'],['datetime','Reading time (MYT)',v=>String(v).replace('T',' ').slice(0,16)],['wqi','WQI'],['status','Status'],
    ['datetime','Freshness',v=>liveIsStale(v)?'Stale ('+Math.round(liveAgeHours(v))+' h old)':'Current']]);
}
window.addEventListener('load',init);
