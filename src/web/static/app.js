'use strict';
const $ = s => document.querySelector(s);
let file = null, previewUrl = null, busy = false, resultModel = null;
function status(text, kind = '') { $('#status').textContent = text; $('#status').className = 'status ' + kind; }
function resetResult() { $('#result').value = ''; $('#timing').textContent = ''; $('#result-model').textContent = 'Результата пока нет'; resultModel = null; render(); }
function render() {
  const text = $('#result').value.trim();
  $('#copy').disabled = !text; $('#download').disabled = !text;
  $('#rendered').replaceChildren();
  if (!text) { const placeholder = document.createElement('span'); placeholder.className = 'placeholder'; placeholder.textContent = 'Здесь появится ваша формула'; $('#rendered').append(placeholder); return; }
  let expression = text.replace(/^\$\$?([\s\S]*?)\$\$?$/, '$1').replace(/^\\\[([\s\S]*)\\\]$/, '$1');
  if (window.katex) { try { katex.render(expression, $('#rendered'), {displayMode:true,throwOnError:true,trust:false,maxExpand:1000}); } catch (_) { $('#rendered').textContent = 'Не удалось отрисовать запись. LaTeX доступен ниже.'; } }
  else $('#rendered').textContent = 'Модуль отображения формул недоступен. LaTeX доступен ниже.';
}
function selectFile(next) {
  if (busy) return;
  if (next && next.size > 10*1024*1024) { status('Файл больше 10 МБ.', 'error'); return; }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  file = next; resetResult(); $('#recognize').disabled = !file; $('#clear').hidden = !file;
  $('#upload-hint').hidden = !!file; $('#preview').hidden = !file;
  $('#file-info').textContent = file ? `${file.name} · ${(file.size/1024).toFixed(0)} КБ` : 'Изображение ещё не выбрано';
  if (file) { previewUrl = URL.createObjectURL(file); $('#preview').src = previewUrl; status('Готово к распознаванию.'); }
  else { previewUrl = null; $('#preview').removeAttribute('src'); $('#file').value = ''; status('Загрузите изображение, чтобы начать.'); }
}
$('#file').addEventListener('change', () => selectFile($('#file').files[0] || null));
$('#clear').addEventListener('click', () => selectFile(null));
$('#drop-zone').addEventListener('keydown', e => { if (!busy && ['Enter',' '].includes(e.key)) { e.preventDefault(); $('#file').click(); } });
for (const event of ['dragover','dragleave','drop']) $('#drop-zone').addEventListener(event, e => {e.preventDefault(); $('#drop-zone').classList.toggle('dragover', event === 'dragover' && !busy); if (event === 'drop') selectFile(e.dataTransfer.files[0] || null);});
$('#model').addEventListener('change', () => {resetResult();$('#model-description').textContent = $('#model').value === 'pix2tex' ? 'Предобучена на формулах · локальный inference' : 'Учебная модель · для сравнения';});
$('#recognize').addEventListener('click', async () => {
  if (!file || busy) return;
  busy = true; resetResult(); const model = $('#model').value;
  for (const id of ['file','model','recognize','clear']) $('#'+id).disabled = true;
  $('#recognize').textContent = 'Распознаём…'; status('Выполняется распознавание. При первом запросе загружается модель в память.', 'busy');
  try {
    const form = new FormData(); form.append('file', file); form.append('model', model);
    const response = await fetch('/api/predict', {method:'POST',body:form});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Некорректный запрос.');
    $('#result').value = data.latex; resultModel = data.model;
    $('#timing').textContent = `${data.inference_seconds.toFixed(2)} с`;
    $('#result-model').textContent = data.model === 'pix2tex' ? 'pix2tex' : 'Baseline'; render();
    status(data.latex.trim() ? 'Готово. Проверьте формулу; запись ниже можно исправить.' : 'Модель вернула пустой результат. Попробуйте другой кроп.');
  } catch (error) { status(error.message || 'Сервер недоступен.', 'error'); }
  finally {busy = false;for (const id of ['file','model','clear']) $('#'+id).disabled = false;$('#recognize').disabled = !file;$('#recognize').textContent = 'Распознать ↗';}
});
$('#result').addEventListener('input', render);
$('#copy').addEventListener('click', async () => {try {await navigator.clipboard.writeText($('#result').value);status('LaTeX скопирован.');} catch (_) {status('Не удалось скопировать автоматически. Выделите запись и скопируйте её.', 'error');}});
$('#download').addEventListener('click', () => {const url = URL.createObjectURL(new Blob([$('#result').value+'\n'],{type:'text/plain;charset=utf-8'}));const link = document.createElement('a');link.href=url;link.download='formula.tex';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
const percent = value => Number.isFinite(value) ? (value*100).toFixed(1)+'%' : '—';
function chart(element, rows, key) {
 element.replaceChildren();const values=rows.filter(r=>Number.isFinite(r[key]));if(!values.length)return;
 const ns='http://www.w3.org/2000/svg';const make=(tag,attributes)=>{const node=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attributes))node.setAttribute(k,v);element.append(node);return node;};
 const max=Math.max(...values.map(r=>r[key]),.01)*1.1;
 for(let i=0;i<4;i++){const y=15+i*50;make('line',{x1:45,y1:y,x2:465,y2:y,stroke:'#e5ece4'});make('text',{x:38,y:y+4,'text-anchor':'end',fill:'#849286','font-size':10}).textContent=(max*(1-i/3)).toFixed(2);}
 const points=values.map((r,i)=>`${45+(values.length>1?i/(values.length-1)*420:210)},${165-r[key]/max*150}`);
 make('polyline',{points:points.join(' '),fill:'none',stroke:'#19705b','stroke-width':2});make('text',{x:45,y:190,fill:'#849286','font-size':10}).textContent=`Эпохи ${values[0].epoch}–${values.at(-1).epoch}`;
}
async function loadMetrics(){
 try {const response=await fetch('/api/model');const data=await response.json();if(!response.ok || !data.available)throw new Error('Сохранённая baseline-модель недоступна.');$('#metrics-status').hidden=true;$('#metrics-content').hidden=false;$('#summary').replaceChildren();
 for(const [name,value] of [['Параметров',Number(data.parameter_count).toLocaleString('ru-RU')],['Словарь',data.vocab_size],['Лучшая эпоха',data.epoch??'—'],['Validation exact',percent(data.validation?.exact_match)]]){const card=document.createElement('div');card.textContent=name;const strong=document.createElement('strong');strong.textContent=value;card.append(strong);$('#summary').append(card);}
 $('#metrics-rows').replaceChildren();const metrics=data.metrics?.splits||{};
 for(const split of new Set(['val','test',...Object.keys(metrics)])){const row=document.createElement('tr');for(const value of [split,metrics[split]?.count??'не рассчитано',percent(metrics[split]?.exact_match),percent(metrics[split]?.cer)]){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}$('#metrics-rows').append(row);}
 chart($('#loss-chart'),data.history||[],'train_loss');chart($('#cer-chart'),data.history||[],'val_cer');$('#history-note').textContent=data.history?.length?'':'История обучения для этого checkpoint не сохранена.';$('#config').textContent=JSON.stringify({checkpoint:data.checkpoint,...data.config},null,2);
 }catch(error){$('#metrics-status').textContent=error.message;$('#metrics-status').hidden=false;}
}
for(const id of ['lab','metrics'])$('#'+id+'-tab').addEventListener('click',()=>{for(const tab of ['lab','metrics']){const active=tab===id;$('#'+tab+'-panel').hidden=!active;$('#'+tab+'-tab').classList.toggle('active',active);$('#'+tab+'-tab').setAttribute('aria-pressed',String(active));}if(id==='metrics')loadMetrics();});
window.addEventListener('pagehide',()=>{if(previewUrl)URL.revokeObjectURL(previewUrl);});
