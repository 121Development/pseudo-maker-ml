'use strict';

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// ---- Tabs ---------------------------------------------------------------

$$('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    $$('.tab').forEach((t) => t.classList.toggle('active', t === tab));
    const target = tab.dataset.tab;
    $$('.panel').forEach((p) => p.classList.toggle('active', p.id === `panel-${target}`));
  });
});

function showStage(panelId, stageId) {
  $$(`#${panelId} .stage`).forEach((s) => s.classList.toggle('active', s.id === stageId));
}

// ---- Client-side validation --------------------------------------------

function luhnValid(digits) {
  if (!/^\d+$/.test(digits) || digits.length === 0) return false;
  let total = 0;
  const parity = digits.length % 2;
  for (let i = 0; i < digits.length; i++) {
    let n = parseInt(digits[i], 10);
    if (i % 2 === parity) {
      n *= 2;
      if (n > 9) n -= 9;
    }
    total += n;
  }
  return total % 10 === 0;
}

function validatePersonnummer(text) {
  const digits = text.replace(/[-+]/g, '');
  if (digits.length !== 10 && digits.length !== 12) return { ok: false, message: 'expected 10 or 12 digits' };
  const ten = digits.length === 12 ? digits.slice(2) : digits;
  if (!/^\d{10}$/.test(ten)) return { ok: false, message: 'non-digit characters' };
  return luhnValid(ten)
    ? { ok: true }
    : { ok: false, message: 'invalid Luhn checksum' };
}

function validateOrgnr(text) {
  const digits = text.replace(/[^\d]/g, '');
  if (digits.length !== 10) return { ok: false, message: 'expected 10 digits' };
  if (parseInt(digits[2], 10) < 2) return { ok: false, message: 'third digit must be ≥ 2' };
  return luhnValid(digits)
    ? { ok: true }
    : { ok: false, message: 'invalid Luhn checksum' };
}

function validateBankgiro(text) {
  const digits = text.replace(/[^\d]/g, '');
  if (digits.length < 7 || digits.length > 8) return { ok: false, message: 'expected 7–8 digits' };
  return luhnValid(digits)
    ? { ok: true }
    : { ok: false, message: 'invalid Luhn checksum' };
}

function validateIban(text) {
  const compact = text.replace(/\s+/g, '').toUpperCase();
  if (!/^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$/.test(compact)) return { ok: false, message: 'bad IBAN format' };
  return { ok: true };
}

function validateCreditCard(text) {
  const digits = text.replace(/[^\d]/g, '');
  if (digits.length < 13 || digits.length > 19) return { ok: false, message: 'expected 13–19 digits' };
  return luhnValid(digits) ? { ok: true } : { ok: false, message: 'invalid Luhn checksum' };
}

function validateEmail(text) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(text)
    ? { ok: true }
    : { ok: false, message: 'not a valid email' };
}

function validateForType(type, value) {
  if (!value || !value.trim()) return { ok: false, message: 'empty' };
  switch (type) {
    case 'PERSONNUMMER': return validatePersonnummer(value);
    case 'ORGNR':        return validateOrgnr(value);
    case 'BANKGIRO':
    case 'PLUSGIRO':     return validateBankgiro(value);
    case 'IBAN':         return validateIban(value);
    case 'CREDIT_CARD':  return validateCreditCard(value);
    case 'EMAIL':        return validateEmail(value);
    default:             return { ok: true };
  }
}

// ---- Clean flow ---------------------------------------------------------

const TYPE_OPTIONS = [
  ['PERSON', 'Person'],
  ['COMPANY', 'Company / organisation'],
  ['CITY', 'City / place'],
  ['STREET', 'Street / address'],
  ['EMAIL', 'Email'],
  ['PHONE', 'Phone'],
  ['PERSONNUMMER', 'Personnummer'],
  ['ORGNR', 'Org.nr'],
  ['ID_NUMBER', 'ID / account number'],
  ['CASE_NUMBER', 'Case number'],
  ['DATE', 'Date'],
  ['CUSTOM', 'Other (placeholder)'],
];

function typeSelectHtml(selected, cls = 'type-select') {
  const known = TYPE_OPTIONS.some(([v]) => v === selected);
  const sel = known ? selected : 'CUSTOM';
  return `<select class="${cls}">${TYPE_OPTIONS.map(([v, label]) =>
    `<option value="${v}"${v === sel ? ' selected' : ''}>${escapeHtml(label)}</option>`).join('')}</select>`;
}

const cleanState = {
  sessionId: null,
  entities: [],
  candidates: [],
  document: [],
  excluded: new Set(),
  customFakes: new Map(), // entity_id → user-chosen fake (string)
  currentFakes: new Map(), // entity_id → current fake (auto or custom), used for duplicate detection
  view: 'table',
};

function wireDropzone(zone, input, onFile) {
  zone.addEventListener('click', () => input.click());
  input.addEventListener('change', (e) => {
    if (e.target.files[0]) onFile(e.target.files[0]);
  });
  ['dragenter', 'dragover'].forEach((evt) =>
    zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.add('dragover'); })
  );
  ['dragleave', 'drop'].forEach((evt) =>
    zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.remove('dragover'); })
  );
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
  });
}

wireDropzone($('#dropzone-clean'), $('#file-clean'), uploadClean);
$('#add-term-type').outerHTML = typeSelectHtml('PERSON', 'type-select').replace('<select', '<select id="add-term-type"');

async function uploadClean(file) {
  showStage('panel-clean', 'clean-processing');
  try {
    const body = new FormData();
    body.append('file', file);
    const res = await fetch('/api/clean/upload', { method: 'POST', body });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    cleanState.sessionId = data.session_id;
    cleanState.excluded = new Set();
    cleanState.customFakes = new Map();
    applyPreviewState(data);
    $('#add-term-msg').hidden = true;
    showStage('panel-clean', 'clean-preview');
  } catch (err) {
    showCleanError(err);
  }
}

/** Take a server preview state ({entities, candidates, document}) and re-render everything. */
function applyPreviewState(data) {
  cleanState.entities = data.entities;
  cleanState.candidates = data.candidates || [];
  cleanState.document = data.document || [];
  const ids = new Set(data.entities.map((e) => e.id));
  cleanState.excluded = new Set([...cleanState.excluded].filter((id) => ids.has(id)));
  cleanState.currentFakes = new Map(data.entities.map((e) => [e.id, e.fake]));
  renderEntities();
  renderDocument();
  renderAmbiguous();
}

async function postForm(url, fields) {
  const body = new FormData();
  for (const [k, v] of Object.entries(fields)) body.append(k, v);
  const res = await fetch(url, { method: 'POST', body });
  if (!res.ok) {
    let msg = await res.text();
    try { msg = JSON.parse(msg).detail || msg; } catch (_) { /* plain text */ }
    throw new Error(msg);
  }
  return res.json();
}

async function addTerm(term, type) {
  const data = await postForm('/api/clean/add-term', { session_id: cleanState.sessionId, term, type });
  applyPreviewState(data);
  return data.added;
}

async function setFake(entityId, fake) {
  const data = await postForm('/api/clean/set-fake', { session_id: cleanState.sessionId, entity_id: entityId, fake });
  applyPreviewState(data);
}

function setExcluded(id, excluded) {
  if (excluded) cleanState.excluded.add(id); else cleanState.excluded.delete(id);
  const cb = $(`#entity-body input[data-id="${id}"]`);
  if (cb) {
    cb.checked = !excluded;
    cb.closest('tr').classList.toggle('unchecked', excluded);
  }
  renderDocument();
}

// ---- View switch

$$('.view-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    cleanState.view = btn.dataset.view;
    $$('.view-btn').forEach((b) => b.classList.toggle('active', b === btn));
    $$('#clean-preview .view').forEach((v) => v.classList.toggle('active', v.id === `view-${cleanState.view}`));
    hidePopover();
  });
});

// ---- Table view

function renderEntities() {
  const body = $('#entity-body');
  body.innerHTML = '';
  $('#entity-count').textContent = `(${cleanState.entities.length})`;
  for (const e of cleanState.entities) {
    const tr = document.createElement('tr');
    tr.dataset.id = e.id;
    tr.dataset.type = e.type;
    const excluded = cleanState.excluded.has(e.id);
    if (excluded) tr.classList.add('unchecked');
    if (cleanState.customFakes.has(e.id)) tr.classList.add('edited');
    const badges = [
      e.uncertain ? `<span class="flag uncertain" title="Lowest model confidence ${Math.round(e.confidence * 100)}%">uncertain</span>` : '',
      (e.sources || []).includes('manual') ? '<span class="flag manual" title="Added by you">added</span>' : '',
    ].join('');
    tr.innerHTML = `
      <td class="col-check"><input type="checkbox" data-id="${e.id}" ${excluded ? '' : 'checked'} /></td>
      <td><span class="type-badge">${escapeHtml(e.type)}</span></td>
      <td class="col-original">${escapeHtml(e.original)}${e.aliases.length ? ` <span class="muted">(+${e.aliases.length} alias)</span>` : ''} ${badges}</td>
      <td class="col-fake">
        <span class="fake-display" tabindex="0">${escapeHtml(e.fake)}</span>
        <span class="fake-warning" hidden></span>
      </td>
      <td class="col-count">${e.occurrences}</td>
    `;
    body.appendChild(tr);
  }
  body.querySelectorAll('input[type="checkbox"]').forEach((cb) => {
    cb.addEventListener('change', (ev) => setExcluded(ev.currentTarget.dataset.id, !ev.currentTarget.checked));
  });
  body.querySelectorAll('.fake-display').forEach((el) => {
    el.addEventListener('click', (e) => beginEdit(e.currentTarget.closest('tr')));
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        beginEdit(e.currentTarget.closest('tr'));
      }
    });
  });
}

/** Soft checks for a new fake; returns a warning message or ''. */
function fakeWarning(id, type, newVal) {
  for (const [otherId, otherFake] of cleanState.currentFakes) {
    if (otherId !== id && otherFake === newVal) return `also used for ${otherId}`;
  }
  const v = validateForType(type, newVal);
  return v.ok ? '' : v.message;
}

/** Validate and apply a new fake for an entity. Returns '' or a warning to show. */
async function commitFake(id, newVal) {
  const entry = cleanState.entities.find((e) => e.id === id);
  newVal = newVal.trim();
  if (!entry || !newVal || newVal === entry.fake) return '';
  if (newVal === entry.original) return 'same as original — not changed';
  const warning = fakeWarning(id, entry.type, newVal);
  cleanState.customFakes.set(id, newVal);
  await setFake(id, newVal);
  return warning;
}

function beginEdit(tr) {
  if (!tr) return;
  if (tr.querySelector('.fake-input')) return; // already editing
  const cell = tr.querySelector('.col-fake');
  const display = cell.querySelector('.fake-display');
  const warn = cell.querySelector('.fake-warning');
  const id = tr.dataset.id;

  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'fake-input';
  input.value = display.textContent;
  input.setAttribute('spellcheck', 'false');
  input.setAttribute('autocomplete', 'off');

  display.hidden = true;
  warn.hidden = true;
  cell.insertBefore(input, warn);
  input.focus();
  input.select();

  let done = false;
  const commit = async () => {
    if (done) return;
    done = true;
    try {
      const warning = await commitFake(id, input.value);
      if (warning) {
        const w = $(`#entity-body tr[data-id="${id}"] .fake-warning`);
        if (w) showWarn(w, warning);
      }
    } catch (err) {
      showWarn(warn, err.message);
    }
    input.remove();
    display.hidden = false;
  };
  const cancel = () => {
    done = true;
    input.remove();
    display.hidden = false;
  };
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); commit(); }
    else if (e.key === 'Escape') { e.preventDefault(); cancel(); }
  });
  input.addEventListener('blur', commit);
}

function showWarn(el, message) {
  el.textContent = `⚠ ${message}`;
  el.hidden = false;
}

$('#select-all').addEventListener('click', () => {
  cleanState.excluded.clear();
  renderEntities();
  renderDocument();
});
$('#select-none').addEventListener('click', () => {
  cleanState.excluded = new Set(cleanState.entities.map((e) => e.id));
  renderEntities();
  renderDocument();
});

// ---- Document view

$('#show-originals').addEventListener('change', renderDocument);

function renderDocument() {
  const view = $('#doc-view');
  const showOriginals = $('#show-originals').checked;
  const entityById = new Map(cleanState.entities.map((e) => [e.id, e]));
  const parts = [];
  let blankRun = 0;
  cleanState.document.forEach((para, pi) => {
    if (!para.text.trim()) {
      if (++blankRun === 1 && parts.length) parts.push('<p class="doc-para blank"></p>');
      return;
    }
    blankRun = 0;
    const marks = [
      ...para.spans.map((s, si) => ({ ...s, kind: 'sub', si })),
      ...para.candidates.map((c, ci) => ({ ...c, kind: 'cand', ci })),
    ].sort((a, b) => a.start - b.start);
    let html = '';
    let pos = 0;
    for (const m of marks) {
      if (m.start < pos) continue; // never nest
      html += escapeHtml(para.text.slice(pos, m.start));
      const original = para.text.slice(m.start, m.end);
      if (m.kind === 'sub') {
        const e = entityById.get(m.entity_id);
        const excluded = cleanState.excluded.has(m.entity_id);
        const shown = excluded || showOriginals ? original : m.fake;
        const cls = ['sub', excluded ? 'excluded' : '', e && e.uncertain ? 'uncertain' : ''].join(' ');
        const tip = excluded ? `Kept as-is (${e ? e.type : ''})` : `${original} → ${m.fake}`;
        html += `<mark class="${cls}" data-eid="${m.entity_id}" data-pi="${pi}" data-si="${m.si}" title="${escapeHtml(tip)}">${escapeHtml(shown)}</mark>`;
      } else {
        html += `<span class="cand" data-pi="${pi}" data-ci="${m.ci}" title="Possibly sensitive — click to add">${escapeHtml(original)}</span>`;
      }
      pos = m.end;
    }
    html += escapeHtml(para.text.slice(pos));
    parts.push(`<p class="doc-para">${html}</p>`);
  });
  view.innerHTML = parts.join('') || '<p class="muted">The document is empty.</p>';
}

$('#doc-view').addEventListener('click', (ev) => {
  const mark = ev.target.closest('mark.sub');
  if (mark) {
    ev.stopPropagation();
    openEntityPopover(mark);
    return;
  }
  const cand = ev.target.closest('.cand');
  if (cand) {
    ev.stopPropagation();
    const para = cleanState.document[+cand.dataset.pi];
    const c = para.candidates[+cand.dataset.ci];
    const info = cleanState.candidates.find((x) => x.type === c.type && x.value.toLowerCase() === c.value.toLowerCase());
    openAddPopover(cand, c.value, c.type, info ? info.reason : '');
  }
});

$('#doc-view').addEventListener('mouseup', () => {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed) return;
  const text = sel.toString().replace(/\s+/g, ' ').trim();
  if (text.length < 2 || text.length > 120) return;
  if (!$('#doc-view').contains(sel.anchorNode)) return;
  const rect = sel.getRangeAt(0).getBoundingClientRect();
  openAddPopover(rect, text, guessType(text), 'Selected text');
});

function guessType(text) {
  if (/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(text)) return 'EMAIL';
  if (/^\+?[\d\s\-()]{7,}$/.test(text)) return 'PHONE';
  if (/\d/.test(text)) return 'ID_NUMBER';
  if (/\b(AB|HB|KB|Ltd|Inc|GmbH)\b/.test(text)) return 'COMPANY';
  if (/^[A-ZÅÄÖ][\wåäöé-]+(\s+[A-ZÅÄÖ][\wåäöé-]+){0,2}$/.test(text)) return 'PERSON';
  return 'CUSTOM';
}

// ---- Popover

const popover = $('#popover');

function placePopover(anchor) {
  const r = anchor.getBoundingClientRect ? anchor.getBoundingClientRect() : anchor;
  popover.hidden = false;
  const pw = popover.offsetWidth;
  const left = Math.min(Math.max(8, r.left + window.scrollX), window.scrollX + document.documentElement.clientWidth - pw - 8);
  popover.style.left = `${left}px`;
  popover.style.top = `${r.bottom + window.scrollY + 6}px`;
}

function hidePopover() {
  popover.hidden = true;
  popover.innerHTML = '';
}

document.addEventListener('mousedown', (ev) => {
  if (!popover.hidden && !popover.contains(ev.target) && !ev.target.closest('mark.sub, .cand')) hidePopover();
});
document.addEventListener('keydown', (ev) => { if (ev.key === 'Escape') hidePopover(); });

function openEntityPopover(mark) {
  const id = mark.dataset.eid;
  const e = cleanState.entities.find((x) => x.id === id);
  if (!e) return;
  const para = cleanState.document[+mark.dataset.pi];
  const span = para.spans[+mark.dataset.si];
  const occurrenceText = para.text.slice(span.start, span.end);
  const excluded = cleanState.excluded.has(id);
  popover.innerHTML = `
    <div class="pop-head"><span class="type-badge">${escapeHtml(e.type)}</span>
      ${e.uncertain ? '<span class="flag uncertain">uncertain</span>' : ''}
      <span class="muted">${e.occurrences}× in document</span></div>
    <div class="pop-row"><span class="pop-label">Original</span><span class="pop-original">${escapeHtml(e.original)}</span></div>
    ${occurrenceText !== e.original ? `<div class="pop-row"><span class="pop-label">Here</span><span>${escapeHtml(occurrenceText)} → ${escapeHtml(span.fake)}</span></div>` : ''}
    <label class="pop-row"><span class="pop-label">Replace with</span>
      <input type="text" class="fake-input pop-fake" value="${escapeHtml(e.fake)}" spellcheck="false" autocomplete="off" /></label>
    <p class="fake-warning pop-warn" hidden></p>
    <label class="pop-check"><input type="checkbox" class="pop-replace" ${excluded ? '' : 'checked'} /> Replace in the cleaned document</label>
    <div class="pop-actions">
      <button class="primary small pop-save">Save</button>
      <button class="secondary small pop-cancel">Close</button>
    </div>`;
  placePopover(mark);
  const input = popover.querySelector('.pop-fake');
  input.focus();
  input.select();
  const save = async () => {
    const replace = popover.querySelector('.pop-replace').checked;
    try {
      const warning = await commitFake(id, input.value);
      setExcluded(id, !replace);
      if (warning) { showWarn(popover.querySelector('.pop-warn'), warning); return; }
      hidePopover();
    } catch (err) {
      showWarn(popover.querySelector('.pop-warn'), err.message);
    }
  };
  popover.querySelector('.pop-save').addEventListener('click', save);
  popover.querySelector('.pop-cancel').addEventListener('click', hidePopover);
  input.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') { ev.preventDefault(); save(); } });
}

function openAddPopover(anchor, value, type, reason) {
  popover.innerHTML = `
    <div class="pop-head"><span class="muted">${escapeHtml(reason || 'Add as sensitive term')}</span></div>
    <div class="pop-row"><span class="pop-label">Term</span><span class="pop-original">${escapeHtml(value)}</span></div>
    <label class="pop-row"><span class="pop-label">Type</span>${typeSelectHtml(type)}</label>
    <p class="fake-warning pop-warn" hidden></p>
    <div class="pop-actions">
      <button class="primary small pop-add">Replace everywhere</button>
      <button class="secondary small pop-cancel">Close</button>
    </div>`;
  placePopover(anchor);
  popover.querySelector('.pop-add').addEventListener('click', async () => {
    const t = popover.querySelector('select').value;
    try {
      const n = await addTerm(value, t);
      if (!n) { showWarn(popover.querySelector('.pop-warn'), 'not found as a whole word (or already replaced)'); return; }
      hidePopover();
      window.getSelection().removeAllRanges();
    } catch (err) {
      showWarn(popover.querySelector('.pop-warn'), err.message);
    }
  });
  popover.querySelector('.pop-cancel').addEventListener('click', hidePopover);
}

// ---- Ambiguous list

function renderAmbiguous() {
  const list = $('#ambiguous-list');
  const items = cleanState.candidates;
  $('#ambiguous-count').textContent = `(${items.length})`;
  $('#ambiguous').classList.toggle('empty', items.length === 0);
  if (!items.length) {
    list.innerHTML = '<li class="muted">Nothing ambiguous found.</li>';
    return;
  }
  list.innerHTML = items.map((c) => `
    <li data-cid="${c.id}">
      <span class="amb-value">${escapeHtml(c.value)}</span>
      <span class="muted">${c.occurrences}×</span>
      <span class="amb-reason muted">${escapeHtml(c.reason)}${c.reason.startsWith('low') ? ` (${Math.round(c.score * 100)}%)` : ''}</span>
      ${typeSelectHtml(c.type)}
      <button class="secondary small amb-add">Add</button>
    </li>`).join('');
  list.querySelectorAll('.amb-add').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const li = btn.closest('li');
      const c = cleanState.candidates.find((x) => x.id === li.dataset.cid);
      btn.disabled = true;
      try {
        await addTerm(c.value, li.querySelector('select').value);
      } catch (err) {
        btn.disabled = false;
        btn.textContent = '⚠ retry';
        btn.title = err.message;
      }
    });
  });
}

// ---- Add a term

$('#add-term-form').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const input = $('#add-term-input');
  const term = input.value.trim();
  const msg = $('#add-term-msg');
  if (!term) return;
  try {
    const n = await addTerm(term, $('#add-term-type').value);
    msg.textContent = n
      ? `✓ Replaced ${n} occurrence${n === 1 ? '' : 's'} of “${term}”.`
      : `“${term}” wasn't found as a whole word, or it's already replaced.`;
    msg.className = `add-term-msg ${n ? 'ok' : 'warn'}`;
    if (n) input.value = '';
  } catch (err) {
    msg.textContent = `⚠ ${err.message}`;
    msg.className = 'add-term-msg warn';
  }
  msg.hidden = false;
});

// ---- Finalize

$('#finalize-btn').addEventListener('click', async () => {
  hidePopover();
  showStage('panel-clean', 'clean-processing');
  try {
    const body = new FormData();
    body.append('session_id', cleanState.sessionId);
    body.append('excluded_ids', Array.from(cleanState.excluded).join(','));
    const res = await fetch('/api/clean/finalize', { method: 'POST', body });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    $('#download-cleaned').href = data.cleaned_url;
    $('#download-cleaned').setAttribute('download', data.cleaned_name);
    $('#download-cleaned').textContent = `📄 Download ${data.cleaned_name}`;
    $('#download-mapping').href = data.mapping_url;
    $('#download-mapping').setAttribute('download', data.mapping_name);
    $('#download-mapping').textContent = `🗝  Download ${data.mapping_name}`;
    const editCount = cleanState.customFakes.size;
    $('#clean-done-summary').textContent =
      `${data.kept_entity_count} entities substituted` +
      (editCount ? ` (${editCount} custom edit${editCount === 1 ? '' : 's'}).` : '.');
    showStage('panel-clean', 'clean-done');
  } catch (err) {
    showCleanError(err);
  }
});

$('#cancel-btn').addEventListener('click', resetCleanFlow);
$('#clean-again-btn').addEventListener('click', resetCleanFlow);
$('#clean-error-reset').addEventListener('click', resetCleanFlow);

function resetCleanFlow() {
  hidePopover();
  closeSession(cleanState.sessionId);
  cleanState.sessionId = null;
  cleanState.entities = [];
  cleanState.candidates = [];
  cleanState.document = [];
  cleanState.excluded = new Set();
  cleanState.customFakes = new Map();
  cleanState.currentFakes = new Map();
  $('#file-clean').value = '';
  $('#add-term-input').value = '';
  $('#add-term-msg').hidden = true;
  showStage('panel-clean', 'clean-upload');
}

function showCleanError(err) {
  $('#clean-error-msg').textContent = (err && err.message) || String(err);
  showStage('panel-clean', 'clean-error');
}

// ---- Restore flow -------------------------------------------------------

const restoreState = { doc: null, mapping: null, sessionId: null };

function closeSession(sessionId) {
  if (!sessionId) return;
  try {
    navigator.sendBeacon(`/api/session/${sessionId}/close`);
  } catch (_) {
    fetch(`/api/session/${sessionId}/close`, { method: 'POST', keepalive: true }).catch(() => {});
  }
}

window.addEventListener('beforeunload', () => {
  closeSession(cleanState.sessionId);
  closeSession(restoreState.sessionId);
});

wireDropzone($('#dropzone-restore-doc'), $('#file-restore-doc'), (f) => {
  restoreState.doc = f;
  $('#restore-doc-name').textContent = f.name;
  $('#dropzone-restore-doc').classList.add('has-file');
  updateRestoreButton();
});
wireDropzone($('#dropzone-restore-mapping'), $('#file-restore-mapping'), (f) => {
  restoreState.mapping = f;
  $('#restore-mapping-name').textContent = f.name;
  $('#dropzone-restore-mapping').classList.add('has-file');
  updateRestoreButton();
});

function updateRestoreButton() {
  $('#restore-go').disabled = !(restoreState.doc && restoreState.mapping);
}

$('#restore-go').addEventListener('click', async () => {
  showStage('panel-restore', 'restore-processing');
  try {
    const body = new FormData();
    body.append('document', restoreState.doc);
    body.append('mapping', restoreState.mapping);
    const res = await fetch('/api/restore', { method: 'POST', body });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    restoreState.sessionId = data.session_id;
    $('#download-restored').href = data.restored_url;
    $('#download-restored').setAttribute('download', data.restored_name);
    $('#download-restored').textContent = `📄 Download ${data.restored_name}`;
    $('#download-audit').href = data.audit_url;
    $('#download-audit').setAttribute('download', data.audit_name);
    $('#download-audit').textContent = `📋 Download ${data.audit_name}`;
    renderRestoreReport(data.report);
    showStage('panel-restore', 'restore-done');
  } catch (err) {
    $('#restore-error-msg').textContent = (err && err.message) || String(err);
    showStage('panel-restore', 'restore-error');
  }
});

function renderRestoreReport(report) {
  const box = $('#restore-report');
  box.innerHTML = '';

  const exactTotal = report.exact.reduce((n, h) => n + (h.count || 1), 0);
  box.appendChild(reportSection(
    `✓ Restored exactly (${exactTotal} occurrence${exactTotal === 1 ? '' : 's'} across ${report.exact.length} entit${report.exact.length === 1 ? 'y' : 'ies'})`,
    'exact',
    report.exact,
    (h) => `
      <span class="type-badge">${escapeHtml(h.type || '')}</span>
      <span class="r-fake">${escapeHtml(h.fake)}</span>
      <span class="r-arrow">→</span>
      <span class="r-original">${escapeHtml(h.original)}</span>
      <span class="muted">×${h.count || 1}</span>`,
    report.exact.length > 0,
  ));

  if (report.fuzzy.length) {
    box.appendChild(reportSection(
      `≈ Fuzzy matches (${report.fuzzy.length}) — review recommended`,
      'fuzzy',
      report.fuzzy,
      (h) => `
        <span class="type-badge">${escapeHtml(h.type || '')}</span>
        <span class="r-fake">“${escapeHtml(h.matched_text)}”</span>
        <span class="r-arrow">→</span>
        <span class="r-original">${escapeHtml(h.original)}</span>
        <span class="muted">(score ${h.score}, expected “${escapeHtml(h.fake || '')}”)</span>`,
      true,
    ));
  }

  if (report.missing.length) {
    box.appendChild(reportSection(
      `✗ Not found in LLM output (${report.missing.length}) — the LLM may have dropped or rewrote these`,
      'missing',
      report.missing,
      (h) => `
        <span class="type-badge">${escapeHtml(h.type || '')}</span>
        <span class="r-fake">${escapeHtml(h.fake)}</span>
        <span class="r-arrow">—</span>
        <span class="r-original">(would have been ${escapeHtml(h.original)})</span>`,
      true,
    ));
  }
}

function reportSection(title, kind, items, renderItem, defaultOpen) {
  const details = document.createElement('details');
  details.className = `report-section ${kind}`;
  if (defaultOpen) details.open = true;
  const summary = document.createElement('summary');
  summary.textContent = title;
  details.appendChild(summary);
  if (items.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'muted';
    empty.textContent = 'None.';
    details.appendChild(empty);
    return details;
  }
  const ul = document.createElement('ul');
  ul.className = 'report-list';
  for (const h of items) {
    const li = document.createElement('li');
    li.innerHTML = renderItem(h);
    ul.appendChild(li);
  }
  details.appendChild(ul);
  return details;
}

$('#restore-again-btn').addEventListener('click', resetRestoreFlow);
$('#restore-error-reset').addEventListener('click', resetRestoreFlow);

function resetRestoreFlow() {
  closeSession(restoreState.sessionId);
  restoreState.sessionId = null;
  restoreState.doc = null;
  restoreState.mapping = null;
  $('#file-restore-doc').value = '';
  $('#file-restore-mapping').value = '';
  $('#restore-doc-name').textContent = '.docx, .txt, or .md';
  $('#restore-mapping-name').textContent = '_MAPPING.json';
  $('#dropzone-restore-doc').classList.remove('has-file');
  $('#dropzone-restore-mapping').classList.remove('has-file');
  updateRestoreButton();
  showStage('panel-restore', 'restore-upload');
}

// ---- Utilities ----------------------------------------------------------

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
