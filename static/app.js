const sourceInputs = document.querySelectorAll('input[name="origem"]');
const sourcePanels = document.querySelectorAll('[data-source]');
const form = document.querySelector('#prediction-form');
const resultPanel = document.querySelector('#results-panel');
const fileInput = document.querySelector('#percurso_upload');

function updateSource() {
  const selected = document.querySelector('input[name="origem"]:checked')?.value || 'existente';
  sourcePanels.forEach((panel) => {
    const active = panel.dataset.source === selected;
    panel.hidden = !active;
    panel.querySelectorAll('input, select').forEach((input) => { input.disabled = !active; });
  });
}

sourceInputs.forEach((input) => input.addEventListener('change', updateSource));
updateSource();

fileInput?.addEventListener('change', () => {
  const label = document.querySelector('[data-file-name]');
  if (label) label.textContent = fileInput.files[0]?.name || 'FIT ou CSV · máximo 32 MB';
});

function loadingMarkup() {
  return `<div class="card loading-result">
    <div class="spinner" aria-hidden="true"></div>
    <div><strong>A analisar o percurso</strong><span>A construir o perfil e a calcular o ritmo…</span></div>
    <div class="loading-lines"><i></i><i></i><i></i><i></i></div>
  </div>`;
}

function errorMarkup(message) {
  const wrapper = document.createElement('div');
  wrapper.className = 'alert enter';
  const title = document.createElement('strong');
  title.textContent = 'Não foi possível calcular';
  const detail = document.createElement('span');
  detail.textContent = message;
  wrapper.append(title, detail);
  return wrapper;
}

form?.addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = form.querySelector('button[type="submit"]');
  const original = button.querySelector('span').textContent;
  button.disabled = true;
  button.querySelector('span').textContent = 'A calcular…';
  resultPanel.setAttribute('aria-busy', 'true');
  resultPanel.innerHTML = loadingMarkup();

  try {
    const response = await fetch('/api/prever', { method: 'POST', body: new FormData(form) });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.erro || 'Erro inesperado na previsão.');
    resultPanel.innerHTML = payload.html;
    if (window.matchMedia('(max-width: 860px)').matches) {
      resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  } catch (error) {
    resultPanel.replaceChildren(errorMarkup(error.message));
  } finally {
    resultPanel.setAttribute('aria-busy', 'false');
    button.disabled = false;
    button.querySelector('span').textContent = original;
  }
});
