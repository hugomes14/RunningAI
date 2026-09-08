const sourceInputs = document.querySelectorAll('input[name="origem"]');
const sourcePanels = document.querySelectorAll('[data-source]');

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

document.querySelector('#prediction-form')?.addEventListener('submit', (event) => {
  const button = event.currentTarget.querySelector('button[type="submit"]');
  button.disabled = true;
  button.querySelector('span').textContent = 'A calcular…';
});
