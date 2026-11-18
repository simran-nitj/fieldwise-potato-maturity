document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('images');
  const previews = document.getElementById('preview-list');
  const feedback = document.getElementById('upload-feedback');
  const zone = document.querySelector('.dropzone');
  const form = document.getElementById('assessment-form');
  if (!input || !previews) return;

  const imageTypes = new Set(['image/jpeg', 'image/png']);
  const maxImages = 12;
  const maxBytes = 8 * 1024 * 1024;
  const showFiles = (fileList) => {
    const files = [...fileList];
    previews.replaceChildren();
    let issue = '';
    if (files.length > maxImages) issue = `Choose no more than ${maxImages} images.`;
    for (const file of files) {
      if (!imageTypes.has(file.type) && !/\.(jpe?g|png)$/i.test(file.name)) issue ||= `${file.name}: choose a JPG, JPEG, or PNG image.`;
      if (file.size > maxBytes) issue ||= `${file.name}: each image must be 8 MB or smaller.`;
      const card = document.createElement('div');
      card.className = 'preview-card';
      const image = document.createElement('img');
      image.alt = `Preview of ${file.name}`;
      image.loading = 'lazy';
      if (imageTypes.has(file.type)) {
        const objectUrl = URL.createObjectURL(file);
        image.src = objectUrl;
        image.addEventListener('load', () => URL.revokeObjectURL(objectUrl), { once: true });
        image.addEventListener('error', () => URL.revokeObjectURL(objectUrl), { once: true });
      }
      const name = document.createElement('strong');
      name.textContent = file.name;
      const size = document.createElement('small');
      size.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
      card.append(image, name, size);
      previews.append(card);
    }
    if (feedback) {
      feedback.textContent = issue || (files.length ? `${files.length} image${files.length === 1 ? '' : 's'} ready for analysis.` : '');
      feedback.classList.toggle('error-text', Boolean(issue));
      feedback.classList.toggle('success-text', Boolean(files.length && !issue));
    }
    input.setCustomValidity(issue);
    return !issue;
  };

  input.addEventListener('change', () => showFiles(input.files));
  if (zone) {
    zone.addEventListener('dragover', (event) => {
      event.preventDefault();
      zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', (event) => {
      if (!zone.contains(event.relatedTarget)) zone.classList.remove('dragover');
    });
    zone.addEventListener('drop', (event) => {
      event.preventDefault();
      zone.classList.remove('dragover');
      const dropped = event.dataTransfer?.files;
      if (!dropped?.length) return;
      try {
        const transfer = new DataTransfer();
        [...dropped].forEach((file) => transfer.items.add(file));
        input.files = transfer.files;
        showFiles(input.files);
      } catch (_error) {
        if (feedback) feedback.textContent = 'This browser could not attach dropped files. Use “browse files” instead.';
      }
    });
  }
  form?.addEventListener('submit', (event) => {
    const files = input.files;
    if (!files.length || !showFiles(files)) {
      event.preventDefault();
      input.reportValidity();
      return;
    }
    const button = form.querySelector('button[type="submit"]');
    if (button) {
      button.disabled = true;
      button.setAttribute('aria-busy', 'true');
      button.innerHTML = '<span class="button-spinner" aria-hidden="true"></span> Analyzing images…';
    }
    if (feedback) feedback.textContent = 'Validating and analyzing your images. This may take a moment.';
  });
});
