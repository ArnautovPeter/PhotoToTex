"""Local model adapters. No HTTP or frontend dependencies."""
from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from time import perf_counter

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]


class ModelUnavailable(RuntimeError):
    pass


class RecognitionService:
    """Load models on demand and serialize inference to bound memory usage."""

    def __init__(self, checkpoint: str | Path = ROOT / 'checkpoints/baseline.pt', device: str = 'cpu'):
        self.checkpoint = Path(checkpoint).resolve()
        self.device = device
        self._models = {}
        self._lock = Lock()
        self._errors = {}

    def models(self):
        return [
            {'id': 'pix2tex', 'name': 'pix2tex · готовая модель', 'description': 'Предобучена на формулах. Веса скачиваются отдельно; распознавание локальное.', 'loaded': 'pix2tex' in self._models, 'error': self._errors.get('pix2tex')},
            {'id': 'baseline', 'name': 'Baseline · наша модель', 'description': 'Учебная CNN / Transformer для сравнения.', 'loaded': 'baseline' in self._models, 'error': self._errors.get('baseline')},
        ]

    def _load(self, backend):
        if backend not in ('pix2tex', 'baseline'):
            raise ValueError('Неизвестная модель.')
        if backend in self._models:
            return self._models[backend]
        try:
            if backend == 'pix2tex':
                from .pix2tex_adapter import load_pix2tex
                weights = ROOT / 'checkpoints/pix2tex/weights.pth'
                if not weights.is_file():
                    raise ModelUnavailable('Веса pix2tex отсутствуют. Выполните uv run --extra cpu python -m src.download_models.')
                if self.device != 'cpu' and not torch.cuda.is_available():
                    raise ModelUnavailable('CUDA недоступна. Запустите с --device cpu.')
                model = load_pix2tex(weights, self.device)
                self._models[backend] = model
            else:
                from src.infer import load_checkpoint
                self._models[backend] = load_checkpoint(self.checkpoint, torch.device(self.device))
        except ModelUnavailable:
            raise
        except Exception as error:
            logging.exception('Could not load %s', backend)
            self._errors[backend] = 'Не удалось загрузить модель. Проверьте зависимости, веса и доступное устройство.'
            raise ModelUnavailable(self._errors[backend]) from error
        self._errors.pop(backend, None)
        return self._models[backend]

    def predict(self, image: Image.Image, backend: str):
        # Reject additional requests instead of accumulating an unbounded thread queue.
        if not self._lock.acquire(blocking=False):
            raise BlockingIOError('Модель занята. Дождитесь завершения предыдущего запроса.')
        try:
            started = perf_counter()
            model = self._load(backend)
            load_seconds = perf_counter() - started
            started = perf_counter()
            with torch.inference_mode():
                if backend == 'pix2tex':
                    from .pix2tex_adapter import predict_pix2tex
                    latex = predict_pix2tex(model, image)
                else:
                    network, tokenizer, config = model
                    # Unlike the old helper, put input on the configured device.
                    from src.data.dataset import preprocess_image
                    tensor = preprocess_image(image, config['image_size']).unsqueeze(0).to(self.device)
                    tokens = network.greedy_decode(tensor, tokenizer.sos_id, tokenizer.eos_id)[0].tolist()
                    latex = tokenizer.decode(tokens)
            return {'latex': latex, 'model': backend, 'inference_seconds': round(perf_counter()-started, 3), 'load_seconds': round(load_seconds, 3)}
        finally:
            self._lock.release()

    def metadata(self):
        """Retained baseline training metrics; never label these pix2tex metrics."""
        import json
        from src.data.dataset import split_annotation
        if not self.checkpoint.is_file():
            return {'available': False, 'checkpoint': self.checkpoint.name}
        # Read metadata once, independently from a request to run inference.
        with self._lock:
            if 'baseline_metadata' not in self._models:
                checkpoint = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
                config = checkpoint['config']
                self._models['baseline_metadata'] = {
                    'available': True, 'checkpoint': self.checkpoint.name,
                    'epoch': checkpoint.get('epoch'), 'validation': checkpoint.get('validation', {}),
                    'config': config, 'vocab_size': len(checkpoint['tokenizer']['tokens']),
                    'parameter_count': sum(v.numel() for v in checkpoint['model'].values()),
                    'history': checkpoint.get('history', []),
                }
            info = dict(self._models['baseline_metadata'])
        for suffix, key, default in [('history', 'history', {'history': info['history']}), ('metrics', 'metrics', {'splits': {}})]:
            path = self.checkpoint.with_suffix(f'.{suffix}.json')
            payload = json.loads(path.read_text()) if path.is_file() else default
            info[key] = payload.get('history', []) if key == 'history' else payload
        train_file = split_annotation(ROOT / info['config']['data_dir'], 'train')
        info['dataset_size'] = sum(bool(line.strip()) for line in train_file.read_text().splitlines()) if train_file.is_file() else None
        return info
