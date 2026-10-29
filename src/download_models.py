"""Explicitly download the official pix2tex v0.0.1 weights into the project cache."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

MODEL_DIR = Path(__file__).resolve().parents[1] / 'checkpoints/pix2tex'
RELEASE = 'https://github.com/lukas-blecher/LaTeX-OCR/releases/download/v0.0.1'


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {'model': 'pix2tex', 'release': 'v0.0.1', 'files': {}}
    for name in ('weights.pth', 'image_resizer.pth'):
        target = MODEL_DIR / name
        if not target.is_file():
            temporary = target.with_suffix('.download')
            try:
                with urlopen(f'{RELEASE}/{name}', timeout=60) as response, temporary.open('wb') as output:
                    while chunk := response.read(1024*1024):
                        output.write(chunk)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest['files'][name] = {'source': f'{RELEASE}/{name}', 'sha256': digest, 'bytes': target.stat().st_size}
        print(name, target.stat().st_size, 'bytes')
    (MODEL_DIR / 'source.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
