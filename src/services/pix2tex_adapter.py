"""pix2tex 0.1.4 preprocessing without CLI clipboard writes or process chdir.

Adapted from the MIT-licensed pix2tex.cli.LatexOCR.__call__ implementation:
https://github.com/lukas-blecher/LaTeX-OCR/blob/main/pix2tex/cli.py
"""
from PIL import Image
import numpy as np


def load_pix2tex(weights, device):
    """Use the bundled architecture/config with absolute paths and no CLI side effects."""
    from pathlib import Path
    from types import SimpleNamespace
    import pix2tex
    import torch
    import yaml
    from munch import Munch
    from transformers import PreTrainedTokenizerFast
    from pix2tex.models import get_model
    from timm.models.resnetv2 import ResNetV2
    from timm.models.layers import StdConv2dSame

    package = Path(pix2tex.__file__).parent / 'model'
    args = Munch(yaml.safe_load((package / 'settings/config.yaml').read_text()))
    args.device = device
    args.wandb = False
    args.encoder_structure = args.get('encoder_structure', 'hybrid')
    args.max_dimensions = [args.max_width, args.max_height]
    args.min_dimensions = [args.get('min_width', 32), args.get('min_height', 32)]
    args.decoder_args = args.get('decoder_args') or {}
    network = get_model(args)
    network.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    network.eval()
    resizer_path = Path(weights).with_name('image_resizer.pth')
    resizer = None
    if resizer_path.is_file():
        resizer = ResNetV2(layers=[2, 3, 3], num_classes=max(args.max_dimensions)//32,
            global_pool='avg', in_chans=1, drop_rate=.05, preact=True,
            stem_type='same', conv_layer=StdConv2dSame).to(device)
        resizer.load_state_dict(torch.load(resizer_path, map_location=device, weights_only=True))
        resizer.eval()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(package / args.tokenizer))
    return SimpleNamespace(args=args, model=network, image_resizer=resizer, tokenizer=tokenizer)


def predict_pix2tex(model, image):
    from pix2tex.cli import minmax_size
    from pix2tex.dataset.transforms import test_transform
    from pix2tex.utils import pad, post_process, token2str

    image = minmax_size(pad(image), model.args.max_dimensions, model.args.min_dimensions)
    if model.image_resizer is not None:
        source = image.convert('RGB').copy()
        ratio, width, height = 1, source.width, source.height
        for _ in range(10):
            height = max(1, int(height * ratio))
            resized = source.resize((width, height), Image.Resampling.BILINEAR if ratio > 1 else Image.Resampling.LANCZOS)
            image = pad(minmax_size(resized, model.args.max_dimensions, model.args.min_dimensions))
            tensor = test_transform(image=np.array(image.convert('RGB')))['image'][:1].unsqueeze(0)
            width = (model.image_resizer(tensor.to(model.args.device)).argmax(-1).item()+1)*32
            if width == image.width:
                break
            ratio = width/image.width
    else:
        tensor = test_transform(image=np.array(pad(image).convert('RGB')))['image'][:1].unsqueeze(0)
    tokens = model.model.generate(tensor.to(model.args.device), temperature=model.args.get('temperature', .25))
    return post_process(token2str(tokens, model.tokenizer)[0])
