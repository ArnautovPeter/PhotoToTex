"""CLI entry point retained for the FastAPI model workbench."""
from __future__ import annotations
import argparse
import uvicorn
from src.api.app import create_app
from src.services.recognition import RecognitionService, ROOT


def main():
    parser = argparse.ArgumentParser(description='Локальный интерфейс PhotoToTex')
    parser.add_argument('--checkpoint', default=str(ROOT / 'checkpoints/baseline.pt'))
    parser.add_argument('--device', default='cpu', help='cpu или cuda:0')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8000, type=int)
    args = parser.parse_args()
    service = RecognitionService(args.checkpoint, args.device)
    uvicorn.run(create_app(service), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
