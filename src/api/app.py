"""FastAPI transport layer for a small local model workbench."""
from __future__ import annotations
import logging
from enum import Enum
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError

from src.services.recognition import RecognitionService, ModelUnavailable

WEB = Path(__file__).resolve().parents[1] / 'web'
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


class Backend(str, Enum):
    pix2tex = 'pix2tex'
    baseline = 'baseline'


class Prediction(BaseModel):
    latex: str
    model: str
    inference_seconds: float
    load_seconds: float


def decode_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError('Выберите изображение размером до 10 МБ.')
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in ('PNG', 'JPEG', 'WEBP', 'BMP'):
                raise ValueError('Поддерживаются PNG, JPEG, WebP и BMP.')
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError('Слишком большое разрешение изображения.')
            source.verify()
        with Image.open(BytesIO(data)) as source:
            oriented = ImageOps.exif_transpose(source).convert('RGBA')
            background = Image.new('RGBA', oriented.size, 'white')
            return Image.alpha_composite(background, oriented).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError('Файл не является корректным изображением.') from error


def create_app(service=None) -> FastAPI:
    service = service or RecognitionService()
    app = FastAPI(title='PhotoToTex', version='0.2.0', description='Локальное распознавание кропов формул. Без обработки страниц.')
    app.state.recognizer = service
    app.mount('/static', StaticFiles(directory=WEB / 'static'), name='static')

    @app.middleware('http')
    async def bound_upload(request: Request, call_next):
        if request.method == 'POST' and request.url.path == '/api/predict':
            try:
                length = int(request.headers.get('content-length', '0'))
            except ValueError:
                return JSONResponse({'detail': 'Некорректный размер запроса.'}, status_code=400)
            if length <= 0:
                return JSONResponse({'detail': 'Не указан размер загрузки.'}, status_code=411)
            if length > MAX_UPLOAD_BYTES + 64*1024:
                return JSONResponse({'detail': 'Изображение превышает 10 МБ.'}, status_code=413)
        return await call_next(request)

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(WEB / 'templates/index.html')

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'models_loaded': any(item['loaded'] for item in service.models())}

    @app.get('/api/models')
    def models():
        return {'models': service.models(), 'default': 'pix2tex'}

    @app.get('/api/model')
    def model():
        try:
            return service.metadata()
        except Exception as error:
            logging.exception('Cannot read baseline metadata')
            raise HTTPException(503, 'Метрики baseline недоступны.') from error

    @app.post('/api/predict', response_model=Prediction)
    async def predict(file: UploadFile = File(...), model: Backend = Form(Backend.pix2tex)):
        try:
            data = await file.read(MAX_UPLOAD_BYTES + 1)
        finally:
            await file.close()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, 'Изображение превышает 10 МБ.')
        try:
            image = await run_in_threadpool(decode_image, data)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        try:
            return await run_in_threadpool(service.predict, image, model.value)
        except BlockingIOError as error:
            raise HTTPException(429, str(error)) from error
        except ModelUnavailable as error:
            raise HTTPException(503, str(error)) from error
        except Exception as error:
            logging.exception('Inference failed')
            raise HTTPException(500, 'Не удалось распознать формулу. Проверьте журнал сервера.') from error

    return app


app = create_app()
