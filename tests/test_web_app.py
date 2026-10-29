from io import BytesIO
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from src.api.app import create_app, decode_image
from src.services.recognition import ModelUnavailable, RecognitionService


class FakeService:
    def models(self):
        return [{'id':'pix2tex', 'loaded':False}]

    def metadata(self):
        return {'available': True, 'parameter_count':1234, 'metrics':{'splits':{}}}

    def predict(self, image, backend):
        assert image.mode == 'RGB'
        return {'latex':r'\frac{1}{2}', 'model':backend, 'inference_seconds':.12,'load_seconds':0}


def png(mode='RGB'):
    buffer=BytesIO()
    Image.new(mode,(8,8)).save(buffer,format='PNG')
    return buffer.getvalue()


@pytest.fixture
def client():
    with TestClient(create_app(FakeService())) as client:
        yield client


def test_decode_transparent_image_has_white_background():
    image=decode_image(png('RGBA'))
    assert image.mode=='RGB'
    assert image.getpixel((0,0))==(255,255,255)


@pytest.mark.parametrize('content',[b'',b'not an image'])
def test_decode_invalid_image(content):
    with pytest.raises(ValueError): decode_image(content)


def test_upload_uses_multipart_and_returns_timing(client):
    response=client.post('/api/predict',files={'file':('a.png',png(),'image/png')},data={'model':'pix2tex'})
    assert response.status_code==200
    assert response.json()['latex']==r'\frac{1}{2}'
    assert response.json()['inference_seconds']==.12


def test_invalid_file_and_backend(client):
    assert client.post('/api/predict',files={'file':('x.png',b'bad','image/png')}).status_code==400
    assert client.post('/api/predict',files={'file':('x.png',png())},data={'model':'invented'}).status_code==422
    assert client.post('/api/predict',json={'image':'legacy-base64'}).status_code==422


def test_static_api_and_metadata(client):
    assert client.get('/').status_code==200
    assert client.get('/static/app.js').status_code==200
    assert client.get('/static/vendor/katex/katex.min.js').status_code==200
    assert client.get('/api/model').json()['parameter_count']==1234
    assert client.get('/api/health').json()['status']=='ok'
    assert '/api/predict' in client.get('/openapi.json').json()['paths']


@pytest.mark.parametrize('error,code',[(BlockingIOError('busy'),429),(ModelUnavailable('missing weights'),503),(RuntimeError('internal path'),500)])
def test_service_errors_are_explicit(error,code):
    class Broken(FakeService):
        def predict(self,*args):raise error
    with TestClient(create_app(Broken())) as client:
        response=client.post('/api/predict',files={'file':('a.png',png())})
    assert response.status_code==code
    if code==500:assert 'internal path' not in response.text


def test_busy_service_rejects_without_loading():
    service=RecognitionService()
    service._lock.acquire()
    try:
        with pytest.raises(BlockingIOError):service.predict(Image.new('RGB',(8,8)),'pix2tex')
    finally:service._lock.release()


def test_body_limit_rejects_before_image_decode(client):
    assert client.post('/api/predict',content=b'x',headers={'content-length':str(12*1024*1024)}).status_code==413
