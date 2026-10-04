import base64
import json
import threading
from io import BytesIO
from http.client import HTTPConnection
from http.server import HTTPServer

import pytest
from PIL import Image

from src.web_app import decode_image, make_handler


def test_decode_image_converts_to_rgb():
    buffer = BytesIO()
    Image.new("L", (8, 8), color=128).save(buffer, format="PNG")

    image = decode_image(base64.b64encode(buffer.getvalue()).decode("ascii"))

    assert image.mode == "RGB"
    assert image.size == (8, 8)


@pytest.mark.parametrize("encoded", ["not-base64", "", None])
def test_decode_image_rejects_invalid_input(encoded):
    with pytest.raises(ValueError):
        decode_image(encoded)


def test_decode_image_rejects_non_image_data():
    encoded = base64.b64encode(b"not an image").decode("ascii")

    with pytest.raises(ValueError, match="корректным изображением"):
        decode_image(encoded)


def test_predict_endpoint_returns_latex():
    server = HTTPServer(("127.0.0.1", 0), make_handler(lambda image: "x^2"))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    buffer = BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    payload = json.dumps({"image": base64.b64encode(buffer.getvalue()).decode("ascii")})

    try:
        connection = HTTPConnection("127.0.0.1", server.server_port)
        connection.request("POST", "/api/predict", body=payload, headers={"Content-Type": "application/json"})
        response = connection.getresponse()

        assert response.status == 200
        assert json.loads(response.read()) == {"latex": "x^2"}
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_model_endpoint_returns_metadata():
    info = {"parameter_count": 1234, "metrics": {"splits": {}}}
    server = HTTPServer(("127.0.0.1", 0), make_handler(lambda image: "x^2", lambda: info))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    try:
        connection = HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/api/model")
        response = connection.getresponse()

        assert response.status == 200
        assert json.loads(response.read()) == info
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
