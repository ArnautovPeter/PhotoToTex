from pathlib import Path

import cv2
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download

filepath = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt",
)
model = YOLOv10(filepath)

folder = Path(__file__).resolve().parent

det_res = model.predict(
    folder / "helemski.png", task="segment", imgsz=1024, conf=0.2, device="cuda:0"
)

annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
cv2.imwrite(folder / "result.jpg", annotated_frame)
