from pathlib import Path

import cv2
from doclayout_yolo import YOLOv10

# from huggingface_hub import hf_hub_download

folder = Path(__file__).resolve().parent

# filepath = hf_hub_download(
#     local_dir=folder,
#     repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
#     filename="doclayout_yolo_docstructbench_imgsz1024.pt",
# )

filepath = folder / "doclayout_yolo_docstructbench_imgsz1024.pt"

print(filepath)
model = YOLOv10(filepath)


det_res = model.predict(
    folder / "idor.png", task="segment", imgsz=1024, conf=0.2, device="cuda:0"
)

annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
cv2.imwrite(folder / "result4.jpg", annotated_frame)

# print(annotated_frame)
