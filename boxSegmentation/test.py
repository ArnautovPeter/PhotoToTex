# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python (Neovim)
#     language: python
#     name: neovim-python
# ---

# %%
from pathlib import Path

import cv2
from doclayout_yolo import YOLOv10

# from huggingface_hub import hf_hub_download

folder = Path().cwd().resolve() / 'boxSegmentation'

# filepath = hf_hub_download(
#     local_dir=folder,
#     repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
#     filename="doclayout_yolo_docstructbench_imgsz1024.pt",
# )

filepath = folder / "doclayout_yolo_docstructbench_imgsz1024.pt"

print(filepath)
model = YOLOv10(filepath)


det_res = model.predict(
    folder / 'data' / 'irodov' / 'page_10.jpg', task="segment", imgsz=1024, conf=0.2, device="cuda:0"
)

annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
cv2.imwrite(folder / "result5.jpg", annotated_frame)

# print(annotated_frame)

# %%

for box in det_res[0].boxes:
    print(type(det_res[0]))
    # print(det_res[0].names[box.cls.item()])
# %%
print(1)
