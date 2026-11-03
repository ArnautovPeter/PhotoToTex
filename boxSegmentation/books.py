from pathlib import Path

import doclayout_yolo
from doclayout_yolo.engine.results import Boxes
from pdf2image import convert_from_path, pdfinfo_from_path

DATA = Path(__file__).resolve().parent / "data/"


class Book:
    def __init__(self, filepath=None, title=None) -> None:
        self.filepath = filepath

        book_num = sum([i.is_dir() for i in DATA.iterdir()])
        self.title = title if title else f"book_{book_num}"

    def extract_images(self, first_page=None, last_page=None, data_path=None):
        if self.filepath:
            if data_path is None:
                data_path = DATA / self.title

                if data_path.is_dir():
                    for item in data_path.iterdir():
                        item.unlink()
                else:
                    data_path.mkdir()

            print(self.filepath)
            info = pdfinfo_from_path(self.filepath)
            page_cnt = info["Pages"]

            if page_cnt == 0:
                raise Exception("No images found")

            batch_size = 30

            first_page = first_page if first_page else 1
            last_page = last_page if last_page else page_cnt

            for i in range(first_page, last_page + 1, batch_size):
                images = convert_from_path(
                    self.filepath,
                    first_page=i,
                    last_page=max(last_page, i + batch_size - 1),
                )
                for j in range(len(images)):
                    images[j].save(data_path / f"page_{i + j}.jpg")
                    print(f"image saved to {data_path / f'page_{i + j}'}")


class Block:
    def __init__(self) -> None:
        self.xyxy = None
        self.cls = None
        self.page_num = None
        self.book = None
        self.names = None
        self.class_name = None

    @classmethod
    def from_yolo_box(
        cls,
        box: Boxes,
        page_num: int | None = None,
        book: Book | None = None,
        names: dict[int, str] | None = None,
    ):
        block = cls()
        block.xyxy = box.xyxy  # [xmin, ymin, xmax, ymax]
        block.cls = int(box.cls.item())
        block.page_num = page_num
        block.book = book
        block.names = names
        block.class_name = names[block.cls] if names else None

        return block
