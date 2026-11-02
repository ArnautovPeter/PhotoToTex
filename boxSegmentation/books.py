from pathlib import Path

from pdf2image import convert_from_path, pdfinfo_from_path

DATA = Path(__file__).resolve().parent / "data/"


class Book:
    def __init__(self, filepath=None, title=None) -> None:
        self.filepath = filepath

        book_num = sum([i.is_dir() for i in DATA.iterdir()])
        self.title = title if title else f"book_{book_num}"

    def extract_images(self, data_path=None):
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
            for i in range(1, page_cnt + 1, batch_size):
                images = convert_from_path(
                    self.filepath, first_page=i, last_page=i + batch_size - 1
                )
                for j in range(len(images)):
                    images[j].save(data_path / f"page_{i + j}.jpg")
                    print(f"image saved to {data_path / f'page_{i + j}'}")
