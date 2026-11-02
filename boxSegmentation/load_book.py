import sys

from books import Book

if __name__ == "__main__":
    if len(sys.argv[1:]) <= 0:
        raise Exception("No book provided.")

    book_path = sys.argv[1:][0]
    book_title = sys.argv[1:][1] if len(sys.argv[1:]) > 1 else None

    book = Book(book_path, book_title)

    book.extract_images()
