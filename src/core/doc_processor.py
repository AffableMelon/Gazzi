from src.utils.convert import pdf_to_images
from src.utils.convert import preprocess_image
from PIL import Image


class DocProcessor:
    # def __init__(self):
    #     self.output_dir = "./data/processed/images"

    def process(self, document):
        if document.is_pdf():
            for page in pdf_to_images(str(document.path)):
                yield preprocess_image(page)
        else:
            images = Image.open(document.path)
            yield preprocess_image(images)
