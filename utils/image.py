import base64
import tarfile
from PIL import Image
from IPython.display import display, HTML


class ImageUtil:
    """
    Utility class to extract and display images from a tar file
    """

    def __init__(self, tar_path: str):
        self.tar = tarfile.open(tar_path)

    def get_image(self, idx: int):
        return Image.open(self.tar.extractfile(f"{idx}.jpg"))

    def embed_image(self, idx: int):
        image_base64 = base64.b64encode(
            self.tar.extractfile(f"{idx}.jpg").read()).decode('ascii')
        return f'<img src="data:image/jpeg;base64,{image_base64}" />'

    def display_image(self, idx):
        image_html = self.embed_image(idx)
        display(HTML(image_html))
