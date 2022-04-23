from PIL import Image

#refs https://holypython.com/python-pil-tutorial/creating-photo-collages/#:~:text=An%20image%20collage%20can%20easily,to%20the%20grid%20through%20iteration.
class ImageGrid:
    def __init__(self, rows, columns, patch_size):
        self.rows = rows
        self.columns = columns
        self.patch_size = patch_size
        self.canvas = Image.new("RGB", (columns*patch_size, (rows)*patch_size))
    def draw(self, row, column, image):
        self.canvas.paste(image.resize((self.patch_size, self.patch_size)), (self.patch_size*column, self.patch_size*row))
    def save(self, filename):
        self.canvas.save(filename)
    def show(self):
        self.canvas.show()