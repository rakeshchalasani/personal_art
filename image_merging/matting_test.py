import matting
import numpy as np

class MattingTest():
    def __init__(self) -> None:
        image = self.__gen_image(100, 100)
        background = self.__gen_background(100, 100)
        self.cutter = matting.ImageCut(image, background)

    def cut(self):
        print(self.cutter.image.shape)
        print(self.cutter.background.shape)
        out = self.cutter.cut()

        

    def __gen_image(self, x = 100, y = 100):
        image = np.zeros([x, y, 3], dtype = np.uint8)
        image[50:100, 50:100, :] = 255
        return image

    def __gen_background(self, x = 100, y = 100):
        background_mask = np.zeros([x, y], dtype = np.uint8)
        background_mask[80:100, 80:100] = 1
        return background_mask

if __name__ == "__main__":
    matting_test = MattingTest()
    print(matting_test.cut())
