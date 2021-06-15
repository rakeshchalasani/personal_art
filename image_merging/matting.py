from typing import FrozenSet
import numpy as np

# Given a binary segmentation map, cuts or replace the background while preserving foreground. Doesn't use any matting techniluqes.
class ImageCut():

    def __init__(self, image, background) -> None:
        self.image = image
        self.background = background # Optional parameter, passing Non equivalent to remove background. 

    # Cut background and return on foreground objects. 
    def cut(self):
        out_img = np.ones(self.image.shape,dtype=self.image.dtype)*255
        
        out_img = np.where(np.repeat(self.background[..., None],3,axis=2) == 1, self.image, out_img)

        return out_img








