from typing import FrozenSet
import numpy as np

# Given a binary segmentation map, cuts or replace the background while preserving foreground. Doesn't use any matting techniluqes.
class ImageCut():

    def __init__(self, image, mask, background = None) -> None:
        self.image = image
        self.mask = mask # mask {0, 1} determing foreground vs background.
        self.background = background # Optional parameter, passing Non equivalent to remove background. 

    # Cut background and return on foreground objects. 
    def cut(self):
        if self.background:
            out_img = self.background
        else: 
            out_img = np.ones(self.image.shape,dtype=self.image.dtype)*255
        
        out_img = np.where(np.repeat(self.mask[..., None],3,axis=2) == 1, self.image, out_img)

        return out_img



class AlphaMatting():

    def __init__(self, image, alpha, background = None) -> None:
        self.image = image
        self.alpha = alpha
        self.background = background # Optional parameter, passing None equivalent to remove background. 

    # Cut background and return on foreground objects. 
    def cut(self):
        if self.background:
            out_img = self.background
        else:
            out_img = np.ones(self.image.shape,dtype=self.image.dtype)*255

        tiled_alpha = np.repeat(self.alpha[..., None],3,axis=2)
        out_img = np.multiply(tiled_alpha, self.image) + np.multiply((1 - tiled_alpha), out_img)

        return out_img






