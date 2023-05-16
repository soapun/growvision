from os import stat
import numpy as np
import configs

class ImageTools:

    @staticmethod
    def _image_colorfulness(img : np.ndarray):
        R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
        return stdRoot + (0.3 * meanRoot)

    @staticmethod
    def _mse(x, y):
        return np.mean((x - y) ** 2)

    def __init__(self, config : configs.ImageToolsConfig):
        self.config = config

    def isNight(self, img : np.ndarray):
        return ImageTools._image_colorfulness(img) <= self.config.isNightThreshold
 
    def isGenBorder(self, prev_mse, cur_mse):
        if self.config.curMseThreshold != None:
            return cur_mse - prev_mse < self.config.isGenBorderThreshold and cur_mse < self.config.curMseThreshold
        else:
            return cur_mse - prev_mse < self.config.isGenBorderThreshold