import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from skimage.io import imread

from lib.configs import CutOutAreaConfig, CutIntoPiecesConfig, GreenPixelExtractorConfig
from lib.segmentation import UNET
from lib.common import PotCoords

logger = logging.getLogger(__name__)
    
class DistortionCorrectionService:

    @classmethod
    def get_calibration_images(cls, dir_path : Path):
        files = dir_path.glob("*")
        return [imread(f) for f in files] 

    @classmethod
    def get_calibration_parameters(cls, calibration_images : List[np.ndarray]) -> Tuple:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((6*9, 3), np.float32)
        objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        
        objpoints = []
        imgpoints = []
        
        for cur in calibration_images:
            cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(cur, (9, 6), None)
            if ret == True:
               corners2 = cv2.cornerSubPix(cur, corners, (11, 11), (-1, -1), criteria)
               
               objpoints.append(objp) 
               imgpoints.append(corners2)
               
        
        _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, cur.shape[::-1], None, None)
        return mtx, dist

    @classmethod
    def get_pot_coords(cls, calibration_images : List[np.ndarray]) -> List[PotCoords]:
        pot_coords = [{"x":100,"y":80,"r":90},{"x":112,"y":279,"r":90},{"x":84,"y":480,"r":90},{"x":364,"y":94,"r":90},{"x":364,"y":292,"r":90},{"x":350,"y":467,"r":90},{"x":563,"y":96,"r":90},{"x":560,"y":286,"r":90},{"x":556,"y":469,"r":90},{"x":759,"y":98,"r":90},{"x":759,"y":262,"r":90},{"x":757,"y":469,"r":90},{"x":985,"y":107,"r":90},{"x":990,"y":286,"r":90},{"x":983,"y":471,"r":90}]
        return [PotCoords(**i) for i in pot_coords]

    def __init__(self) -> None:
        pass
               
    def process(
        self, 
        img : np.ndarray, 
        mtx : np.ndarray, 
        dist : np.ndarray
        ):
        h, w = img.shape[:2]
        
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        
        img = dst[y:y+h, x:x+w]
        return img

class CutOutAreaService:

    def __init__(self, config : CutOutAreaConfig):
        logger.info(f"Initializing {self.__class__.__name__}")
        self.config = config

    def process(self, img : np.ndarray, ) -> np.ndarray:
        center = self.config.upper_left[::-1]

        matrix = cv2.getRotationMatrix2D(center=center, angle=self.config.theta, scale=1)
        rotated_img = cv2.warpAffine(src=img, M=matrix, dsize=img.shape[:2][::-1])

        x, y = self.config.upper_left[::-1]

        rotated_img = rotated_img[ y:y+self.config.h, x:x+self.config.w ]
        return rotated_img

class CutIntoPiecesService:

    def __init__(self, config : CutIntoPiecesConfig):
        logger.info(f"Initializing {self.__class__.__name__}")
        self.config = config

    def tiles_from_lines(self, img : np.ndarray) -> List[np.ndarray]:
        vertical_lines = [0] + self.config.vertical_lines + [img.shape[1]]
        horizontal_lines = [0] + self.config.horizontal_lines + [img.shape[0]]

        tiles = []
        for i in range(len(vertical_lines) - 1):
            for j in range(len(horizontal_lines) - 1):
                tiles.append(
                    img[
                        horizontal_lines[j]:horizontal_lines[j+1],
                        vertical_lines[i]:vertical_lines[i+1]
                    ]
                )
        return tiles

    def tiles_from_pot_coords(self, img : np.ndarray, pot_coords_list : List[PotCoords]) -> List[np.ndarray]:
        tiles = []
        for pot_coords in pot_coords_list:
            tiles.append(
                img[
                    max(0, pot_coords.y - pot_coords.r):min(img.shape[0], pot_coords.y + pot_coords.r),
                    max(0, pot_coords.x - pot_coords.r):min(img.shape[1], pot_coords.x + pot_coords.r)
                ]
            )
        return tiles

    def process(self, img : np.ndarray, pot_coords_list : List[PotCoords]) -> List[np.ndarray]:
        #if self.config.pot_coords_list != None:
        tiles = self.tiles_from_pot_coords(img, pot_coords_list)
        #else:
        #    tiles = self.tiles_from_lines(img)  
        return tiles

class ExtractGreenPixelsService:

    def __init__(self, config : GreenPixelExtractorConfig) -> None:
        self.config = config

        self.model = torch.load(self.config.checkpoint_path, map_location=torch.device('cpu'))
        self.model.to(self.config.device)
        self.model.eval()

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.config.expected_height, self.config.expected_width)),            
        ])        

    @staticmethod
    def predb_to_mask(predb):
        p = torch.functional.F.softmax(predb[0], 0)
        return p.argmax(0).cpu()

    def count(self, mask : np.ndarray):
        return np.count_nonzero(mask)

    def extract(self, img : np.ndarray):
        img_tensor = self.transforms(img).unsqueeze(0)
        img_tensor = img_tensor.to(self.model.device)

        with torch.no_grad():
            mask = self.predb_to_mask(self.model(img_tensor))
        return mask
