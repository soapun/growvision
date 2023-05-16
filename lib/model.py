import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import dacite
import numpy as np
from skimage.io import imread

from lib.services import DistortionCorrectionService, CutIntoPiecesService, CutOutAreaService, ExtractGreenPixelsService
from lib.segmentation import UNET
from lib.common import PotCoords

from numpy import ndarray

logger = logging.getLogger(__name__)

@dataclass
class CalibrationParameters:
    mtx : ndarray
    dist : ndarray
    pot_coords : List[PotCoords]

@dataclass
class Pot:
    image : ndarray
    index : int
    green_pixel_count : int

@dataclass
class Farm:
    file_name : str
    work_area : ndarray
    pots : List[Pot]

@dataclass
class Model:

    distortion_corrector : DistortionCorrectionService
    cut_out_area_service : CutOutAreaService
    cut_into_pieces_service : CutIntoPiecesService
    green_pixel_extractor : ExtractGreenPixelsService

    @classmethod
    def configure_services(cls, config):
        distortion_corrector = DistortionCorrectionService()

        cut_out_area_service = CutOutAreaService(config.cut_out_area_config)
        cut_into_pieces_service = CutIntoPiecesService(config.cut_into_pieces_config)
        green_pixel_extractor = ExtractGreenPixelsService(config.green_pixel_extractor_config)

        return distortion_corrector, cut_out_area_service,  cut_into_pieces_service, green_pixel_extractor

    @classmethod
    def from_config(cls, config ):
        services = cls.configure_services(config)
        return cls(*services)


    def get_distortion_correction_parameters(self, calibration_images : List[np.ndarray]) -> CalibrationParameters:
        result = CalibrationParameters(
            *self.distortion_corrector.get_calibration_parameters(calibration_images)
        )
        return result        

    def get_info(self, img : np.ndarray, calibration_parameters : CalibrationParameters) -> Farm:
        mtx, dist = calibration_parameters.mtx, calibration_parameters.dist
        
        img_undistorted = self.distortion_corrector.process(
            img, 
            calibration_parameters.mtx,
            calibration_parameters.dist
        )

        work_area = self.cut_out_area_service.process(img_undistorted)

        tiles = self.cut_into_pieces_service.process(work_area, calibration_parameters.pot_coords)
        tiles_counts = [
            self.green_pixel_extractor.count(self.green_pixel_extractor.extract(tile))
            for tile in tiles
        ]

        result = Farm("", work_area, pots=[
            Pot(tile, k, tile_count)
            for k, (tile, tile_count) in enumerate(zip(tiles, tiles_counts))
        ])

        return result