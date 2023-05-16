from lib.model import Model

from lib.serializers import Serializer
from typing import List
from lib.segmentation import UNET
from dataclasses import dataclass
from numpy import ndarray
from functools import wraps, partial

from lib.requests import GetInfoRequest, GetCalibrationParametersRequest, GetInfoResponse, GetCalibrationParametersResponse
from lib.model import CalibrationParameters
from lib.converters import Converter
from lib.view import FarmInfo, CalibrationParametersInfo

import logging
logger = logging.getLogger(__name__)

def load(load_func):
    def decorator(func):
        @wraps(func)
        def wrapper(self, request):
            return func(self, load_func(request))
        return wrapper
    return decorator

def save(save_func):
    def decorator(func):
        @wraps(func)
        def wrapper(self, request):
            return save_func(func(self,request))
        return wrapper
    return decorator
    

class Controller:

    def __init__(self, model : Model, converter:Converter=None, serializer:Serializer=None) -> None:
        self.model = model

        if serializer is None:
            self.serializer = Serializer()
        else:
            self.serializer = serializer

        if converter is None:
            self.converter = Converter(self.serializer)
        else:
            self.converter = converter

        if serializer is None:
            self.serializer = Serializer()
        else:
            self.serializer = serializer

    def get_info(self, request : GetInfoRequest) -> GetInfoResponse:
        request = self.serializer.load(request, GetInfoRequest)

        image = self.serializer.load_image_file(request.image)
        calibration_parameters = self.converter.calibration_parameters_from_info(request.calibration_parameters)

        farm = self.model.get_info(
            image,
            calibration_parameters
        )
        info = self.converter.farm_to_info(farm)
        result = GetInfoResponse(
            info.file_name,
            info.work_area,
            info.pot_infos
        )

        result = self.serializer.save(result)

        return result

    def get_calibration_parameters(self, request : GetCalibrationParametersRequest) -> GetCalibrationParametersResponse:
        request = self.serializer.load(request, GetCalibrationParametersRequest)
        
        calibration_images = [
            self.serializer.load_image_file(i)
            for i in request.calibration_images    
        ]
        
        mtx, dist = self.model.distortion_corrector.get_calibration_parameters(calibration_images)
        pot_coords = self.model.distortion_corrector.get_pot_coords(calibration_images)

        calibration_parameters = CalibrationParameters(mtx, dist, pot_coords)

        result = self.converter.calibration_parameters_to_info(calibration_parameters)
        result = self.serializer.save(result)

        return result