from dataclasses import dataclass
from typing import List


from lib.view import FileInfo, FarmInfo, CalibrationParametersInfo
from lib.model import CalibrationParameters

@dataclass 
class GetInfoRequest:
    image : FileInfo
    calibration_parameters : CalibrationParametersInfo

@dataclass
class GetInfoResponse(FarmInfo):
    pass
    # можно убрать лишнюю вложенность
#    info : FarmInfo

@dataclass
class GetCalibrationParametersRequest:
    calibration_images : List[FileInfo]

@dataclass
class GetCalibrationParametersResponse:
    calibration_parameters : CalibrationParametersInfo
    rms : float # rms reprojection error