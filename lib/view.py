from dataclasses import dataclass
from lib.model import CalibrationParameters
from typing import List
from lib.configs import PotCoords

@dataclass
class ArrayInfo:

    dtype : str
    shape : List[int]
    data : str

@dataclass
class FileInfo:
    mime_type : str
    data : str # base 64 encoded file

@dataclass
class PotInfo:
    #image : ArrayInfo
    #index : int
    green_pixel_count : int
        
@dataclass
class FarmInfo:
    file_name : str
    work_area : ArrayInfo
    pot_infos : List[PotInfo]

@dataclass
class CalibrationParametersInfo:
    mtx : ArrayInfo # camera matrix
    dist : ArrayInfo # distortion coefficients
    pots_coords : List[PotCoords] = None # массив с положениями горшков в рабочей области