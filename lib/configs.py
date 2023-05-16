from pathlib import Path
from dataclasses import dataclass
from typing import List
import dacite
import json
from lib.common import PotCoords

import logging
logger = logging.getLogger(__name__)

@dataclass
class CutOutAreaConfig:

    upper_left : List[int]
    h : int
    w : int
    theta : float

    def __post_init__(self):
        self.upper_left = tuple(self.upper_left)

@dataclass
class CutIntoPiecesConfig:

    horizontal_lines : List[int]
    vertical_lines : List[int]
    pot_coords_list : List[PotCoords] = None

@dataclass
class MainConfig:
    output_dir : Path

@dataclass
class CalibrationConfig:
    dir_path : Path

@dataclass
class GreenPixelExtractorConfig:
    checkpoint_path : Path
    expected_height : int
    expected_width : int
    device : str = "cpu"

type_hooks = {Path : Path}

@dataclass
class ModelConfig:
    main_config : MainConfig
    cut_out_area_config : CutOutAreaConfig
    cut_into_pieces_config : CutIntoPiecesConfig
    green_pixel_extractor_config : GreenPixelExtractorConfig

    @classmethod
    def from_path(cls, path : Path):
        config_json = get_config_json_merged(path)
        config = dacite.from_dict(data_class=ModelConfig, data=config_json, config=dacite.Config(type_hooks=type_hooks))
        return config

def get_config_json(path : Path):
    with open(path, "r", encoding="utf-8") as f:
        config_json = json.load(f)
    return config_json

def update_dict_recursive(first, second):
    for k in second.keys():
        if (type(first.get(k, None)) == type(second[k]) == dict):
            update_dict_recursive(first[k], second[k])
        else:
            first[k] = second[k]

def get_config_json_merged(path : Path):
    
    config = dict()
    try:
        config = get_config_json(path)
    except IOError:
        message = f"No settings found for path: {path}"
        logger.error(message)
        raise IOError(message)

    try:
        config_manual = get_config_json(path.with_suffix(".manual" + path.suffix))
    except IOError:
        logger.error(f"No manual settings found for path: {path}")
    else:
        logger.info("Using manual settings")
        update_dict_recursive(config, config_manual)

    return config