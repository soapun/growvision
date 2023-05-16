import io
from numpy import ndarray, frombuffer, dtype, ascontiguousarray
from typing import Dict
from pathlib import Path
import dacite
import base64
from dataclasses import asdict
from skimage.io import imread

import tempfile

import json
import logging

from mimetypes import types_map
from lib.view import ArrayInfo, FileInfo

logger = logging.getLogger(__name__)

class Serializer:

    @staticmethod
    def save_ndarray(arr : ndarray) -> ArrayInfo:
        return ArrayInfo(
            dtype=str(arr.dtype),
            shape=list(arr.shape),
            data=base64.b64encode(ascontiguousarray(arr)).decode("utf-8")
        )

    @staticmethod
    def load_ndarray(arr_info : ArrayInfo) -> ndarray:


        data_type = dtype(arr_info.dtype)
        arr = frombuffer(base64.decodebytes(arr_info.data.encode('utf-8')), data_type)
        arr.shape = tuple(arr_info.shape)
        return arr

    @staticmethod
    def load_image_file(file_info : FileInfo) -> ndarray:
        arr_data = io.BytesIO(base64.decodebytes(file_info.data.encode("utf-8")))
        arr = imread(arr_data)
        
        return arr        

    @staticmethod
    def save_image_file(file_path : Path):
        with open(file_path, "rb") as f:
            return FileInfo(
                mime_type=types_map[file_path.suffix],
                data=base64.b64encode(f.read()).decode("utf-8")
            ) 

    @classmethod
    def load(cls, data : Dict, data_class : type) -> object:
        return dacite.from_dict(
            data_class=data_class, data=data, config=dacite.Config(
                type_hooks=type_hooks
            )
        )

    @classmethod
    def save(cls, data) -> Dict:
        return asdict(data)

type_hooks = {Path : Path}