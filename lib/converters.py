from lib.model import Pot, Farm, CalibrationParameters
from lib.view import PotInfo, FarmInfo, CalibrationParametersInfo
from lib.serializers import Serializer

class Converter:

    def __init__(self, serializer : Serializer):
        self.serializer = serializer

    def calibration_parameters_to_info(self, item : CalibrationParameters) -> CalibrationParametersInfo:
        return CalibrationParametersInfo(
            mtx=self.serializer.save_ndarray(item.mtx),
            dist=self.serializer.save_ndarray(item.dist),
            pots_coords=[self.serializer.save(i) for i in item.pot_coords]
        )

    def calibration_parameters_from_info(self, info : CalibrationParametersInfo) -> CalibrationParameters:
        return CalibrationParameters(
            self.serializer.load_ndarray(info.mtx),
            self.serializer.load_ndarray(info.dist),
            info.pots_coords
        )

    def pot_to_info(self, item : Pot) -> PotInfo:
        return PotInfo(
            green_pixel_count=item.green_pixel_count   
        )

    def pot_from_info(self, info : PotInfo) -> Pot:
        return Pot(
            image=self.serializer.load_ndarray(info.image),
            index=info.index,
            green_pixel_count=info.green_pixel_count
        )

    def farm_to_info(self, item : Farm) -> FarmInfo:
        return FarmInfo(
            file_name=item.file_name,
            work_area=self.serializer.save_ndarray(item.work_area),
            pot_infos=[
                self.pot_to_info(i) for i in item.pots
            ]
        )

    def farm_from_info(self, info : FarmInfo) -> Farm:
        return Farm(
            file_name=info.file_name,
            work_area=self.serializer.load_ndarray(info.work_area),
            pots=[
                self.pot_from_info(i) for i in info.pot_infos
            ]
        )