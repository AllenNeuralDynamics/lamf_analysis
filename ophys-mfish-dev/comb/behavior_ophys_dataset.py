# This class combines a OphysPlaneDataset and BehaviorDatase to create a BehaviorOphysDataset,
# inputs are raw_folder_path and processed_folder_path
# set at

from comb.behavior_session_dataset import BehaviorSessionDataset
from comb.ophys_plane_dataset import OphysPlaneDataset

from typing import Union, Optional
from pathlib import Path


class BehaviorOphysDataset:
    """A class to combine an OphysPlaneDataset and a BehaviorDataset 
    into a single object.

    All attributes of the Other Dataset classes are available as attributes of this class.

    Example #1:
    Assuming the local folders are stuctured like CodeOcean data assets.

    from data_objects.behavior_ophys_dataset import BehaviorOphysDataset
    processed_path = "/allen/programs/mindscope/workgroups/learning/mattd/co_dev/data/1299958728/processed/"
    plane_folder_path = processed_path + "/1299958728"
    raw_path = "/allen/programs/mindscope/workgroups/learning/mattd/co_dev/data/1299958728/raw"
    bod = BehaviorOphysDataset(raw_path, plane_folder_path)

    Example #2:
    Assume raw and processed data assets are attached to capsule.

    """
    def __init__(self,
                plane_folder_path: Union[str, Path],
                raw_folder_path: Union[str, Path]):

        self.ophys_plane_dataset = OphysPlaneDataset(plane_folder_path=plane_folder_path,raw_folder_path=raw_folder_path)
        self.behavior_dataset = BehaviorSessionDataset(raw_folder_path=raw_folder_path)

    def __getattr__(self, name):
        if hasattr(self.ophys_plane_dataset, name):
            return getattr(self.ophys_plane_dataset, name)
        elif hasattr(self.behavior_dataset, name):
            return getattr(self.behavior_dataset, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __dir__(self):
        return list(set(dir(self.ophys_plane_dataset) + dir(self.behavior_dataset)))
    