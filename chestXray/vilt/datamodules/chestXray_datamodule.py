from vilt.datasets import CHESTXRAYDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class CHESTXRAYDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CHESTXRAYDataset

    @property
    def dataset_name(self):
        return "chestXray"

    def setup(self, stage):
        super().setup(stage)

