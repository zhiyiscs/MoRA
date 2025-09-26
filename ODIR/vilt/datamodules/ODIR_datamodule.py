from vilt.datasets import ODIRDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class ODIRDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return ODIRDataset

    @property
    def dataset_name(self):
        return "ODIR"

    def setup(self, stage):
        super().setup(stage)

