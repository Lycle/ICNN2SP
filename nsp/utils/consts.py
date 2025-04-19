from enum import Enum


class DataManagerModes(Enum):
    GEN_INSTANCE = 1
    GEN_DATASET_E = 2


class LearningModelTypes(Enum):
    icnn_e = 1
    nn_e = 2


class LossPenaltyTypes(Enum):
    none = 0
    lasso = 1
    ridge = 2
    elastic = 3
