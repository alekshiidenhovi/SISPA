from enum import Enum


class TrainingStep(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"


class Operation(Enum):
    ADD = "add"
    REMOVE = "remove"
