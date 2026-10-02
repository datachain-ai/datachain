from enum import Enum


class JobStatus(int, Enum):
    CREATED = 1
    SCHEDULED = 10
    PROVISIONING = 12
    QUEUED = 2
    INIT = 3
    RUNNING = 4
    COMPLETE = 5
    FAILED = 6
    CANCELING = 7
    CANCELED = 8
    CANCELING_SCHEDULED = 9
    TASK = 11
    PENDING = 13
    DID_NOT_RUN = 14

    @classmethod
    def finished(cls) -> tuple[int, ...]:
        return cls.COMPLETE, cls.FAILED, cls.CANCELED, cls.TASK, cls.DID_NOT_RUN


class JobQueryType(int, Enum):
    PYTHON = 1
    SHELL = 2
