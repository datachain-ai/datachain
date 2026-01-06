class DataChainError(RuntimeError):
    pass


class SchemaDriftError(DataChainError):
    pass


class InvalidDatasetNameError(RuntimeError):
    pass


class InvalidNamespaceNameError(RuntimeError):
    pass


class InvalidProjectNameError(RuntimeError):
    pass


class NotFoundError(Exception):
    pass


class NamespaceNotFoundError(NotFoundError):
    pass


class NotAllowedError(Exception):
    pass


class NamespaceCreateNotAllowedError(NotAllowedError):
    pass


class ProjectCreateNotAllowedError(NotAllowedError):
    pass


class ProjectDeleteNotAllowedError(NotAllowedError):
    pass


class NamespaceDeleteNotAllowedError(NotAllowedError):
    pass


class ProjectNotFoundError(NotFoundError):
    pass


class DatasetNotFoundError(NotFoundError):
    pass


class DatasetVersionNotFoundError(NotFoundError):
    pass


class DatasetInvalidVersionError(Exception):
    pass


class PendingIndexingError(Exception):
    """An indexing operation is already in progress."""


class QueryScriptCompileError(Exception):
    pass


class QueryScriptRunError(Exception):
    """Error raised by `subprocess.run`.

    Attributes:
        message      Explanation of the error
        return_code  Code returned by the subprocess
    """

    def __init__(self, message: str, return_code: int = 0):
        self.message = message
        self.return_code = return_code
        super().__init__(message)


class QueryScriptCancelError(QueryScriptRunError):
    pass


class ClientError(RuntimeError):
    def __init__(self, message, error_code=None):
        super().__init__(message)
        # error code from the cloud itself
        self.error_code = error_code


class TableMissingError(DataChainError):
    pass


class OutdatedDatabaseSchemaError(DataChainError):
    pass


class CheckpointNotFoundError(NotFoundError):
    pass


class JobNotFoundError(NotFoundError):
    pass


class JobAncestryDepthExceededError(DataChainError):
    pass
