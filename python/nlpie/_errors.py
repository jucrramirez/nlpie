"""Package-specific exception hierarchy for nlpie."""


class NlpieError(Exception):
    """Base exception for all errors in the nlpie package."""


class PreprocessingError(NlpieError, ValueError):
    """Exception raised when an embedding preprocessing or normalization operation fails."""
