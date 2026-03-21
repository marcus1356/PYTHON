class SubmissionNotFoundError(Exception):
    def __init__(self, submission_id: str) -> None:
        self.submission_id = submission_id
        super().__init__(f"Submission '{submission_id}' not found.")


class SubmissionAlreadyAnalyzingError(Exception):
    def __init__(self, submission_id: str) -> None:
        self.submission_id = submission_id
        super().__init__(f"Submission '{submission_id}' is already being analyzed.")


class SubmissionAlreadyAnalyzedError(Exception):
    def __init__(self, submission_id: str) -> None:
        self.submission_id = submission_id
        super().__init__(
            f"Submission '{submission_id}' was already analyzed. "
            "Use force_reanalyze=true to re-run."
        )


class SubmissionImmutableFieldError(Exception):
    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(f"Field '{field}' is immutable after creation.")


class AnalysisResultNotFoundError(Exception):
    def __init__(self, submission_id: str) -> None:
        self.submission_id = submission_id
        super().__init__(
            f"No analysis result for submission '{submission_id}'. "
            "Submit to /analyze first."
        )
