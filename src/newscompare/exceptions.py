"""Custom exceptions."""


class FeedFetchError(Exception):
    """Failed to fetch or parse a feed."""


class ExtractionError(Exception):
    """Failed to extract article body from URL."""
