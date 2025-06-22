"""
Custom warnings and exceptions for justuse.
"""


class JustuseIssue(Exception):
    pass


class NirvanaWarning(Warning, JustuseIssue):
    pass


class VersionWarning(Warning, JustuseIssue):
    pass


class NotReloadableWarning(Warning, JustuseIssue):
    pass


class NoValidationWarning(Warning, JustuseIssue):
    pass


class AmbiguityWarning(Warning, JustuseIssue):
    pass


class UnexpectedHash(ImportError, JustuseIssue):
    pass


class InstallationError(ImportError, JustuseIssue):
    pass
