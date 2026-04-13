import logging
import os
from abc import ABCMeta, abstractmethod

logger = logging.getLogger(__name__)


class PrivSQLGrammar:
    def __init__(self):
        file_path = os.path.join(os.path.dirname(__file__), "sqlgrammar.lark")
        self._grammar = open(file_path).read()
        logger.debug(
            "Loaded grammar file: %s (length=%s)", file_path, len(self._grammar)
        )

    @property
    def grammar(self):
        logger.debug("Access grammar property")
        return self._grammar


class PrivacyChecker(metaclass=ABCMeta):
    @abstractmethod
    def privacy_check(
        self, privacy_unit_columns: dict[str | None, dict[str, str]]
    ) -> bool:
        """Check if the privacy unit columns satisfy the privacy constraints.
        Args:
            privacy_unit_columns (dict[str | None, dict[str, str]]): The privacy unit
              columns of the database.
        Returns:
            bool: True if the privacy unit columns satisfy the privacy constraints,
              False otherwise.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def privacy_unit_column(
        self, privacy_unit_columns: dict[str | None, dict[str, str]]
    ) -> str | None:
        """Return the privacy unit column for the table.
        If the table is not a private table, return None.
        Args:
            privacy_unit_columns (dict[str | None, dict[str, str]]): The privacy unit
              columns of the database.
        Returns:
            str | None: The privacy unit column for the table, or None if the table
              is not a private table.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def is_private_table(
        self, privacy_unit_columns: dict[str | None, dict[str, str]]
    ) -> bool:
        """Check if the table is a private table.
        Args:
            privacy_unit_columns (dict[str | None, dict[str, str]]): The privacy unit
              columns of the database.
        Returns:
            bool: True if the table is a private table, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement this method")
