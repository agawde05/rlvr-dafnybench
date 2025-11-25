# dafny_file.py
#
# This module provides utilities for working with and testing a single Dafny
# file.

from pathlib import Path
from typing import Optional
import uuid
import os
import subprocess
from tempfile import NamedTemporaryFile


def _is_subsequence(a, b):
    """Return True if list a is a subsequence of list b (order preserved)."""
    it = iter(b)
    return all(x in it for x in a)


class DafnyFile:
    """Represents a Dafny file, as either source code or a file path."""

    file_name: Optional[Path]
    code: Optional[str]

    def __init__(self, file_name: Optional[Path] = None, code: Optional[str] = None):
        if file_name is None and code is None:
            raise ValueError("Either file_name or code must be provided.")
        self.file_name = file_name
        self.code = code

    @staticmethod
    def from_file(file_name: Path) -> "DafnyFile":
        """Creates a DafnyFile instance from a file."""
        return DafnyFile(file_name=file_name)

    @staticmethod
    def from_code(code: str) -> "DafnyFile":
        """Creates a DafnyFile instance from code."""
        return DafnyFile(code=code)

    def populate_code_from_file(self):
        """Populates the code attribute by reading from the file_name."""
        if self.file_name is None:
            raise ValueError("file_name is not set.")
        with open(self.file_name, "r") as f:
            self.code = f.read()

    def get_code(self) -> Optional[str]:
        """Returns the Dafny code, reading from file if necessary."""
        if self.code is None and self.file_name is not None:
            self.populate_code_from_file()
        return self.code

    def set_code(self, code: str):
        """Sets the Dafny code directly."""
        self.code = code

    @staticmethod
    def validate_no_assume(original: str, modified: str) -> bool:
        """Returns True if no new 'assume' statements were added in the modified code."""
        # find all lines with 'assume' in them, and make sure the lines are
        # identical
        original_lines = [
            line.strip() for line in original.splitlines() if "assume" in line
        ]
        modified_lines = [
            line.strip() for line in modified.splitlines() if "assume" in line
        ]
        return original_lines == modified_lines

    @staticmethod
    def validate_no_deletion(original: str, modified: str) -> bool:
        """Validates that the modified code does not delete any lines from the
        original code."""
        original_lines = [line.strip() for line in original.splitlines()]
        modified_lines = [line.strip() for line in modified.splitlines()]
        return _is_subsequence(original_lines, modified_lines)


class Dafny:
    dafny_path: Path

    def __init__(self, dafny_path: Path):
        self.dafny_path = dafny_path

    def verify(self, dafny_file: DafnyFile) -> bool:
        """Verifies the Dafny File using the Dafny verifier.

        TODO: determine if we should return Dafny output as well as a bool.

        Returns True if verification succeeds, False otherwise.
        """
        with NamedTemporaryFile(mode="w+", suffix=".dfy") as f:
            code = dafny_file.get_code()
            if code is None:
                raise ValueError("DafnyFile has no code to write.")
            f.write(code)
            f.flush()

            # run the Dafny verifier
            proc = subprocess.run(
                [str(self.dafny_path), "verify", str(f.name)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

        print(proc.stdout)
        return proc.returncode == 0
