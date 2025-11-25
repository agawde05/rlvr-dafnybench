# dafny_file.py
#
# This module provides utilities for working with and testing a single Dafny
# file.

from pathlib import Path
from typing import Optional
import uuid
import os
import subprocess


class DafnyFile:
    """Represents a Dafny file, as either source code or a file path."""

    file_name: Optional[Path]
    code: Optional[str]

    def __init__(self, file_name: Optional[Path] = None, code: Optional[str] = None):
        if file_name is None and code is None:
            raise ValueError("Either file_name or code must be provided.")
        self.file_name = file_name
        self.code = code

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
        """Makes sure that the new code does not introduce any 'assume'
            statements that were not in the original code."""
        # find all lines with 'assume' in them, and make sure the lines are
        # identical
        original_lines = [
            line.strip() for line in original.splitlines() if 'assume' in line
        ]
        modified_lines = [
            line.strip() for line in modified.splitlines() if 'assume' in line
        ]
        return original_lines == modified_lines
    


class Dafny:
    dafny_path: Path
    uuid: uuid.UUID
    folder_created: bool = False

    def __init__(self, dafny_path: Path):
        self.dafny_path = dafny_path
        self.uuid = uuid.uuid4()
    
    def __enter__(self):
        # create temp folder
        temp_folder = Path(self._get_folder())
        temp_folder.mkdir(parents=True)
        self.folder_created = True
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        # remove temp folder
        if self.folder_created:
            temp_folder = Path(self._get_folder())
            for root, dirs, files in os.walk(temp_folder, topdown=False):
                for name in files:
                    os.remove(Path(root) / name)
                for name in dirs:
                    os.rmdir(Path(root) / name)
            os.rmdir(temp_folder)
        
    
    def _get_folder(self) -> Path:
        return Path(f"tmp/dafny_{self.uuid}")

    @staticmethod
    def _get_random_filename() -> str:
        return f"temp_{uuid.uuid4().hex}"
    

    def verify(self, dafny_file: DafnyFile) -> bool:
        """Verifies the Dafny File using the Dafny verifier.

        TODO: determine if we should return Dafny output as well as a bool.

        Returns True if verification succeeds, False otherwise.
        """
        if not self.folder_created:
            raise RuntimeError("Dafny context not entered. Use 'with Dafny(...) as d:' syntax.")
        
        # write the Dafny file to the temp folder
        # generate a random file name
        temp_folder = Path(self._get_folder())
        random_file_name = self._get_random_filename() + ".dfy"
        temp_file_path = temp_folder / random_file_name
        with open(temp_file_path, "w") as f:
            code = dafny_file.get_code()
            if code is None:
                raise ValueError("DafnyFile has no code to write.")
            f.write(code)
        
        # run the Dafny verifier
        proc = subprocess.run(
            [str(self.dafny_path), "verify", str(temp_file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # delete the temp file
        os.remove(temp_file_path)
        
        print(proc.stdout)
        return proc.returncode == 0

