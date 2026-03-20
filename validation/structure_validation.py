from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Union

STRUCTURE_DATASET_SIZE = 110

def validate_structure_submission(
    structure_predictions_file: Union[str, Path],
    expected_ids: set[str] | None = None,
    require_lig_resname: bool = False,
) -> tuple[bool, list[str]]:

    errors: list[str] = []
    path = Path(structure_predictions_file)

    if not path.exists():
        return False, [f"File does not exist: {path}"]

    if path.suffix.lower() != ".zip":
        return False, ["Structure predictions file must be a .zip file."]

    try:
        with zipfile.ZipFile(path, "r") as zip_file:
            pdb_files = [name for name in zip_file.namelist() if name.lower().endswith(".pdb")]

            if not pdb_files:
                return False, ["Zip file contains no PDB files."]

            if expected_ids is not None:
                submitted_ids = {Path(name).stem for name in pdb_files}
                expected_ids = {str(x) for x in expected_ids}
                missing = sorted(expected_ids - submitted_ids)
                extra = sorted(submitted_ids - expected_ids)

                if missing:
                    errors.append(f"Missing {len(missing)} expected structure(s): {missing[:20]}")
                if extra:
                    errors.append(f"Found {len(extra)} unexpected structure(s): {extra[:20]}")
            elif len(pdb_files) != STRUCTURE_DATASET_SIZE:
                errors.append(
                    f"Zip file contains {len(pdb_files)} .pdb files, expected {STRUCTURE_DATASET_SIZE}."
                )

            if require_lig_resname:
                missing_lig = []
                for name in pdb_files:
                    content = zip_file.read(name).decode("utf-8", errors="ignore")
                    if " LIG " not in content:
                        missing_lig.append(Path(name).name)

                if missing_lig:
                    errors.append(
                        f"Found {len(missing_lig)} structure(s) without residue name 'LIG': {missing_lig[:20]}"
                    )

    except zipfile.BadZipFile:
        return False, ["File is not a valid zip archive."]
    except Exception as exc:
        return False, [f"Unexpected error during validation: {exc}"]

    return len(errors) == 0, errors
