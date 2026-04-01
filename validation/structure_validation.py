from __future__ import annotations

import zipfile
import tempfile
import MDAnalysis as mda
from pathlib import Path
from typing import Union
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

STRUCTURE_DATASET_SIZE = 78


def validate_structure_submission(
    structure_predictions_file: Union[str, Path],
    expected_ids: set[str] | None = None,
    expected_ligand_smiles: dict[str, str] | None = None,
    require_lig_resname: bool = True,
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

            # --- ID Consistency Checks ---
            submitted_ids = {Path(name).stem for name in pdb_files}
            if expected_ids is not None:
                expected_ids = {str(x) for x in expected_ids}
                missing = sorted(expected_ids - submitted_ids)
                if missing:
                    errors.append(f"Missing {len(missing)} expected structure(s): {missing[:20]}")
            elif len(pdb_files) != STRUCTURE_DATASET_SIZE:
                errors.append(
                    f"Zip file contains {len(pdb_files)} .pdb files, expected {STRUCTURE_DATASET_SIZE}."
                )

            # --- MDAnalysis Structural Checks ---
            if require_lig_resname:
                with tempfile.TemporaryDirectory() as tmpdir:
                    for name in pdb_files:
                        # Extract to temp file so MDAnalysis can read it
                        tmp_path = zip_file.extract(name, path=tmpdir)
                        
                        try:
                            # Suppress warnings for missing chain IDs/elements if necessary
                            u = mda.Universe(tmp_path)
                            
                            # 1. Check for residue name 'LIG'
                            ligands = u.select_atoms("resname LIG")
                            if len(ligands) == 0:
                                errors.append(f"{name}: Missing residue 'LIG'")
                                continue

                            # 2. Ensure only ONE residue named LIG exists
                            if len(ligands.residues) > 1:
                                errors.append(f"{name}: Found {len(ligands.residues)} 'LIG' residues, expected 1")

                            # 3. Ensure max two chain exists in the entire PDB
                            if len(u.segments) > 2:
                                errors.append(f"{name}: Found {len(u.segments)} chains, expected 2 or fewer")

                            # 4. Check ligand graph matches expected SMILES connectivity
                            if expected_ligand_smiles is not None:
                                pdb_id = Path(name).stem
                                expected_smi = expected_ligand_smiles.get(pdb_id)
                                if expected_smi is not None:
                                    ref_mol = Chem.MolFromSmiles(expected_smi)
                                    if ref_mol is None:
                                        errors.append(
                                            f"{name}: Could not parse expected SMILES for '{pdb_id}'"
                                        )
                                    else:
                                        lig_pdb_path = Path(tmpdir) / f"{pdb_id}_lig.pdb"
                                        ligands.write(str(lig_pdb_path))
                                        lig_mol = Chem.MolFromPDBFile(
                                            str(lig_pdb_path), removeHs=True, sanitize=False
                                        )
                                        if lig_mol is None:
                                            errors.append(
                                                f"{name}: RDKit could not parse LIG residue"
                                            )
                                        else:
                                            try:
                                                RDLogger.DisableLog("rdApp.*")
                                                AllChem.AssignBondOrdersFromTemplate(ref_mol, lig_mol)
                                                RDLogger.EnableLog("rdApp.*")
                                            except ValueError:
                                                RDLogger.EnableLog("rdApp.*")
                                                errors.append(
                                                    f"{name}: Ligand connectivity does not match "
                                                    f"expected SMILES '{expected_smi}'"
                                                )

                        except Exception as e:
                            errors.append(f"{name}: MDAnalysis failed to parse file: {e}")


    except zipfile.BadZipFile:
        return False, ["File is not a valid zip archive."]
    except Exception as exc:
        return False, [f"Unexpected error during validation: {exc}"]

    return len(errors) == 0, errors