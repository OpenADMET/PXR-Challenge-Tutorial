from __future__ import annotations

import zipfile
import tempfile
import MDAnalysis as mda
from pathlib import Path
from typing import Mapping, Union

from rdkit import Chem
from rdkit.Chem import AllChem

STRUCTURE_DATASET_SIZE = 110


def _canonical_no_stereo_smiles(mol: Chem.Mol) -> str:
    """Return a canonical, non-isomeric SMILES string for robust matching."""
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=False)


def _validate_ligand_bond_orders(
    pdb_path: str,
    expected_smiles: str,
) -> tuple[bool, str | None]:
    """
    Validate ligand chemistry by assigning bond orders from a SMILES template.

    This mirrors the template-based approach used during prep/inspection workflows,
    but is embedded here so validation does not rely on external helper scripts.
    """
    # Parse ligand-only PDB generated from a single LIG residue
    lig_mol = Chem.MolFromPDBFile(pdb_path, removeHs=True)
    if lig_mol is None:
        return False, "RDKit could not parse extracted LIG residue"

    template = Chem.MolFromSmiles(expected_smiles)
    if template is None:
        return False, f"Invalid expected SMILES provided: {expected_smiles!r}"

    try:
        assigned = AllChem.AssignBondOrdersFromTemplate(template, lig_mol)
    except Exception as exc:
        return False, f"Bond-order assignment from template failed: {exc}"

    expected_canon = _canonical_no_stereo_smiles(template)
    assigned_canon = _canonical_no_stereo_smiles(assigned)
    if expected_canon != assigned_canon:
        return (
            False,
            f"Ligand chemistry mismatch after bond-order assignment: expected {expected_canon}, got {assigned_canon}",
        )

    return True, None


def _smiles_for_structure(
    expected_ligand_smiles: Mapping[str, str] | set[str] | list[str] | None,
    structure_id: str,
) -> str | None:
    """
    Resolve expected SMILES for a given structure ID.

    Supported formats:
    - Mapping[str, str]: preferred; keys are structure IDs.
    - set/list[str]: fallback; accepted SMILES pool (ID-agnostic).
    """
    if expected_ligand_smiles is None:
        return None

    if isinstance(expected_ligand_smiles, Mapping):
        return expected_ligand_smiles.get(structure_id)

    # Fallback mode: if a non-mapping iterable of SMILES is provided,
    # we cannot resolve ID-specific templates here.
    return None

def validate_structure_submission(
    structure_predictions_file: Union[str, Path],
    expected_ids: set[str] | None = None,
    expected_ligand_smiles: Mapping[str, str] | set[str] | list[str] | None = None,
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
                        
                        structure_id = Path(name).stem

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

                            # 4. Optional ligand bond-order/chemistry check against expected SMILES
                            expected_smiles = _smiles_for_structure(expected_ligand_smiles, structure_id)
                            if expected_smiles is not None:
                                lig_res = ligands.residues[0]
                                lig_atoms = lig_res.atoms

                                # Write ligand-only PDB for robust template assignment
                                lig_tmp_path = Path(tmpdir) / f"{structure_id}_LIG.pdb"
                                lig_atoms.write(str(lig_tmp_path))

                                is_ok, msg = _validate_ligand_bond_orders(str(lig_tmp_path), expected_smiles)
                                if not is_ok:
                                    errors.append(f"{name}: {msg}")
                            elif isinstance(expected_ligand_smiles, (set, list)):
                                # Backward-compatible pool mode: validate assigned chemistry against any provided SMILES.
                                lig_res = ligands.residues[0]
                                lig_atoms = lig_res.atoms
                                lig_tmp_path = Path(tmpdir) / f"{structure_id}_LIG.pdb"
                                lig_atoms.write(str(lig_tmp_path))

                                lig_mol = Chem.MolFromPDBFile(str(lig_tmp_path), removeHs=True)
                                if lig_mol is None:
                                    errors.append(f"{name}: RDKit could not parse extracted LIG residue")
                                else:
                                    matched = False
                                    for smiles in expected_ligand_smiles:
                                        template = Chem.MolFromSmiles(smiles)
                                        if template is None:
                                            continue
                                        try:
                                            assigned = AllChem.AssignBondOrdersFromTemplate(template, lig_mol)
                                        except Exception:
                                            continue
                                        if _canonical_no_stereo_smiles(assigned) == _canonical_no_stereo_smiles(template):
                                            matched = True
                                            break
                                    if not matched:
                                        errors.append(
                                            f"{name}: Ligand did not match any expected SMILES after bond-order assignment"
                                        )

                        except Exception as e:
                            errors.append(f"{name}: MDAnalysis failed to parse file: {e}")


    except zipfile.BadZipFile:
        return False, ["File is not a valid zip archive."]
    except Exception as exc:
        return False, [f"Unexpected error during validation: {exc}"]

    return len(errors) == 0, errors