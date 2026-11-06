"""Grading identical to evaluation/cif_structure_validator_mp52.py (functions copied verbatim)."""
import signal, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pymatgen.core import Structure, Element
from pymatgen.analysis.structure_matcher import StructureMatcher


def is_element_valid(composition):
    try:
        for element in composition.elements:
            Element(element.symbol)
        return True
    except:
        return False

def get_safe_space_group_number(structure):
    try:
        spg_info = structure.get_space_group_info()
        return spg_info[1] if spg_info else np.nan
    except Exception:
        return np.nan

def structure_validity(structure, cutoff=0.5):
    dist_mat = structure.distance_matrix
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or structure.volume < 0.1:
        return False
    spg_number = get_safe_space_group_number(structure)
    return not pd.isna(spg_number)

def validate_structure(cif_string):
    try:
        structure = Structure.from_str(cif_string, fmt="cif")
        valid_element = is_element_valid(structure.composition)
        valid_struct = structure_validity(structure)
        comp = structure.composition.reduced_formula if valid_element else np.nan
        spg = get_safe_space_group_number(structure) if valid_element and valid_struct else np.nan
        if valid_element and valid_struct:
            return True, structure, comp, spg
        else:
            return False, None, comp, spg
    except Exception:
        return False, None, np.nan, np.nan

def compare_structures(struct1, struct2, stol=0.5, ltol=0.3, angle_tol=10):
    matcher = StructureMatcher(stol=stol, ltol=ltol, angle_tol=angle_tol)
    try:
        rms_result = matcher.get_rms_dist(struct1, struct2)
        if rms_result is not None:
            return rms_result[0]
        return np.nan
    except:
        return np.nan

def near_miss_rms(struct1, struct2, stol=1.5, ltol=1.0, angle_tol=25):
    try:
        matcher = StructureMatcher(stol=stol, ltol=ltol, angle_tol=angle_tol,
                                   primitive_cell=False)
        rms_result = matcher.get_rms_dist(struct1, struct2)
        if rms_result is not None:
            return rms_result[0]
        return np.nan
    except:
        return np.nan


class _Timeout(Exception):
    pass

def _alarm(*_):
    raise _Timeout()

def _f(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else x

def grade_one(args):
    """(generated text, reference CIF) -> grade dict. Same decision rule as the evaluation's validator."""
    text, ref = args
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(180)
    g = dict(match=0, valid=0, parseable=0, comp_ok=0, spg_ok=0, rms=None, near_miss_rms=None,
             gen_formula=None, gen_spg=None, category="not_parseable")
    try:
        valid_gt, s_gt, comp_gt, spg_gt = validate_structure(ref)
        valid, s, comp, spg = validate_structure(text if isinstance(text, str) else "")
        g["parseable"] = int(_f(comp) is not None)
        g["valid"] = int(bool(valid)); g["gen_formula"] = _f(comp)
        g["gen_spg"] = None if _f(spg) is None else int(spg)
        g["comp_ok"] = int(_f(comp) is not None and comp == comp_gt)
        g["spg_ok"] = int(_f(spg) is not None and _f(spg_gt) is not None and int(spg) == int(spg_gt))
        if valid_gt and valid:
            rms = compare_structures(s, s_gt)
            m = 1 if isinstance(rms, float) and not np.isnan(rms) else 0
            g["match"] = m
            g["rms"] = float(rms) if m else None
            nm = rms if m else near_miss_rms(s, s_gt)
            g["near_miss_rms"] = None if _f(float(nm) if nm is not None else None) is None else float(nm)
        g["category"] = ("match" if g["match"] else
                         ("valid_no_match" if g["comp_ok"] else "valid_wrong_composition") if g["valid"] else
                         "invalid_structure" if g["parseable"] else "not_parseable")
    except _Timeout:
        g["category"] = "grader_timeout"
    finally:
        signal.alarm(0)
    return g
