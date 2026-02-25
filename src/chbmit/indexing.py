"""
Dataset indexing utilities for CHB-MIT database.
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import json

from .parse_summary import get_seizure_times, count_seizures


@dataclass
class SubjectInfo:
    """Information about a single subject."""
    subject_id: str
    edf_files: List[str]
    total_seizures: int
    files_with_seizures: List[str]
    seizures_by_file: Dict[str, List[tuple]]


@dataclass  
class DatasetIndex:
    """Complete index of the CHB-MIT dataset."""
    subjects: List[str]
    subject_info: Dict[str, SubjectInfo]
    total_edf_files: int
    total_seizures: int


def get_subject_list(data_root: str | Path) -> List[str]:
    """
    Get list of all subject folders in the dataset.
    
    Args:
        data_root: Root path to CHB-MIT data
        
    Returns:
        Sorted list of subject IDs (e.g., ['chb01', 'chb02', ...])
    """
    data_root = Path(data_root)
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    
    subjects = []
    for item in data_root.iterdir():
        if item.is_dir() and item.name.startswith("chb"):
            subjects.append(item.name)
    
    return sorted(subjects)


def get_edf_files(data_root: str | Path, subject: str) -> List[str]:
    """
    Get list of EDF files for a subject.
    
    Args:
        data_root: Root path to CHB-MIT data
        subject: Subject ID (e.g., 'chb01')
        
    Returns:
        Sorted list of EDF filenames
    """
    subject_dir = Path(data_root) / subject
    
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
    
    edf_files = []
    for f in subject_dir.iterdir():
        if f.suffix.lower() == ".edf":
            edf_files.append(f.name)
    
    return sorted(edf_files)


def get_subject_info(data_root: str | Path, subject: str) -> SubjectInfo:
    """
    Get detailed information about a subject.
    
    Args:
        data_root: Root path to CHB-MIT data
        subject: Subject ID
        
    Returns:
        SubjectInfo dataclass
    """
    data_root = Path(data_root)
    
    edf_files = get_edf_files(data_root, subject)
    
    try:
        seizures_by_file = get_seizure_times(data_root, subject)
    except FileNotFoundError:
        seizures_by_file = {}
    
    # Convert to serializable format
    seizures_dict = {
        k: [(s, e) for s, e in v] 
        for k, v in seizures_by_file.items()
    }
    
    files_with_seizures = [f for f, s in seizures_by_file.items() if len(s) > 0]
    total_seizures = sum(len(s) for s in seizures_by_file.values())
    
    return SubjectInfo(
        subject_id=subject,
        edf_files=edf_files,
        total_seizures=total_seizures,
        files_with_seizures=files_with_seizures,
        seizures_by_file=seizures_dict,
    )


def build_dataset_index(data_root: str | Path) -> DatasetIndex:
    """
    Build complete index of the CHB-MIT dataset.
    
    Args:
        data_root: Root path to CHB-MIT data
        
    Returns:
        DatasetIndex dataclass
    """
    data_root = Path(data_root)
    subjects = get_subject_list(data_root)
    
    subject_info = {}
    total_edf = 0
    total_seizures = 0
    
    for subject in subjects:
        info = get_subject_info(data_root, subject)
        subject_info[subject] = info
        total_edf += len(info.edf_files)
        total_seizures += info.total_seizures
    
    return DatasetIndex(
        subjects=subjects,
        subject_info=subject_info,
        total_edf_files=total_edf,
        total_seizures=total_seizures,
    )


def save_index(index: DatasetIndex, path: str | Path) -> None:
    """Save dataset index to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable dict
    data = {
        "subjects": index.subjects,
        "total_edf_files": index.total_edf_files,
        "total_seizures": index.total_seizures,
        "subject_info": {
            k: {
                "subject_id": v.subject_id,
                "edf_files": v.edf_files,
                "total_seizures": v.total_seizures,
                "files_with_seizures": v.files_with_seizures,
                "seizures_by_file": v.seizures_by_file,
            }
            for k, v in index.subject_info.items()
        }
    }
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_index(path: str | Path) -> DatasetIndex:
    """Load dataset index from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    
    subject_info = {}
    for k, v in data["subject_info"].items():
        subject_info[k] = SubjectInfo(
            subject_id=v["subject_id"],
            edf_files=v["edf_files"],
            total_seizures=v["total_seizures"],
            files_with_seizures=v["files_with_seizures"],
            seizures_by_file=v["seizures_by_file"],
        )
    
    return DatasetIndex(
        subjects=data["subjects"],
        subject_info=subject_info,
        total_edf_files=data["total_edf_files"],
        total_seizures=data["total_seizures"],
    )
