"""
Parser for CHB-MIT summary files (chbXX-summary.txt).

Each subject folder contains a summary file with seizure annotations.
Format example:

File Name: chb01_03.edf
File Start Time: 13:43:04
File End Time: 14:43:04
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SeizureInfo:
    """Information about a single seizure event."""
    file_name: str
    start_sec: float
    end_sec: float
    
    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class FileInfo:
    """Information about an EDF file."""
    file_name: str
    start_time: Optional[str]
    end_time: Optional[str]
    num_seizures: int
    seizures: List[Tuple[float, float]]  # (start_sec, end_sec)


def parse_summary_file(summary_path: str | Path) -> Dict[str, List[Tuple[float, float]]]:
    """
    Parse a CHB-MIT summary file to extract seizure times.
    
    Args:
        summary_path: Path to the summary file (e.g., chb01-summary.txt)
        
    Returns:
        Dict mapping EDF filename to list of (start_sec, end_sec) tuples
    """
    summary_path = Path(summary_path)
    
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Result dict: filename -> list of seizure intervals
    seizures_by_file: Dict[str, List[Tuple[float, float]]] = {}
    
    # Split content into file blocks
    # Each block starts with "File Name:"
    blocks = re.split(r"(?=File Name:)", content)
    
    for block in blocks:
        if not block.strip():
            continue
            
        # Extract file name
        file_match = re.search(r"File Name:\s*(\S+\.edf)", block, re.IGNORECASE)
        if not file_match:
            continue
            
        file_name = file_match.group(1)
        
        # Extract number of seizures
        num_seizures_match = re.search(
            r"Number of Seizures in File:\s*(\d+)", block, re.IGNORECASE
        )
        num_seizures = int(num_seizures_match.group(1)) if num_seizures_match else 0
        
        seizures = []
        
        if num_seizures > 0:
            # Find all seizure start/end times
            # Pattern handles various formats:
            # "Seizure Start Time: 2996 seconds"
            # "Seizure 1 Start Time: 2996 seconds"
            start_pattern = r"Seizure\s*\d*\s*Start\s*Time:\s*(\d+)\s*seconds?"
            end_pattern = r"Seizure\s*\d*\s*End\s*Time:\s*(\d+)\s*seconds?"
            
            starts = re.findall(start_pattern, block, re.IGNORECASE)
            ends = re.findall(end_pattern, block, re.IGNORECASE)
            
            # Pair them up
            for start, end in zip(starts, ends):
                seizures.append((float(start), float(end)))
        
        seizures_by_file[file_name] = seizures
    
    return seizures_by_file


def get_seizure_times(
    data_root: str | Path,
    subject: str,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Get seizure times for a subject from their summary file.
    
    Args:
        data_root: Root path to CHB-MIT data
        subject: Subject ID (e.g., 'chb01')
        
    Returns:
        Dict mapping EDF filename to list of (start_sec, end_sec) tuples
    """
    data_root = Path(data_root)
    
    # Try different summary file naming patterns
    summary_patterns = [
        f"{subject}-summary.txt",
        f"{subject}_summary.txt",
    ]
    
    summary_path = None
    for pattern in summary_patterns:
        candidate = data_root / subject / pattern
        if candidate.exists():
            summary_path = candidate
            break
    
    if summary_path is None:
        # Try to find any file with "summary" in name
        subject_dir = data_root / subject
        if subject_dir.exists():
            for f in subject_dir.iterdir():
                if "summary" in f.name.lower() and f.suffix == ".txt":
                    summary_path = f
                    break
    
    if summary_path is None:
        raise FileNotFoundError(
            f"No summary file found for subject {subject} in {data_root / subject}"
        )
    
    return parse_summary_file(summary_path)


def get_all_seizure_info(data_root: str | Path, subject: str) -> List[SeizureInfo]:
    """
    Get all seizures for a subject as SeizureInfo objects.
    
    Args:
        data_root: Root path to CHB-MIT data
        subject: Subject ID (e.g., 'chb01')
        
    Returns:
        List of SeizureInfo objects
    """
    seizures_by_file = get_seizure_times(data_root, subject)
    
    all_seizures = []
    for file_name, seizures in seizures_by_file.items():
        for start, end in seizures:
            all_seizures.append(SeizureInfo(
                file_name=file_name,
                start_sec=start,
                end_sec=end,
            ))
    
    return all_seizures


def count_seizures(data_root: str | Path, subject: str) -> int:
    """Count total seizures for a subject."""
    seizures_by_file = get_seizure_times(data_root, subject)
    return sum(len(s) for s in seizures_by_file.values())


def get_files_with_seizures(
    data_root: str | Path,
    subject: str,
) -> List[str]:
    """Get list of EDF files that contain seizures."""
    seizures_by_file = get_seizure_times(data_root, subject)
    return [f for f, s in seizures_by_file.items() if len(s) > 0]
