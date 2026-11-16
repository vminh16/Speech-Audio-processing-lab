"""Run the full Lab 5 pipeline, figures and 6-page report from the project root."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.lab import run_full_lab

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    run_full_lab(root)
    from tools.build_report import build_report
    build_report(root)
    print('Complete: results/lab_summary.json, all WAV/figures, report.pdf and report.md.')
