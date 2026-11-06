# Safely fix Q3 imports from the project root.
# Creates .bak backups before modifying files. Does not start training.

from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent
TARGETS = [
    PROJECT_ROOT / "src" / "q3" / "run_experiments.py",
    PROJECT_ROOT / "src" / "q3" / "q3_utils.py",
    PROJECT_ROOT / "src" / "q3" / "train_yolo.py",
]

OLD = 'sys.path.insert(0, str(Path(__file__).resolve().parent))'
NEW = (
    "# Make both src/ and src/q3 importable.\n"
    "_SRC_DIR = Path(__file__).resolve().parent.parent\n"
    "_Q3_DIR = Path(__file__).resolve().parent\n"
    "for _directory in (str(_SRC_DIR), str(_Q3_DIR)):\n"
    "    if _directory not in sys.path:\n"
    "        sys.path.insert(0, _directory)"
)

def main():
    for path in TARGETS:
        if not path.exists():
            raise FileNotFoundError(f'Required file not found: {path}')
        text = path.read_text(encoding='utf-8')
        if NEW in text:
            print(f'Already patched: {path.relative_to(PROJECT_ROOT)}')
            continue
        if OLD not in text:
            raise RuntimeError(f'Expected import line not found in {path.name}. No change made to this file.')
        backup = path.with_suffix(path.suffix + '.bak')
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(text.replace(OLD, NEW, 1), encoding='utf-8')
        print(f'Patched: {path.relative_to(PROJECT_ROOT)}')
        print(f'Backup: {backup.relative_to(PROJECT_ROOT)}')
    print('\nImport-path patch completed. No training was started.')

if __name__ == '__main__':
    main()
