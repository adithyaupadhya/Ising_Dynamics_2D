"""Execute tutorials in fresh kernels without changing source notebooks."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    for path in sorted(folder.glob('[0-9]*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(notebook, timeout=240, kernel_name='python3',
                       resources={'metadata': {'path': str(folder)}}).execute()
        print(f'PASS {path.name}', flush=True)
