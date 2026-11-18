"""Download the trained weights from Hugging Face into weights/.

Usage:
    python download_weights.py                 # all 4 folds (recommended)
    python download_weights.py --folds 0       # one fold only: smaller and faster, slightly less accurate

If the Hugging Face repository is private, log in once first with `hf auth login`
(a token with read access), or set the HF_TOKEN environment variable.
"""
from pathlib import Path
import argparse

from huggingface_hub import snapshot_download

REPO = Path(__file__).resolve().parent
HF_REPO = 'adriana-neves02/nnUNet-OCTA-AV-segmentation'  # <user or organization>/<model repository> on Hugging Face
MODEL = 'Dataset501_OCTA_all/nnUNetTrainer__nnUNetResEncUNetLPlans__2d'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', default=HF_REPO, help='Hugging Face model repository')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--revision', default=None, help='Branch, tag or commit of the weights (default: main)')
    parser.add_argument('-o', '--output', default=str(REPO / 'weights'))
    args = parser.parse_args()

    patterns = [f'{MODEL}/*.json'] + [f'{MODEL}/fold_{k}/checkpoint_final.pth' for k in args.folds]
    path = snapshot_download(repo_id=args.repo, revision=args.revision, local_dir=args.output,
                             allow_patterns=patterns)
    model = Path(path) / MODEL
    found = sorted(p.parent.name for p in model.glob('fold_*/checkpoint_final.pth'))
    missing = [f'fold_{k}' for k in args.folds if f'fold_{k}' not in found]
    print(f'Model in {model}: {", ".join(found) or "no folds found"}')
    if missing:
        raise SystemExit(f'Missing on Hugging Face: {", ".join(missing)}. Expected the layout {MODEL}/fold_<k>/')


if __name__ == '__main__':
    main()
