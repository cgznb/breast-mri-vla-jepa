"""Check tracked publication contents, relative doc links and packaged configs.

Run from any directory after staging files. This check uses only the standard
library and never prints secret contents. Runtime patient data and outputs are
not part of the source repository.
"""
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRS = {
    "data", "runs", "artifacts", "experiments", "checkpoints", "results",
    "weights", "reports", "logs", ".codex", ".venv", ".cache",
}
PRIVATE_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".npy", ".npz", ".nii", ".gz", ".dcm",
    ".xlsx", ".pem", ".key", ".zip", ".tar", ".pyc",
}
SECRET = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|"
    r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----)"
)
LOCAL_DOC_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^\s)]+)(?:\s+[^)]*)?\)")


def tracked_files():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "-z"], cwd=ROOT,
        capture_output=True, check=True,
    )
    return [Path(p) for p in result.stdout.decode().split("\0") if p]


def main():
    errors = []
    files = tracked_files()
    if not files:
        errors.append("No tracked files; stage the publication before checking")
    for relative in files:
        path = ROOT / relative
        if any(part in PRIVATE_DIRS for part in relative.parts):
            errors.append(f"Private runtime directory tracked: {relative}")
        if path.suffix.lower() in PRIVATE_SUFFIXES or path.name.startswith(".env"):
            errors.append(f"Runtime or credential file tracked: {relative}")
        if path.is_symlink():
            errors.append(f"Symlink requires explicit review: {relative}")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            errors.append(f"Expected text source file: {relative}")
            continue
        if SECRET.search(content):
            errors.append(f"Credential pattern in {relative}; contents omitted")
        if any(marker in content for marker in ("/root/" + "autodl-tmp", "seetacloud" + ".com")):
            errors.append(f"Machine-specific deployment reference: {relative}")
        if path.suffix == ".md":
            for link in LOCAL_DOC_LINK.findall(content):
                if "://" in link or link.startswith(("#", "mailto:")):
                    continue
                target = link.split("#", 1)[0]
                if target and not (path.parent / target).exists():
                    errors.append(f"Missing relative link in {relative}: {target}")
    config_names = {p.name for p in (ROOT / "configs").glob("*.yaml")}
    packaged = ROOT / "src/mri_vla_jepa/resources/configs"
    if config_names != {p.name for p in packaged.glob("*.yaml")}:
        errors.append("Root and packaged configuration filenames differ")
    for name in sorted(config_names):
        resource = packaged / name
        if resource.exists() and (ROOT / "configs" / name).read_bytes() != resource.read_bytes():
            errors.append(f"Packaged configuration differs: {name}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Publication checks passed: {len(files)} tracked text files, {len(config_names)} configuration pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
