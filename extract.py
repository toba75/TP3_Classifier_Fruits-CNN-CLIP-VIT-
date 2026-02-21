#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def resolve_zip_path(user_path: Path | None) -> Path:
    if user_path is not None:
        return user_path

    candidates = [
        Path("fleur.zip"),
        Path("fleurs.zip"),
        Path("docs/sujet/fleur.zip"),
        Path("docs/sujet/fleurs.zip"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Aucun zip trouvé automatiquement. Utilise --zip-path <chemin> (ex: docs/sujet/fleurs.zip)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrait uniquement les répertoires train/ et test/ depuis fleur(s).zip vers data/."
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Chemin vers fleur.zip/fleurs.zip (défaut: détection auto)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Dossier de sortie (défaut: data)",
    )
    args = parser.parse_args()

    zip_path = resolve_zip_path(args.zip_path)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = archive.namelist()

        targets = []
        for name in members:
            normalized = name.replace("\\", "/")
            if "/train/" in normalized or "/test/" in normalized or normalized.startswith("train/") or normalized.startswith("test/"):
                targets.append(name)

        for member in targets:
            archive.extract(member, path=out_dir)
            extracted += 1

    print(f"Zip: {zip_path}")
    print(f"Sortie: {out_dir}")
    print(f"Entrées extraites (train/test): {extracted}")


if __name__ == "__main__":
    main()
