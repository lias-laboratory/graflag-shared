"""Install DGL 2.4.0 (torch 2.4, CUDA 12.1, CPython 3.12) from the DGL team's conda channel.

    python3 install_dgl.py

Why not pip: the wheel index DGL documents, data.dgl.ai, failed the TLS
handshake from every network tried on 2026-11-17 (SSL_ERROR_SYSCALL), and the
S3 bucket behind it serves repo.html but answers 403 for the wheel itself. The
same version, for the same torch and CUDA, is published by the DGL team on
anaconda.org (channel dglteam, label th24_cu121). It is a separate build, not
the wheel repackaged: its sparse library links libcuda.so.1 directly, which
is why the Dockerfile checks dgl.sparse at run time rather than at build time. A conda package is a
tarball whose lib/python3.12/site-packages/ is the installed tree, egg-info
included, so it is extracted next to torch and pip then sees it as installed.

Fails loudly: the archive's SHA-256 must equal the one pinned here (taken
from the channel's repodata.json), and the extraction must produce the dgl
package and its metadata. Dependencies are not resolved -- the Dockerfile
installs them by name, from the egg-info's requires.txt.
"""

import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = ("https://conda.anaconda.org/dglteam/label/th24_cu121/linux-64/"
       "dgl-2.4.0.th24.cu121-py312_0.tar.bz2")
SHA256 = "6a0746cb2cb7610e9b47b184155506b2db4787e8deb17550ccd6b4596f1f382d"
PREFIX = "lib/python3.12/site-packages/"


def main() -> int:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"[ERROR] this build is for CPython 3.12, not {sys.version.split()[0]}")
    import torch  # installed first by the Dockerfile; DGL goes beside it
    site = Path(torch.__file__).resolve().parents[1]

    with urllib.request.urlopen(URL, timeout=600) as resp:
        blob = resp.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"[ERROR] {URL}: sha256 {digest}, expected {SHA256}")
    print(f"[OK] {len(blob)} bytes, sha256 {digest}")

    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:bz2") as tar:
        members = []
        for m in tar.getmembers():
            if not m.name.startswith(PREFIX) or m.name == PREFIX.rstrip("/"):
                continue
            m.name = m.name[len(PREFIX):]
            if m.name.startswith(("/", "..")) or "/../" in m.name:
                raise SystemExit(f"[ERROR] unsafe path in archive: {m.name}")
            members.append(m)
        if not members:
            raise SystemExit(f"[ERROR] nothing under {PREFIX} in the archive")
        tar.extractall(site, members=members, filter="tar")

    if not (site / "dgl" / "__init__.py").is_file():
        raise SystemExit(f"[ERROR] no dgl package under {site} after extraction")
    if not list(site.glob("dgl-2.4.0*.egg-info")):
        raise SystemExit(f"[ERROR] no dgl egg-info under {site}; pip would not see it")
    print(f"[OK] dgl 2.4.0+cu121 extracted into {site} ({len(members)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
