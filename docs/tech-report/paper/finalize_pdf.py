# /// script
# requires-python = ">=3.10"
# dependencies = ["pypdf>=5"]
# ///
"""Copy main.pdf to the released path without creation/modification dates in its document information.

    uv run finalize_pdf.py main.pdf ../pev.pdf
"""
import sys

from pypdf import PdfReader, PdfWriter

src, dst = sys.argv[1:3]
reader = PdfReader(src)
writer = PdfWriter(clone_from=reader)
info = {k: v for k, v in (reader.metadata or {}).items() if k not in ("/CreationDate", "/ModDate", "/Producer")}
writer.metadata = None
writer.add_metadata(info)
with open(dst, "wb") as fh:
    writer.write(fh)
print(f"{dst}: {len(writer.pages)} pages, info keys {sorted(info)}")
