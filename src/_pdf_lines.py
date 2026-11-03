import sys

import pymupdf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
doc = pymupdf.open(sys.argv[1])
page = doc[0]
d = page.get_text("dict")
for block in d["blocks"]:
    if block.get("type") != 0:
        continue
    for line in block["lines"]:
        y = round(line["bbox"][1], 1)
        spans = " | ".join(
            f'{s["font"]}:{s["size"]:.1f}:{s["text"][:40]}' for s in line["spans"]
        )
        print(f"y={y:>7}  {spans}")
