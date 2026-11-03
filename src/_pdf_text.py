import sys

import pymupdf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

path = sys.argv[1] if len(sys.argv) > 1 else "paper/_test.pdf"
doc = pymupdf.open(path)
print("pages:", doc.page_count)
for i, page in enumerate(doc):
    print(f"\n===== PAGE {i + 1} =====")
    txt = page.get_text()
    print(txt if txt.strip() else "(no extractable text)")
    for img in page.get_images(full=True):
        print(f"[image xref={img[0]} w={img[2]} h={img[3]}]")
