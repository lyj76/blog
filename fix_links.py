import os
import re

DIR = "/mnt/e/test/fuwari/src/content/posts/点拓"

FIXES = {
    "Rm序列收敛定理": "$\\mathbb{R}^m$ 空间的序列收敛定理",
    "BW定理的序列证明": "Bolzano-Weierstrass 定理的序列证明",
    "BW定理的紧证明": "Bolzano-Weierstrass 定理的紧致性证明",
    "R的连通性": "实数集 ($\\mathbb{R}$) 的连通性",
    "R与闭区间不可数": "实数集与闭区间不可数定理",
    "紧致集完备定理": "紧致空间必定完备",
}

for filename in os.listdir(DIR):
    if not filename.endswith(".md"): continue
    filepath = os.path.join(DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for old, new in FIXES.items():
        new_content = new_content.replace(f"#{old}]]", f"#{new}]]")
        
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed links in {filename}")

