import os
import re

DIR = "/mnt/e/test/fuwari/src/content/posts/点拓"

RENAME_MAP = {
    "00-总览": "00-点集拓扑总览",
    "01-度量空间": "01-度量空间与序列收敛",
    "02-连续映射": "02-连续映射与同胚",
    "03-开集与闭集": "03-开集、闭集与拓扑子空间",
    "04-闭包与导集": "04-闭包、导集与聚点",
    "05-各种度量与乘积": "05-等价度量与乘积空间",
    "06-完备性": "06-完备性与柯西序列",
    "07-紧致性": "07-序列紧致与海涅-博雷尔定理",
    "08-紧致性与映射": "08-紧致性与连续映射",
    "09-连通性": "09-连通性与介值定理",
    "10-覆盖紧致与勒贝格数": "10-覆盖紧致与勒贝格数",
    "11-康托尔集": "11-康托尔集的分形与拓扑性质"
}

files = [f for f in os.listdir(DIR) if f.endswith(".md")]

# First, update all links in all files
for filename in files:
    filepath = os.path.join(DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    new_content = content
    for old, new in RENAME_MAP.items():
        if old == new: continue
        # Update [[old]] -> [[new]]
        new_content = new_content.replace(f"[[{old}]]", f"[[{new}]]")
        # Update [[old#heading]] -> [[new#heading]]
        new_content = new_content.replace(f"[[{old}#", f"[[{new}#")
        # Update [[old|alias]] -> [[new|alias]]
        new_content = new_content.replace(f"[[{old}|", f"[[{new}|")
        
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated links in {filename}")

# Then, rename the files
for old, new in RENAME_MAP.items():
    if old == new: continue
    old_file = os.path.join(DIR, f"{old}.md")
    new_file = os.path.join(DIR, f"{new}.md")
    if os.path.exists(old_file):
        os.rename(old_file, new_file)
        print(f"Renamed {old}.md -> {new}.md")

