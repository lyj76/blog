import os
import re

DIR = "/mnt/e/test/fuwari/src/content/posts/点拓"
files = [f for f in os.listdir(DIR) if f.endswith(".md")]

all_headings = set()
for filename in files:
    filepath = os.path.join(DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'## <<(.+?)>>', line)
            if match:
                clean_filename = filename.replace('.md', '')
                all_headings.add(f"{clean_filename}#{match.group(1)}")

broken_links = []
for filename in files:
    filepath = os.path.join(DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        links = re.findall(r'\[\[([^\]]*?#([^\]]*?))\]\]', content)
        for full_link, heading in links:
            if full_link not in all_headings:
                broken_links.append((filename, full_link))

if broken_links:
    for source, broken in broken_links:
        print(f"BROKEN: In {source}: {broken}")
else:
    print("ALL OK")
