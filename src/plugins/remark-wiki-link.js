import fs from "node:fs";
import path from "node:path";
import { visit } from "unist-util-visit";

/* Obsidian 式 wiki 链接：[[文章标题]] 或 [[文件名]] 或 [[标题|别名]]
 * 构建时扫描 src/content/posts/ 建立「名字 → URL」映射（模块级缓存，只扫一次），
 * 渲染时把 [[...]] 替换成可点击的链接；找不到目标的标灰显示。 */

const WIKI_RE = /\[\[([^\[\]]+)\]\]/g;
const POSTS_DIR = path.join(process.cwd(), "src/content/posts");

function readFrontmatterTitle(file) {
	const src = fs.readFileSync(file, "utf8");
	const fm = src.match(/^---\r?\n([\s\S]*?)\r?\n---/);
	if (!fm) return null;
	const titleLine = fm[1].match(/^title:\s*(.+)$/m);
	if (!titleLine) return null;
	return titleLine[1].trim().replace(/^["']|["']$/g, "");
}

function buildWikiLinkMap() {
	const map = new Map();

	const walk = (dir) => {
		for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
			const full = path.join(dir, entry.name);
			if (entry.isDirectory()) {
				walk(full);
			} else if (/\.(md|mdx)$/.test(entry.name)) {
				const rel = path
					.relative(POSTS_DIR, full)
					.replace(/\\/g, "/")
					.replace(/\.(md|mdx)$/, "");
				const name = entry.name.replace(/\.(md|mdx)$/, "");
				const url = `/posts/${rel}/`;
				const title = readFrontmatterTitle(full);
				if (title) map.set(title, url);
				map.set(name, url);
				map.set(rel, url);
			}
		}
	};

	walk(POSTS_DIR);
	return map;
}

/* 模块级缓存：整次构建只扫描一次目录（避免每渲染一篇 md 就 O(N) 扫全库）。
 * dev 下用目录 mtime 失效重建，增删文章后无需重启也保持准确。 */
let cachedMap = null;
let cachedMtime = 0;

function getWikiLinkMap() {
	const mtime = fs.statSync(POSTS_DIR).mtimeMs;
	if (!cachedMap || mtime !== cachedMtime) {
		cachedMap = buildWikiLinkMap();
		cachedMtime = mtime;
	}
	return cachedMap;
}

export function remarkWikiLink() {
	const map = getWikiLinkMap();

	return (tree) => {
		visit(tree, "text", (node, index, parent) => {
			if (!parent || index === undefined) return;
			// 跳过代码块里的 [[...]]
			if (parent.type === "code" || parent.type === "inlineCode") return;
			const value = node.value;
			if (!value.includes("[[")) return;

			const segments = [];
			let last = 0;
			let matched = false;

			for (const m of value.matchAll(WIKI_RE)) {
				matched = true;
				if (m.index > last) {
					segments.push({ type: "text", value: value.slice(last, m.index) });
				}
				const raw = m[1];
				const [name, alias] = raw.split("|").map((s) => s.trim());
				const url = map.get(name);
				if (url) {
					segments.push({
						type: "link",
						url,
						data: { hProperties: { "data-wiki-link": "true" } },
						children: [{ type: "text", value: alias || name }],
					});
				} else {
					segments.push({
						type: "html",
						value: `<span class="wiki-unresolved" title="未找到文章：${name}">[[${raw}]]</span>`,
					});
				}
				last = m.index + m[0].length;
			}

			if (!matched) return;
			if (last < value.length) {
				segments.push({ type: "text", value: value.slice(last) });
			}
			parent.children.splice(index, 1, ...segments);
			return index + segments.length - 1;
		});
	};
}
