import fs from "node:fs";
import path from "node:path";
import { visit } from "unist-util-visit";

/* Obsidian 式 wiki 链接：[[文件名]]（主键） / [[相对路径]] / [[标题]]（fallback）
 *
 * - 构建时扫描 src/content/posts/ 建立三张「名字 → URL」映射，模块级缓存，
 *   签名（所有 md 文件 mtime + 目录 mtime）变化才重建——增删改文章都自动失效。
 * - 解析顺序：文件名 > 相对路径 > 标题。文件名是文章最稳定的标识（重组改标题、
 *   改路径都不影响）；标题是可变的展示层——用标题命中会打警告，建议改用文件名。
 * - 诊断收集（构建期间累加，构建结束时 flush）：
 *     · 文件名/标题冲突   → 报警（裸名可能指向错误目标，需用相对路径区分）
 *     · 标题命中          → 报警（建议改文件名）
 *     · 未解析            → 报警（渲染为灰色虚线 + file:line）
 *   WIKI_LINK_STRICT=true 时，存在冲突或未解析链接会让构建失败（CI 用）。
 */

const WIKI_RE = /\[\[([^\[\]]+)\]\]/g;
const POSTS_DIR = path.join(process.cwd(), "src/content/posts");

const diagnostics = {
	collisions: [], // { key, kind, fileA, fileB, urlA, urlB }
	titleHits: [], // { file, line, name, suggest }
	unresolved: [], // { file, line, name }
};

function relFrom(file) {
	return path.relative(POSTS_DIR, file).replace(/\\/g, "/").replace(/\.(md|mdx)$/, "");
}

function relFile(vfile) {
	try {
		return path.relative(POSTS_DIR, vfile.path || "");
	} catch {
		return String(vfile.path || "?");
	}
}

function warn(msg) {
	console.warn(msg); // 开发/构建时立即反馈
}

function recordCollision(entry) {
	diagnostics.collisions.push(entry);
	warn(
		`[wiki-link] 键冲突(${entry.kind})：「${entry.key}」同时指向\n` +
			`  ${entry.fileA} → ${entry.urlA}\n` +
			`  ${entry.fileB} → ${entry.urlB}\n` +
			`  建议把其中一处改用相对路径形式区分（如 [[${relFrom(entry.fileB)}]]）`
	);
}

function recordTitleHit(filePath, line, name, suggest) {
	diagnostics.titleHits.push({ file: filePath, line, name, suggest });
	warn(
		`[wiki-link] ${filePath}:${line} 用标题「${name}」解析成功——标题是可变的展示层，` +
			(suggest ? `建议改用文件名 [[${suggest}]]` : "建议改用文件名形式")
	);
}

function recordUnresolved(filePath, line, name) {
	diagnostics.unresolved.push({ file: filePath, line, name });
	warn(`[wiki-link] ${filePath}:${line} 未找到文章「${name}」——将渲染为灰色虚线链接`);
}

function readFrontmatterTitle(file) {
	let src;
	try {
		// 只读文件头即可拿到 frontmatter（< 16KB 时全读，防超长 description 截断）
		const fd = fs.openSync(file, "r");
		try {
			const head = Buffer.alloc(16 * 1024);
			const n = fs.readSync(fd, head, 0, head.length, 0);
			src = head.subarray(0, n).toString("utf8");
			if (!/^---\r?\n[\s\S]*?\r?\n---/.test(src)) {
				src = fs.readFileSync(file, "utf8");
			}
		} finally {
			fs.closeSync(fd);
		}
	} catch {
		// 文件在 readdir 后消失（编辑器保存 = 删除+改名）→ 跳过，不崩管线
		return null;
	}
	const fm = src.match(/^---\r?\n([\s\S]*?)\r?\n---/);
	if (!fm) return null;
	const titleLine = fm[1].match(/^title:\s*(.+)$/m);
	if (!titleLine) return null;
	return titleLine[1].trim().replace(/^["']|["']$/g, "");
}

function buildIndex() {
	const nameMap = new Map(); // 文件名 → url（主键）
	const relMap = new Map(); // 相对路径 → url
	const titleMap = new Map(); // 标题 → url（fallback）
	const nameOwner = new Map(); // 文件名 → 源文件
	const titleOwner = new Map(); // 标题 → 源文件

	const put = (map, ownerMap, kind, key, url, file) => {
		const existing = map.get(key);
		if (existing && existing !== url) {
			// 保留先注册的（扫描顺序确定），报警让作者用相对路径区分
			recordCollision({
				key,
				kind,
				fileA: ownerMap?.get(key),
				fileB: file,
				urlA: existing,
				urlB: url,
			});
			return;
		}
		map.set(key, url);
		ownerMap?.set(key, file);
	};

	const walk = (dir) => {
		for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
			const full = path.join(dir, entry.name);
			if (entry.isDirectory()) {
				walk(full);
				continue;
			}
			if (!/\.(md|mdx)$/.test(entry.name)) continue;
			const rel = path.relative(POSTS_DIR, full).replace(/\\/g, "/").replace(/\.(md|mdx)$/, "");
			const name = entry.name.replace(/\.(md|mdx)$/, "");
			const url = `/posts/${rel}/`;
			put(nameMap, nameOwner, "文件名", name, url, full);
			put(relMap, null, "相对路径", rel, url, full);
			const title = readFrontmatterTitle(full);
			if (title) put(titleMap, titleOwner, "标题", title, url, full);
		}
	};
	walk(POSTS_DIR);

	// 跨表冲突：某文件标题恰好等于另一文件文件名 → [[x]] 按优先级命中文件名
	for (const [title, url] of titleMap) {
		const nameUrl = nameMap.get(title);
		if (nameUrl && nameUrl !== url) {
			recordCollision({
				key: title,
				kind: "标题=另一文件名",
				fileA: titleOwner.get(title),
				fileB: nameOwner.get(title),
				urlA: url,
				urlB: nameUrl,
			});
		}
	}

	return { nameMap, relMap, titleMap, titleOwner };
}

function computeSignature() {
	const parts = [];
	const walk = (dir) => {
		for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
			const full = path.join(dir, entry.name);
			if (entry.isDirectory()) {
				walk(full);
			} else if (/\.(md|mdx)$/.test(entry.name)) {
				try {
					parts.push(`${path.relative(POSTS_DIR, full)}:${Math.round(fs.statSync(full).mtimeMs)}`);
				} catch {
					// stat 时文件已消失 → 跳过，等下次签名变化再重建
				}
			}
		}
	};
	walk(POSTS_DIR);
	try {
		parts.push(`root:${Math.round(fs.statSync(POSTS_DIR).mtimeMs)}`);
	} catch {}
	return parts.join("|");
}

let cachedIndex = null;
let cachedSig = "";

function getIndex() {
	const sig = computeSignature();
	if (!cachedIndex || sig !== cachedSig) {
		cachedIndex = buildIndex();
		cachedSig = sig;
	}
	return cachedIndex;
}

export function remarkWikiLink() {
	const { nameMap, relMap, titleMap, titleOwner } = getIndex();

	return (tree, file) => {
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
				const [target, alias] = raw.split("|").map((s) => s.trim());
				const [name, anchor] = target.split("#").map((s) => s.trim());

				// 解析顺序：文件名 > 相对路径 > 标题（支持 [[名#锚点]] 定位到小节）
				let url = nameMap.get(name) ?? relMap.get(name);
				if (!url && titleMap.has(name)) {
					url = titleMap.get(name);
					const owner = titleOwner.get(name);
					recordTitleHit(
						relFile(file),
						node.position?.start?.line ?? 0,
						name,
						owner ? path.basename(owner).replace(/\.(md|mdx)$/, "") : ""
					);
				}
				if (url && anchor) url = `${url}#${anchor}`;

				if (url) {
					segments.push({
						type: "link",
						url,
						data: { hProperties: { "data-wiki-link": "true" } },
						children: [{ type: "text", value: alias || (anchor ? `${name}#${anchor}` : name) }],
					});
				} else {
					recordUnresolved(relFile(file), node.position?.start?.line ?? 0, name);
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

export function flushWikiLinkDiagnostics({ strict = false } = {}) {
	const total =
		diagnostics.collisions.length + diagnostics.titleHits.length + diagnostics.unresolved.length;
	if (total > 0) {
		console.warn(
			`[wiki-link] 诊断汇总：冲突 ${diagnostics.collisions.length} / 标题命中 ${diagnostics.titleHits.length} / 未解析 ${diagnostics.unresolved.length}（共 ${total}）`
		);
	}
	const problems = diagnostics.collisions.length + diagnostics.unresolved.length;
	if (strict && problems > 0) {
		const lines = [
			...diagnostics.collisions.map(
				(c) => `  冲突(${c.kind})「${c.key}」：${c.fileA} ↔ ${c.fileB}`
			),
			...diagnostics.unresolved.map((u) => `  未解析 ${u.file}:${u.line}「${u.name}」`),
		];
		throw new Error(`[wiki-link] WIKI_LINK_STRICT：存在 ${problems} 个链接问题：\n${lines.join("\n")}`);
	}
	diagnostics.collisions.length = 0;
	diagnostics.titleHits.length = 0;
	diagnostics.unresolved.length = 0;
}
