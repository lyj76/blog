import { visit } from "unist-util-visit";

/* 荧光笔条目标题：## <<名称>> → 去掉 <<>> 标记、给 heading 加 hl-heading 类，
 * CSS 渲染成荧光笔高亮（醒目、一眼看出这是检索锚点）；
 * 文本清洗后 rehype-slug 对干净名称生成锚点 id（= 名称本身）。 */

const HIGHLIGHT_RE = /<<([^<>]+)>>/;

export function remarkHighlightHeading() {
	return (tree) => {
		visit(tree, "heading", (node) => {
			let hit = false;
			for (const child of node.children) {
				if (child.type === "text") {
					const m = child.value.match(HIGHLIGHT_RE);
					if (m) {
						child.value = m[1];
						hit = true;
					}
				}
			}
			if (hit) {
				// 自设锚点 id = 标题原文（去空格），rehype-slug 见已有 id 不再覆盖。
				// 约定：[[篇目#锚点]] 的锚点 = 荧光笔标题去掉空格。
				const name = node.children
					.filter((c) => c.type === "text")
					.map((c) => c.value)
					.join("")
					.trim();
				node.data = node.data || {};
				node.data.hProperties = {
					...((node.data && node.data.hProperties) || {}),
					className: ["hl-heading"],
					id: name.replace(/\s+/g, ""),
				};
			}
		});
	};
}
