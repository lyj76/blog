/// <reference types="mdast" />
import { h } from "hastscript";

/**
 * Creates an admonition component.
 *
 * @param {Object} properties - The properties of the component.
 * @param {string} [properties.title] - An optional title.
 * @param {('tip'|'note'|'important'|'caution'|'warning')} type - The admonition type.
 * @param {import('mdast').RootContent[]} children - The children elements of the component.
 * @returns {import('mdast').Parent} The created admonition component.
 */
export function AdmonitionComponent(properties, children, type) {
	if (!Array.isArray(children) || children.length === 0)
		return h(
			"div",
			{ class: "hidden" },
			'Invalid admonition directive. (Admonition directives must be of block type ":::note{name="name"} <content> :::")',
		);

	let label = null;
	if (properties?.["has-directive-label"]) {
		label = children[0]; // The first child is the label
		// biome-ignore lint/style/noParameterAssign: <check later>
		children = children.slice(1);
		label.tagName = "div"; // Change the tag <p> to <div>
	}

	return h("blockquote", { class: `admonition bdm-${type}` }, [
		h("span", { class: "bdm-title" }, label ? label : type.toUpperCase()),
		...children,
	]);
}

/**
 * amsthm 式证明块：:::proof
 * 加粗「证明」标题 + 散文/居中数学内容 + 右对齐实心方块 ■（QED）。
 *
 * @param {Object} properties - The properties of the component.
 * @param {import('mdast').RootContent[]} children - The children elements of the component.
 * @returns {import('mdast').Parent} The created proof block.
 */
export function ProofComponent(properties, children) {
	if (!Array.isArray(children) || children.length === 0)
		return h("div", { class: "hidden" }, "Invalid proof directive. (:::proof <content> :::)");

	let label = null;
	if (properties?.["has-directive-label"]) {
		label = children[0];
		// biome-ignore lint/style/noParameterAssign: <check later>
		children = children.slice(1);
		label.tagName = "div"; // Change the tag <p> to <div>
	}

	return h("div", { class: "proof-block" }, [
		h("div", { class: "proof-title" }, label ? label : "证明"),
		...children,
		h("span", { class: "proof-qed", "aria-hidden": "true" }, "■"),
	]);
}
