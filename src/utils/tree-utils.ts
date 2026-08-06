import { getCollection } from "astro:content";
import { getPostUrlBySlug } from "./url-utils";

export interface FileTreeNode {
	name: string;
	type: "dir" | "file";
	path: string;
	slug?: string;
	url?: string;
	count?: number;
	category?: string; // 文件夹下文章的分类（用于计数徽章跳转归档页）
	children: FileTreeNode[];
}

export async function getDirectoryTree(): Promise<FileTreeNode[]> {
	const allPosts = await getCollection("posts", ({ data }) => {
		return import.meta.env.PROD ? data.draft !== true : true;
	});

	const root: FileTreeNode[] = [];

	for (const post of allPosts) {
		// post.id is like "计算机/操作系统/进程.md" or "markdown.md"
		const normalizedId = post.id.replace(/\\/g, "/");
		const parts = normalizedId.split("/");
		const postCategory =
			typeof post.data.category === "string" && post.data.category.trim()
				? post.data.category.trim()
				: undefined;

		let currentLevel = root;
		let currentPath = "";

		// Process directory parts
		for (let i = 0; i < parts.length - 1; i++) {
			const dirName = parts[i];
			currentPath = currentPath ? `${currentPath}/${dirName}` : dirName;

			let dirNode = currentLevel.find(
				(node) => node.type === "dir" && node.name === dirName,
			);

			if (!dirNode) {
				dirNode = {
					name: dirName,
					type: "dir",
					path: currentPath,
					count: 0,
					children: [],
					category: postCategory,
				};
				currentLevel.push(dirNode);
			}

			if (dirNode.count !== undefined) {
				dirNode.count++;
			}

			if (postCategory && !dirNode.category) {
				dirNode.category = postCategory;
			}

			currentLevel = dirNode.children;
		}

		// Add file node
		const fileName = parts[parts.length - 1];
		currentLevel.push({
			name: post.data.title || fileName.replace(/\.(md|mdx)$/i, ""),
			type: "file",
			path: normalizedId,
			slug: post.slug,
			url: getPostUrlBySlug(post.slug),
			children: [],
		});
	}

	// Helper to recursively sort tree: directories first, then alphabetical
	function sortNodes(nodes: FileTreeNode[]): FileTreeNode[] {
		nodes.sort((a, b) => {
			if (a.type !== b.type) {
				return a.type === "dir" ? -1 : 1;
			}
			return a.name.localeCompare(b.name, "zh-CN");
		});

		for (const node of nodes) {
			if (node.type === "dir" && node.children.length > 0) {
				sortNodes(node.children);
			}
		}

		return nodes;
	}

	return sortNodes(root);
}
