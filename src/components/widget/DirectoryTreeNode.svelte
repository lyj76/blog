<script context="module" lang="ts">
import { writable } from "svelte/store";

// 悬停提示框：所有目录树节点共享同一个 tooltip（模块级单例），
// 与右侧 TOC 的提示框共用 .toc-tooltip 主题样式。
let tooltip: HTMLDivElement | null = null;

// 当前页面路径（响应式 store）：swup 导航时更新，让高亮跟随页面切换。
// 侧边栏不在 swup 的 containers 里，DOM 不会刷新，所以必须用 store 通知。
export const activePathStore = writable("");

let swupBound = false;
function bindSwupPathSync() {
	if (swupBound) return;
	swupBound = true;
	const bind = () => {
		window.swup?.hooks?.on("visit:start", (visit: { to: { url: string } }) => {
			activePathStore.set(visit.to.url.split("?")[0]);
		});
	};
	if (window?.swup?.hooks) {
		bind();
	} else {
		document.addEventListener("swup:enable", bind);
	}
}

function getTooltip(): HTMLDivElement {
	if (!tooltip) {
		tooltip = document.createElement("div");
		tooltip.className = "toc-tooltip";
		document.body.appendChild(tooltip);
	}
	return tooltip;
}

function showTooltip(target: HTMLElement, text: string) {
	if (!text) return;
	const tip = getTooltip();
	tip.textContent = text;

	// 先显示拿到实际尺寸，再做边界钳制
	tip.classList.add("visible");
	tip.style.left = "0px";
	tip.style.top = "0px";
	const tw = tip.offsetWidth;
	const th = tip.offsetHeight;
	const rect = target.getBoundingClientRect();
	let left = Math.min(rect.left, window.innerWidth - tw - 12);
	let top = rect.bottom + 8;
	if (top + th > window.innerHeight - 12) {
		top = rect.top - th - 8; // 下方放不下就移到上方
	}
	tip.style.left = `${Math.max(12, left)}px`;
	tip.style.top = `${top}px`;
}

function hideTooltip() {
	tooltip?.classList.remove("visible");
}

// 滚动页面时隐藏提示框（swup 导航触发的滚动也会走到这里）
if (typeof document !== "undefined") {
	document.addEventListener("scroll", hideTooltip, { passive: true });
	bindSwupPathSync();
}
</script>

<script lang="ts">
import { onMount } from "svelte";
import type { FileTreeNode } from "@utils/tree-utils";
import Icon from "@iconify/svelte";

export let node: FileTreeNode;
export let currentPath: string = "";
export let level: number = 0;

let isOpen = true;

function toggle() {
	isOpen = !isOpen;
}

// 水合后用 SSR 传入的初始路径填充 store（保证首次加载高亮正确）
onMount(() => {
	activePathStore.set(currentPath);
});

$: isFile = node.type === "file";
$: activePath = $activePathStore || currentPath;
$: isActive = isFile && node.url && activePath.includes(encodeURI(node.url).replace(/\/$/, ""));
</script>

<div class="tree-node flex flex-col text-sm select-none">
    {#if isFile}
        <a
            href={node.url}
            class="flex items-center gap-2 py-1.5 px-2 rounded-lg transition-all duration-200 hover:bg-black/5 dark:hover:bg-white/10 text-neutral-700 dark:text-neutral-300 hover:text-[var(--primary)] dark:hover:text-[var(--primary)]"
            class:active-link={isActive}
            style="padding-left: {level * 0.75 + 0.5}rem"
            on:mouseenter={(e) => showTooltip(e.currentTarget as HTMLElement, node.name)}
            on:mouseleave={hideTooltip}
        >
            <Icon icon="material-symbols:description-outline-rounded" class={isActive ? "text-base shrink-0 text-[var(--primary)]" : "text-base shrink-0 text-neutral-400 dark:text-neutral-500"} />
            <span class="truncate">{node.name}</span>
        </a>
    {:else}
        <button
            on:click={toggle}
            on:mouseenter={(e) => showTooltip(e.currentTarget as HTMLElement, node.name)}
            on:mouseleave={hideTooltip}
            class="flex items-center justify-between py-1.5 px-2 rounded-lg transition-all duration-200 hover:bg-black/5 dark:hover:bg-white/10 text-neutral-800 dark:text-neutral-200 font-medium w-full text-left"
            style="padding-left: {level * 0.75 + 0.5}rem"
        >
            <div class="flex items-center gap-2 overflow-hidden">
                <Icon
                    icon={isOpen ? "material-symbols:folder-open-outline-rounded" : "material-symbols:folder-outline-rounded"}
                    class="text-base shrink-0 text-[var(--primary)]"
                />
                <span class="truncate">{node.name}</span>
            </div>
            {#if node.count !== undefined}
                <span class="text-xs px-1.5 py-0.5 rounded-full bg-black/5 dark:bg-white/10 text-neutral-500 dark:text-neutral-400 shrink-0 ml-1">
                    {node.count}
                </span>
            {/if}
        </button>

        {#if isOpen && node.children.length > 0}
            <div class="tree-children flex flex-col border-l border-neutral-200 dark:border-neutral-800 ml-3.5 my-0.5">
                {#each node.children as childNode}
                    <svelte:self node={childNode} {currentPath} level={level + 1} />
                {/each}
            </div>
        {/if}
    {/if}
</div>

<style>
    .active-link {
        background-color: var(--btn-plain-bg-active, rgba(0, 0, 0, 0.05));
        color: var(--primary);
        font-weight: 600;
    }
    :global(.dark) .active-link {
        background-color: rgba(255, 255, 255, 0.1);
        color: var(--primary);
    }
</style>
