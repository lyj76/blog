<script lang="ts">
import type { FileTreeNode } from "@utils/tree-utils";
import Icon from "@iconify/svelte";

export let node: FileTreeNode;
export let currentPath: string = "";
export let level: number = 0;

let isOpen = true;

function toggle() {
	isOpen = !isOpen;
}

$: isFile = node.type === "file";
$: isActive = isFile && node.url && currentPath.includes(encodeURI(node.url).replace(/\/$/, ""));
</script>

<div class="tree-node flex flex-col text-sm select-none">
    {#if isFile}
        <a
            href={node.url}
            class="flex items-center gap-2 py-1.5 px-2 rounded-lg transition-all duration-200 hover:bg-black/5 dark:hover:bg-white/10 text-neutral-700 dark:text-neutral-300 hover:text-[var(--primary)] dark:hover:text-[var(--primary)]"
            class:active-link={isActive}
            style="padding-left: {level * 0.75 + 0.5}rem"
        >
            <Icon icon="material-symbols:description-outline-rounded" class={isActive ? "text-base shrink-0 text-[var(--primary)]" : "text-base shrink-0 text-neutral-400 dark:text-neutral-500"} />
            <span class="truncate">{node.name}</span>
        </a>
    {:else}
        <button
            on:click={toggle}
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
