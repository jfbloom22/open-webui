<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { getContext, onDestroy } from 'svelte';
	import type { Readable } from 'svelte/store';
	const i18n: any = getContext('i18n');

	import { config, settings, showSidebar } from '$lib/stores';
	import { copyToClipboard } from '$lib/utils';
	import {
		ARTIFACT_FULLSCREEN_UNAVAILABLE_MESSAGE,
		getArtifactFullscreenMode,
		requestArtifactIframeFullscreen,
		setArtifactAppFullscreenChrome,
		type FullscreenIframe
	} from '$lib/utils/artifactFullscreen';
	import { injectCsp } from '$lib/utils/csp';

	import Icon from './FileNav/Icon.svelte';
	import Tooltip from '../common/Tooltip.svelte';
	import SvgPanZoom from '../common/SVGPanZoom.svelte';

	export let overlay = false;
	export let artifacts: Readable<{ type: string; content: string }[]>;
	/** The code block to show; a new object selects it again. */
	export let selection: { code: string } | null = null;
	export let chatId = '';

	let contents: { type: string; content: string }[] = [];
	let selectedContentIdx = 0;
	let appliedSelection: typeof selection = null;

	const updateContents = (next: { type: string; content: string }[]) => {
		if (next.length === 0) selectedContentIdx = 0;
		else if (next.length > contents.length) selectedContentIdx = next.length - 1;
		else if (selectedContentIdx > next.length - 1) selectedContentIdx = next.length - 1;
		return next;
	};
	$: contents = updateContents($artifacts ?? []);
	$: if (selection && selection !== appliedSelection && contents.length > 0) {
		appliedSelection = selection;
		const codeIdx = contents.findIndex((content) => content.content.includes(selection.code));
		if (codeIdx !== -1) selectedContentIdx = codeIdx;
	}

	let copied = false;
	let iframeElement: HTMLIFrameElement;
	let fullscreenError = '';
	let appFullscreen = false;

	onDestroy(() => setArtifactAppFullscreenChrome(false));

	function navigateContent(direction: 'prev' | 'next') {
		selectedContentIdx =
			direction === 'prev'
				? Math.max(selectedContentIdx - 1, 0)
				: Math.min(selectedContentIdx + 1, contents.length - 1);
	}

	const iframeLoadHandler = () => {
		iframeElement.contentWindow.addEventListener(
			'click',
			function (e) {
				const target = e.target.closest('a');
				if (target && target.href) {
					e.preventDefault();
					const url = new URL(target.href, iframeElement.baseURI);
					if (url.origin === window.location.origin) {
						iframeElement.contentWindow.history.pushState(
							null,
							'',
							url.pathname + url.search + url.hash
						);
					} else {
						console.info('External navigation blocked:', url.href);
					}
				}
			},
			true
		);

		// Cancel drag when hovering over iframe
		iframeElement.contentWindow.addEventListener('mouseenter', function (e) {
			e.preventDefault();
			iframeElement.contentWindow.addEventListener('dragstart', (event) => {
				event.preventDefault();
			});
		});
	};

	const showFullScreen = async () => {
		fullscreenError = '';

		// Standalone iPad web apps already run without Safari chrome. Use an
		// app-level viewport mode because element fullscreen is unreliable there.
		if (getArtifactFullscreenMode() === 'app') {
			const enteringFullscreen = !appFullscreen;
			if (enteringFullscreen) showSidebar.set(false);
			setArtifactAppFullscreenChrome(enteringFullscreen);
			appFullscreen = enteringFullscreen;
			return;
		}

		const iframe = iframeElement as FullscreenIframe | undefined;
		if (!iframe) {
			fullscreenError = ARTIFACT_FULLSCREEN_UNAVAILABLE_MESSAGE;
			toast.error(ARTIFACT_FULLSCREEN_UNAVAILABLE_MESSAGE);
			return;
		}

		try {
			await requestArtifactIframeFullscreen(iframe);
		} catch (error) {
			console.error('Failed to enter artifact fullscreen:', error);
			fullscreenError = ARTIFACT_FULLSCREEN_UNAVAILABLE_MESSAGE;
			toast.error(ARTIFACT_FULLSCREEN_UNAVAILABLE_MESSAGE);
		}
	};

	const downloadArtifact = () => {
		const blob = new Blob([contents[selectedContentIdx].content], { type: 'text/html' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `artifact-${chatId}-${selectedContentIdx}.html`;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
	};
</script>

<div
	class="artifact-container w-full h-full relative flex flex-col bg-white dark:bg-gray-850"
	class:artifact-app-fullscreen={appFullscreen}
	id="artifacts-container"
>
	<div class="w-full min-h-0 flex flex-col flex-1 relative">
		{#if contents.length > 0}
			<div
				class="pointer-events-auto z-20 shrink-0 flex items-center justify-between gap-2 px-2 pb-1.5"
			>
				<div class="flex min-w-0 items-center gap-0.5" dir="ltr">
					<button
						aria-label={$i18n.t('Previous version')}
						class="flex h-5 w-5 shrink-0 items-center justify-center rounded text-gray-400 transition-colors duration-100 hover:text-gray-600 disabled:text-gray-200 dark:text-gray-500 dark:hover:text-gray-300 dark:disabled:text-gray-700"
						on:click={() => navigateContent('prev')}
						disabled={selectedContentIdx === 0}
						><Icon name="chevron-left" size={11} strokeWidth={1.5} /></button
					>
					<span class="truncate text-xs text-gray-500 dark:text-gray-400">
						{$i18n.t('Version {{selectedVersion}} of {{totalVersions}}', {
							selectedVersion: selectedContentIdx + 1,
							totalVersions: contents.length
						})}
					</span>
					<button
						aria-label={$i18n.t('Next version')}
						class="flex h-5 w-5 shrink-0 items-center justify-center rounded text-gray-400 transition-colors duration-100 hover:text-gray-600 disabled:text-gray-200 dark:text-gray-500 dark:hover:text-gray-300 dark:disabled:text-gray-700"
						on:click={() => navigateContent('next')}
						disabled={selectedContentIdx === contents.length - 1}
						><Icon name="chevron-right" size={11} strokeWidth={1.5} /></button
					>
				</div>
				<div class="flex shrink-0 items-center gap-0.5">
					<Tooltip content={copied ? $i18n.t('Copied') : $i18n.t('Copy')}>
						<button
							aria-label={copied ? $i18n.t('Copied') : $i18n.t('Copy')}
							class="flex h-5 w-5 items-center justify-center rounded text-gray-400 transition-colors duration-100 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300"
							on:click={() => {
								copyToClipboard(contents[selectedContentIdx].content);
								copied = true;
								setTimeout(() => {
									copied = false;
								}, 2000);
							}}><Icon name={copied ? 'check' : 'copy'} size={11} strokeWidth={1.4} /></button
						>
					</Tooltip>
					<Tooltip content={$i18n.t('Download')}>
						<button
							aria-label={$i18n.t('Download')}
							class="flex h-5 w-5 items-center justify-center rounded text-gray-400 transition-colors duration-100 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300"
							on:click={downloadArtifact}
							><Icon name="download" size={11} strokeWidth={1.4} /></button
						>
					</Tooltip>
					{#if contents[selectedContentIdx].type === 'iframe'}
						<Tooltip content={$i18n.t('Open in full screen')} touch={false}>
							<button
								type="button"
								aria-label={$i18n.t('Open in full screen')}
								aria-pressed={appFullscreen}
								class="flex h-5 w-5 items-center justify-center rounded text-gray-400 transition-colors duration-100 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300"
								on:click={showFullScreen}><Icon name="expand" size={11} strokeWidth={1.4} /></button
							>
						</Tooltip>
					{/if}
				</div>
			</div>
			{#if fullscreenError}
				<div class="px-2.5 pb-1 text-xs text-red-600 dark:text-red-400" role="alert">
					{fullscreenError}
				</div>
			{/if}
		{/if}

		{#if overlay}
			<div class=" absolute top-0 left-0 right-0 bottom-0 z-10"></div>
		{/if}

		<div class="flex-1 w-full min-h-0 bg-white dark:bg-gray-850">
			<div class=" h-full flex flex-col">
				{#if contents.length > 0}
					<div class="max-w-full w-full h-full">
						{#if contents[selectedContentIdx].type === 'iframe'}
							<iframe
								bind:this={iframeElement}
								title={$i18n.t('Content')}
								srcdoc={injectCsp(
									contents[selectedContentIdx].content,
									$config?.ui?.iframe_csp ?? ''
								)}
								class="w-full border-0 h-full rounded-none"
								allowfullscreen
								allow="fullscreen"
								sandbox="{($settings?.iframeSandboxAllowScripts ?? true)
									? 'allow-scripts'
									: ''}{($settings?.iframeSandboxAllowDownloads ?? true)
									? ' allow-downloads'
									: ''}{($settings?.iframeSandboxAllowForms ?? true)
									? ' allow-forms'
									: ''}{($settings?.iframeSandboxAllowSameOrigin ?? false)
									? ' allow-same-origin'
									: ''}"
								on:load={iframeLoadHandler}
							></iframe>
						{:else if contents[selectedContentIdx].type === 'svg'}
							<SvgPanZoom
								className=" w-full h-full max-h-full overflow-hidden"
								svg={contents[selectedContentIdx].content}
							/>
						{/if}
					</div>
				{:else}
					<div class="m-auto p-6 text-center font-normal text-xs text-gray-900 dark:text-white">
						{$i18n.t('No HTML, CSS, or JavaScript content found.')}
					</div>
				{/if}
			</div>
		</div>
	</div>
</div>

<style>
	.artifact-app-fullscreen {
		position: fixed;
		inset: 0;
		z-index: 1000;
		width: 100vw;
		height: 100dvh;
		overflow: hidden;
		padding-top: env(safe-area-inset-top);
		padding-bottom: env(safe-area-inset-bottom);
	}

	:global(body.artifact-app-fullscreen-active #sidebar),
	:global(body.artifact-app-fullscreen-active #controls-resizer) {
		visibility: hidden;
		pointer-events: none;
	}
</style>
