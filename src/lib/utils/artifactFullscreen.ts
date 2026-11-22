export const ARTIFACT_APP_FULLSCREEN_BODY_CLASS = 'artifact-app-fullscreen-active';

export const ARTIFACT_FULLSCREEN_UNAVAILABLE_MESSAGE =
	'Full screen is unavailable in this browser or embedded context.';

export type FullscreenIframe = HTMLIFrameElement & {
	webkitRequestFullscreen?: () => void | Promise<void>;
	msRequestFullscreen?: () => void | Promise<void>;
};

export type ArtifactFullscreenMode = 'app' | 'element';

export function isStandaloneWebApp(
	win: Window | undefined = typeof window !== 'undefined' ? window : undefined
): boolean {
	if (!win) return false;

	const nav = win.navigator as Navigator & { standalone?: boolean };
	if (nav.standalone === true) return true;

	return win.matchMedia?.('(display-mode: standalone)').matches ?? false;
}

export function getArtifactFullscreenMode(
	win: Window | undefined = typeof window !== 'undefined' ? window : undefined
): ArtifactFullscreenMode {
	return isStandaloneWebApp(win) ? 'app' : 'element';
}

export function setArtifactAppFullscreenChrome(
	active: boolean,
	doc: Document | undefined = typeof document !== 'undefined' ? document : undefined
): void {
	if (!doc) return;
	doc.body.classList.toggle(ARTIFACT_APP_FULLSCREEN_BODY_CLASS, active);
}

export async function requestArtifactIframeFullscreen(iframe: FullscreenIframe): Promise<void> {
	if (typeof iframe.requestFullscreen === 'function') {
		await iframe.requestFullscreen();
		return;
	}
	if (typeof iframe.webkitRequestFullscreen === 'function') {
		await iframe.webkitRequestFullscreen();
		return;
	}
	if (typeof iframe.msRequestFullscreen === 'function') {
		await iframe.msRequestFullscreen();
		return;
	}
	throw new Error('Fullscreen API is unavailable');
}
