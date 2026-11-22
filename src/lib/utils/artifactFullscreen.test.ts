import { describe, expect, it, vi } from 'vitest';
import {
	ARTIFACT_APP_FULLSCREEN_BODY_CLASS,
	getArtifactFullscreenMode,
	isStandaloneWebApp,
	requestArtifactIframeFullscreen,
	setArtifactAppFullscreenChrome,
	type FullscreenIframe
} from './artifactFullscreen';

const mockWindow = (options: { standalone?: boolean; displayModeStandalone?: boolean }): Window => {
	const mediaQueryList = {
		matches: options.displayModeStandalone ?? false
	};
	return {
		navigator: { standalone: options.standalone },
		matchMedia: vi.fn(() => mediaQueryList)
	} as unknown as Window;
};

describe('isStandaloneWebApp', () => {
	it('returns false when window is undefined', () => {
		expect(isStandaloneWebApp(undefined)).toBe(false);
	});

	it('detects iOS standalone home-screen web apps', () => {
		expect(isStandaloneWebApp(mockWindow({ standalone: true }))).toBe(true);
	});

	it('detects display-mode: standalone', () => {
		expect(isStandaloneWebApp(mockWindow({ displayModeStandalone: true }))).toBe(true);
	});

	it('returns false in a normal browser tab', () => {
		expect(isStandaloneWebApp(mockWindow({}))).toBe(false);
	});
});

describe('getArtifactFullscreenMode', () => {
	it('uses app-level fullscreen in standalone web apps', () => {
		expect(getArtifactFullscreenMode(mockWindow({ standalone: true }))).toBe('app');
	});

	it('uses element fullscreen outside standalone web apps', () => {
		expect(getArtifactFullscreenMode(mockWindow({}))).toBe('element');
	});
});

describe('setArtifactAppFullscreenChrome', () => {
	it('toggles the body class used to hide sidebar chrome', () => {
		const toggle = vi.fn();
		const doc = { body: { classList: { toggle } } } as unknown as Document;

		setArtifactAppFullscreenChrome(true, doc);
		expect(toggle).toHaveBeenCalledWith(ARTIFACT_APP_FULLSCREEN_BODY_CLASS, true);

		setArtifactAppFullscreenChrome(false, doc);
		expect(toggle).toHaveBeenCalledWith(ARTIFACT_APP_FULLSCREEN_BODY_CLASS, false);
	});

	it('no-ops when document is unavailable', () => {
		expect(() => setArtifactAppFullscreenChrome(true, undefined)).not.toThrow();
	});
});

describe('requestArtifactIframeFullscreen', () => {
	it('prefers the standard Fullscreen API', async () => {
		const requestFullscreen = vi.fn().mockResolvedValue(undefined);
		const iframe = { requestFullscreen } as unknown as FullscreenIframe;

		await requestArtifactIframeFullscreen(iframe);

		expect(requestFullscreen).toHaveBeenCalledOnce();
	});

	it('falls back to webkitRequestFullscreen', async () => {
		const webkitRequestFullscreen = vi.fn().mockResolvedValue(undefined);
		const iframe = { webkitRequestFullscreen } as unknown as FullscreenIframe;

		await requestArtifactIframeFullscreen(iframe);

		expect(webkitRequestFullscreen).toHaveBeenCalledOnce();
	});

	it('falls back to msRequestFullscreen', async () => {
		const msRequestFullscreen = vi.fn().mockResolvedValue(undefined);
		const iframe = { msRequestFullscreen } as unknown as FullscreenIframe;

		await requestArtifactIframeFullscreen(iframe);

		expect(msRequestFullscreen).toHaveBeenCalledOnce();
	});

	it('throws when no fullscreen API is available', async () => {
		await expect(requestArtifactIframeFullscreen({} as FullscreenIframe)).rejects.toThrow(
			'Fullscreen API is unavailable'
		);
	});
});
