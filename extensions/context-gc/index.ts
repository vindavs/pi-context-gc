import { createHash } from "node:crypto";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import type { ContextEvent, ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

type AgentMessage = ContextEvent["messages"][number];
type ToolResultMessage = Extract<AgentMessage, { role: "toolResult" }>;

export const DEFAULT_MIN_RESULT_BYTES = 16 * 1024;

export interface ContextGcCandidate {
	index: number;
	message: ToolResultMessage;
	text: string;
	bytes: number;
	estimatedTokens: number;
}

export interface ContextGcStats {
	boundaryFound: boolean;
	eligibleResults: number;
	prunedResults: number;
	originalBytes: number;
	originalEstimatedTokens: number;
	markerEstimatedTokens: number;
	netEstimatedTokens: number;
	artifacts: string[];
	byTool: Record<string, { results: number; estimatedTokens: number }>;
	failures: string[];
}

export interface ContextGcResult {
	messages: AgentMessage[];
	stats: ContextGcStats;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

function hasToolCall(message: AgentMessage): boolean {
	if (message.role !== "assistant" || !Array.isArray(message.content)) return false;
	return message.content.some((block) => isRecord(block) && block.type === "toolCall");
}

/** A completed text response marks the end of a meaningful work batch. */
export function findLastCompletedAssistantIndex(messages: readonly AgentMessage[]): number {
	for (let index = messages.length - 1; index >= 0; index--) {
		const message = messages[index];
		if (message.role !== "assistant") continue;
		if (message.stopReason === "stop" && !hasToolCall(message)) return index;
	}
	return -1;
}

function extractTextResult(message: ToolResultMessage): string | undefined {
	if (!Array.isArray(message.content) || message.content.length === 0) return undefined;

	const textBlocks: string[] = [];
	for (const block of message.content) {
		if (!isRecord(block) || block.type !== "text" || typeof block.text !== "string") {
			// Images and unknown blocks need their original provider representation.
			return undefined;
		}
		textBlocks.push(block.text);
	}
	return textBlocks.join("\n\n");
}

/** Plan GC without changing messages. Results from the current work batch are protected. */
export function findContextGcCandidates(
	messages: readonly AgentMessage[],
	minResultBytes = DEFAULT_MIN_RESULT_BYTES,
): ContextGcCandidate[] {
	const boundary = findLastCompletedAssistantIndex(messages);
	if (boundary < 0) return [];

	const candidates: ContextGcCandidate[] = [];
	for (let index = 0; index < boundary; index++) {
		const message = messages[index];
		if (message.role !== "toolResult" || message.isError) continue;

		const text = extractTextResult(message);
		if (text === undefined) continue;
		const bytes = Buffer.byteLength(text, "utf8");
		if (bytes < minResultBytes) continue;

		candidates.push({
			index,
			message,
			text,
			bytes,
			estimatedTokens: Math.ceil(text.length / 4),
		});
	}
	return candidates;
}

function safeName(value: string): string {
	const normalized = value.replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
	return normalized.slice(0, 48) || "tool";
}

function artifactPath(root: string, candidate: ContextGcCandidate): string {
	const digest = createHash("sha256").update(candidate.message.toolCallId).digest("hex").slice(0, 16);
	return join(root, `${safeName(candidate.message.toolName)}-${digest}.txt`);
}

export type ContextGcArtifactWriter = (path: string, text: string) => Promise<void>;

async function preserveOutput(path: string, text: string): Promise<void> {
	await mkdir(dirname(path), { recursive: true, mode: 0o700 });
	try {
		await writeFile(path, text, { encoding: "utf8", flag: "wx", mode: 0o600 });
	} catch (error) {
		if (isRecord(error) && error.code === "EEXIST") return;
		throw error;
	}
}

export class ContextGcArtifactStore {
	readonly root: string;
	private readonly writer: ContextGcArtifactWriter;
	private readonly preservedPaths = new Set<string>();

	constructor(root: string, writer: ContextGcArtifactWriter = preserveOutput) {
		this.root = root;
		this.writer = writer;
	}

	get preservedCount(): number {
		return this.preservedPaths.size;
	}

	async preserve(candidate: ContextGcCandidate): Promise<string> {
		const path = artifactPath(this.root, candidate);
		if (this.preservedPaths.has(path)) return path;

		await this.writer(path, candidate.text);
		this.preservedPaths.add(path);
		return path;
	}

	async clear(): Promise<void> {
		try {
			await rm(this.root, { recursive: true, force: true });
		} finally {
			this.preservedPaths.clear();
		}
	}
}

function formatBytes(bytes: number): string {
	if (bytes < 1024) return `${bytes} B`;
	if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
	return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTokens(tokens: number): string {
	if (tokens < 1000) return String(tokens);
	if (tokens < 10_000) return `${(tokens / 1000).toFixed(1)}k`;
	return `${Math.round(tokens / 1000)}k`;
}

function buildMarker(candidate: ContextGcCandidate, path: string): string {
	return [
		`[Context GC removed a consumed ${candidate.message.toolName} result from future requests.]`,
		`Original size: ${formatBytes(candidate.bytes)} (~${formatTokens(candidate.estimatedTokens)} estimated tokens).`,
		`Full output: ${path}`,
		"Use read with offset/limit if the exact output is needed again.",
	].join("\n");
}

function emptyStats(boundaryFound: boolean, eligibleResults: number): ContextGcStats {
	return {
		boundaryFound,
		eligibleResults,
		prunedResults: 0,
		originalBytes: 0,
		originalEstimatedTokens: 0,
		markerEstimatedTokens: 0,
		netEstimatedTokens: 0,
		artifacts: [],
		byTool: {},
		failures: [],
	};
}

/** Preserve eligible output, then replace only the outbound context copy with recovery markers. */
export async function garbageCollectContext(
	messages: readonly AgentMessage[],
	options: { artifactStore: ContextGcArtifactStore; minResultBytes?: number },
): Promise<ContextGcResult> {
	const boundary = findLastCompletedAssistantIndex(messages);
	const candidates = findContextGcCandidates(messages, options.minResultBytes);
	const stats = emptyStats(boundary >= 0, candidates.length);
	if (candidates.length === 0) return { messages: [...messages], stats };

	const nextMessages = [...messages];
	for (const candidate of candidates) {
		let path: string;
		try {
			path = await options.artifactStore.preserve(candidate);
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			stats.failures.push(`${candidate.message.toolName}: ${message}`);
			continue;
		}

		const marker = buildMarker(candidate, path);
		nextMessages[candidate.index] = {
			...candidate.message,
			content: [{ type: "text", text: marker }],
		};

		const markerTokens = Math.ceil(marker.length / 4);
		stats.prunedResults++;
		stats.originalBytes += candidate.bytes;
		stats.originalEstimatedTokens += candidate.estimatedTokens;
		stats.markerEstimatedTokens += markerTokens;
		stats.artifacts.push(path);
		const toolStats = stats.byTool[candidate.message.toolName] ?? { results: 0, estimatedTokens: 0 };
		toolStats.results++;
		toolStats.estimatedTokens += candidate.estimatedTokens - markerTokens;
		stats.byTool[candidate.message.toolName] = toolStats;
	}
	stats.netEstimatedTokens = stats.originalEstimatedTokens - stats.markerEstimatedTokens;

	return { messages: nextMessages, stats };
}

function resolveArtifactRoot(ctx: ExtensionContext): string {
	const sessionId = safeName(ctx.sessionManager.getSessionId());
	const sessionFile = ctx.sessionManager.getSessionFile();
	if (sessionFile) return join(dirname(sessionFile), "context-gc", sessionId);
	return join(tmpdir(), "pi-context-gc", sessionId);
}

function formatStatus(
	enabled: boolean,
	stats: ContextGcStats | undefined,
	preservedArtifacts: number,
): string {
	const lines = [
		`Context GC: ${enabled ? "on" : "off"}`,
		`Policy: results >= ${formatBytes(DEFAULT_MIN_RESULT_BYTES)}; prune between completed work batches`,
		"Protected: current batch, errors, images",
		`Runtime artifacts: ${preservedArtifacts}`,
	];
	if (!stats) {
		lines.push("No outbound context build observed since load.");
		return lines.join("\n");
	}

	lines.push(
		`Last build: ${stats.prunedResults} result(s), ~${formatTokens(stats.netEstimatedTokens)} tokens removed`,
	);
	const tools = Object.entries(stats.byTool).sort((a, b) => b[1].estimatedTokens - a[1].estimatedTokens);
	if (tools.length > 0) {
		lines.push(`By tool: ${tools.map(([name, value]) => `${name} ${formatTokens(value.estimatedTokens)}`).join(", ")}`);
	}
	if (stats.failures.length > 0) lines.push(`Preservation failures: ${stats.failures.length}`);
	return lines.join("\n");
}

export default function contextGcExtension(pi: ExtensionAPI) {
	let enabled = true;
	let lastStats: ContextGcStats | undefined;
	let artifactStore: ContextGcArtifactStore | undefined;

	function getArtifactStore(ctx: ExtensionContext): ContextGcArtifactStore {
		const root = resolveArtifactRoot(ctx);
		if (!artifactStore || artifactStore.root !== root) artifactStore = new ContextGcArtifactStore(root);
		return artifactStore;
	}

	async function clearCurrentArtifacts(ctx: ExtensionContext): Promise<void> {
		const store = getArtifactStore(ctx);
		await store.clear();
		lastStats = undefined;
	}

	pi.on("session_start", async (_event, ctx) => {
		try {
			await clearCurrentArtifacts(ctx);
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			ctx.ui.notify(`Context GC could not clear stale artifacts: ${message}`, "warning");
		}
	});

	pi.on("session_shutdown", async (_event, ctx) => {
		try {
			await clearCurrentArtifacts(ctx);
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			ctx.ui.notify(`Context GC could not clear artifacts: ${message}`, "warning");
		}
	});

	pi.on("context", async (event, ctx) => {
		if (!enabled) return;
		const result = await garbageCollectContext(event.messages, {
			artifactStore: getArtifactStore(ctx),
		});
		lastStats = result.stats;
		if (result.stats.prunedResults === 0) return;
		return { messages: result.messages };
	});

	pi.registerCommand("context-gc", {
		description: "Show, toggle, or clean consumed large-tool-result garbage collection",
		handler: async (args, ctx) => {
			const action = args.trim().toLowerCase();
			if (action === "on") enabled = true;
			else if (action === "off") enabled = false;
			else if (action === "clean") {
				if (!ctx.isIdle()) {
					ctx.ui.notify("Context GC artifacts can only be cleaned while the agent is idle.", "warning");
					return;
				}
				try {
					await clearCurrentArtifacts(ctx);
					ctx.ui.notify("Context GC artifacts cleared; needed outputs will be recreated automatically.", "info");
				} catch (error) {
					const message = error instanceof Error ? error.message : String(error);
					ctx.ui.notify(`Context GC could not clear artifacts: ${message}`, "warning");
				}
				return;
			} else if (action !== "" && action !== "status") {
				ctx.ui.notify("Usage: /context-gc [status|on|off|clean]", "warning");
				return;
			}
			ctx.ui.notify(formatStatus(enabled, lastStats, artifactStore?.preservedCount ?? 0), "info");
		},
	});
}
