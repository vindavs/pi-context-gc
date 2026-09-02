import { afterEach, describe, expect, test } from "bun:test";
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type {
	ContextEvent,
	ExtensionAPI,
	ExtensionCommandContext,
	ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import contextGcExtension, {
	ContextGcArtifactStore,
	DEFAULT_MIN_RESULT_BYTES,
	findContextGcCandidates,
	findLastCompletedAssistantIndex,
	garbageCollectContext,
} from "../extensions/context-gc/index";

type AgentMessage = ContextEvent["messages"][number];

const EMPTY_USAGE = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function assistant(options: { completed?: boolean; toolCall?: boolean } = {}): AgentMessage {
	const content = options.toolCall
		? [{ type: "toolCall" as const, id: "next-call", name: "read", arguments: { path: "next.ts" } }]
		: [{ type: "text" as const, text: "Done" }];
	return {
		role: "assistant",
		content,
		api: "test",
		provider: "test",
		model: "test",
		usage: EMPTY_USAGE,
		stopReason: options.completed === false ? "toolUse" : "stop",
		timestamp: 1,
	} as AgentMessage;
}

function toolResult(
	text: string,
	options: { id?: string; tool?: string; isError?: boolean; image?: boolean } = {},
): AgentMessage {
	return {
		role: "toolResult",
		toolCallId: options.id ?? "call-1",
		toolName: options.tool ?? "bash",
		content: options.image
			? [{ type: "image", data: "abc", mimeType: "image/png" }]
			: [{ type: "text", text }],
		isError: options.isError ?? false,
		timestamp: 1,
	} as AgentMessage;
}

const roots: string[] = [];
afterEach(async () => {
	await Promise.all(roots.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

async function tempRoot(): Promise<string> {
	const path = await mkdtemp(join(tmpdir(), "context-gc-test-"));
	roots.push(path);
	return path;
}

async function pathExists(path: string): Promise<boolean> {
	try {
		await access(path);
		return true;
	} catch {
		return false;
	}
}

describe("context GC boundaries", () => {
	test("requires a completed assistant response", () => {
		const messages = [assistant({ completed: false, toolCall: true }), toolResult("x".repeat(100))];
		expect(findLastCompletedAssistantIndex(messages)).toBe(-1);
		expect(findContextGcCandidates(messages, 10)).toHaveLength(0);
	});

	test("does not collect results from the current work batch", () => {
		const oldResult = toolResult("old".repeat(100), { id: "old" });
		const currentResult = toolResult("current".repeat(100), { id: "current" });
		const messages = [
			oldResult,
			assistant(),
			{ role: "user", content: [{ type: "text", text: "Next task" }], timestamp: 2 } as AgentMessage,
			assistant({ completed: false, toolCall: true }),
			currentResult,
		];

		expect(findContextGcCandidates(messages, 10).map((candidate) => candidate.message.toolCallId)).toEqual([
			"old",
		]);
	});

	test("protects errors, images, and small results", () => {
		const messages = [
			toolResult("x".repeat(100), { id: "error", isError: true }),
			toolResult("x".repeat(100), { id: "image", image: true }),
			toolResult("tiny", { id: "small" }),
			assistant(),
		];
		expect(findContextGcCandidates(messages, 10)).toHaveLength(0);
	});
});

describe("context GC replacement", () => {
	test("preserves full output and changes only the outbound copy", async () => {
		const root = await tempRoot();
		const originalText = "important output\n".repeat(100);
		const original = toolResult(originalText);
		const messages = [original, assistant()];

		const result = await garbageCollectContext(messages, {
			artifactStore: new ContextGcArtifactStore(root),
			minResultBytes: 10,
		});

		expect(result.stats.prunedResults).toBe(1);
		expect(result.stats.netEstimatedTokens).toBeGreaterThan(0);
		expect(result.messages[0]).not.toBe(original);
		expect(messages[0]).toBe(original);
		expect((original as Extract<AgentMessage, { role: "toolResult" }>).content[0]).toEqual({
			type: "text",
			text: originalText,
		});

		const replacement = result.messages[0] as Extract<AgentMessage, { role: "toolResult" }>;
		const marker = replacement.content[0];
		expect(marker.type).toBe("text");
		if (marker.type !== "text") throw new Error("Expected text marker");
		expect(marker.text).toContain("Context GC removed a consumed bash result");
		expect(marker.text).toContain("Full output:");
		expect(await readFile(result.stats.artifacts[0], "utf8")).toBe(originalText);
	});

	test("uses a stable artifact and can repeat future context builds", async () => {
		const root = await tempRoot();
		const messages = [toolResult("repeat".repeat(100)), assistant()];

		const first = await garbageCollectContext(messages, {
			artifactStore: new ContextGcArtifactStore(root),
			minResultBytes: 10,
		});
		const second = await garbageCollectContext(messages, {
			artifactStore: new ContextGcArtifactStore(root),
			minResultBytes: 10,
		});

		expect(second.stats.failures).toEqual([]);
		expect(second.stats.artifacts).toEqual(first.stats.artifacts);
		expect(second.messages).toEqual(first.messages);
	});

	test("keeps the original context when artifact preservation fails", async () => {
		const root = await tempRoot();
		const unusableRoot = join(root, "not-a-directory");
		await writeFile(unusableRoot, "occupied");
		const original = toolResult("cannot discard".repeat(100));
		const messages = [original, assistant()];

		const result = await garbageCollectContext(messages, {
			artifactStore: new ContextGcArtifactStore(unusableRoot),
			minResultBytes: 10,
		});

		expect(result.stats.prunedResults).toBe(0);
		expect(result.stats.failures).toHaveLength(1);
		expect(result.messages[0]).toBe(original);
	});
});

describe("context GC artifact store", () => {
	test("preserves each artifact only once per runtime", async () => {
		const root = await tempRoot();
		let attempts = 0;
		const store = new ContextGcArtifactStore(root, async () => {
			attempts++;
		});
		const messages = [toolResult("repeat".repeat(100)), assistant()];

		await garbageCollectContext(messages, { artifactStore: store, minResultBytes: 10 });
		await garbageCollectContext(messages, { artifactStore: store, minResultBytes: 10 });

		expect(attempts).toBe(1);
		expect(store.preservedCount).toBe(1);
	});

	test("retries preservation failures", async () => {
		const root = await tempRoot();
		let attempts = 0;
		const store = new ContextGcArtifactStore(root, async () => {
			attempts++;
			if (attempts === 1) throw new Error("temporary failure");
		});
		const messages = [toolResult("retry".repeat(100)), assistant()];

		const first = await garbageCollectContext(messages, { artifactStore: store, minResultBytes: 10 });
		const second = await garbageCollectContext(messages, { artifactStore: store, minResultBytes: 10 });

		expect(first.stats.prunedResults).toBe(0);
		expect(first.stats.failures).toHaveLength(1);
		expect(second.stats.prunedResults).toBe(1);
		expect(attempts).toBe(2);
		expect(store.preservedCount).toBe(1);
	});

	test("clear removes artifacts and resets runtime tracking", async () => {
		const root = await tempRoot();
		const store = new ContextGcArtifactStore(root);
		const messages = [toolResult("clear".repeat(100)), assistant()];
		await garbageCollectContext(messages, { artifactStore: store, minResultBytes: 10 });

		expect(await pathExists(root)).toBe(true);
		expect(store.preservedCount).toBe(1);
		await store.clear();
		expect(await pathExists(root)).toBe(false);
		expect(store.preservedCount).toBe(0);
	});
});

type TestEventHandler = (event: { type: string; [key: string]: unknown }, ctx: ExtensionContext) => unknown;
type TestCommand = {
	handler: (args: string, ctx: ExtensionCommandContext) => Promise<void> | void;
};

function createExtensionHarness(sessionFile: string, sessionId = "session-1") {
	const handlers = new Map<string, TestEventHandler[]>();
	const commands = new Map<string, TestCommand>();
	const notifications: Array<{ message: string; level: string }> = [];
	let idle = true;

	const ctx = {
		sessionManager: {
			getSessionId: () => sessionId,
			getSessionFile: () => sessionFile,
		},
		isIdle: () => idle,
		ui: {
			notify: (message: string, level: string) => notifications.push({ message, level }),
		},
	} as unknown as ExtensionCommandContext;

	const pi = {
		on: (event: string, handler: TestEventHandler) => {
			const eventHandlers = handlers.get(event) ?? [];
			eventHandlers.push(handler);
			handlers.set(event, eventHandlers);
		},
		registerCommand: (name: string, command: TestCommand) => commands.set(name, command),
	} as unknown as ExtensionAPI;

	contextGcExtension(pi);

	return {
		ctx,
		notifications,
		setIdle(value: boolean) {
			idle = value;
		},
		async runEvent(type: string, event: { type: string; [key: string]: unknown } = { type }) {
			let result: unknown;
			for (const handler of handlers.get(type) ?? []) result = await handler(event, ctx);
			return result;
		},
		async command(args: string) {
			const command = commands.get("context-gc");
			if (!command) throw new Error("context-gc command was not registered");
			await command.handler(args, ctx);
		},
	};
}

describe("context GC extension lifecycle", () => {
	test("toggles collection and reports runtime status", async () => {
		const sessionDir = await tempRoot();
		const harness = createExtensionHarness(join(sessionDir, "session.jsonl"));
		const messages = [toolResult("x".repeat(DEFAULT_MIN_RESULT_BYTES)), assistant()];

		await harness.command("off");
		expect(await harness.runEvent("context", { type: "context", messages })).toBeUndefined();
		await harness.command("on");
		const transformed = (await harness.runEvent("context", {
			type: "context",
			messages,
		})) as { messages: AgentMessage[] };
		expect(transformed.messages[0]).not.toBe(messages[0]);
		await harness.command("status");
		expect(harness.notifications.at(-1)?.message).toContain("Context GC: on");
		expect(harness.notifications.at(-1)?.message).toContain("Runtime artifacts: 1");
	});

	test("cleans only the current session on start and shutdown", async () => {
		const sessionDir = await tempRoot();
		const sessionId = "current-session";
		const ownRoot = join(sessionDir, "context-gc", sessionId);
		const siblingRoot = join(sessionDir, "context-gc", "another-session");
		await mkdir(ownRoot, { recursive: true });
		await mkdir(siblingRoot, { recursive: true });
		await writeFile(join(ownRoot, "stale.txt"), "stale");
		await writeFile(join(siblingRoot, "keep.txt"), "keep");
		const harness = createExtensionHarness(join(sessionDir, "session.jsonl"), sessionId);

		await harness.runEvent("session_start");
		expect(await pathExists(ownRoot)).toBe(false);
		expect(await pathExists(siblingRoot)).toBe(true);

		const messages = [toolResult("x".repeat(DEFAULT_MIN_RESULT_BYTES)), assistant()];
		await harness.runEvent("context", { type: "context", messages });
		expect(await pathExists(ownRoot)).toBe(true);
		await harness.runEvent("session_shutdown");
		expect(await pathExists(ownRoot)).toBe(false);
		expect(await pathExists(siblingRoot)).toBe(true);
	});

	test("clean is idle-only and artifacts are recreated on demand", async () => {
		const sessionDir = await tempRoot();
		const sessionId = "clean-session";
		const ownRoot = join(sessionDir, "context-gc", sessionId);
		const harness = createExtensionHarness(join(sessionDir, "session.jsonl"), sessionId);
		const messages = [toolResult("x".repeat(DEFAULT_MIN_RESULT_BYTES)), assistant()];
		await harness.runEvent("context", { type: "context", messages });
		expect(await pathExists(ownRoot)).toBe(true);

		harness.setIdle(false);
		await harness.command("clean");
		expect(await pathExists(ownRoot)).toBe(true);
		expect(harness.notifications.at(-1)?.level).toBe("warning");

		harness.setIdle(true);
		await harness.command("clean");
		expect(await pathExists(ownRoot)).toBe(false);
		expect(harness.notifications.at(-1)?.message).toContain("artifacts cleared");

		await harness.runEvent("context", { type: "context", messages });
		expect(await pathExists(ownRoot)).toBe(true);
	});
});
