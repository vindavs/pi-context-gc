# pi-context-gc

A [Pi](https://github.com/earendil-works/pi-mono) extension that reduces future parent-context occupancy by replacing consumed large text tool results with recovery markers.

## Install

Install the tagged private Git package through SSH:

```bash
pi install git:git@github.com:vindavs/pi-context-gc.git@v0.1.0
```

Then run `/reload` in Pi.

## Behavior

- Results remain unchanged during the work batch that produced them.
- After a successful text-only assistant response completes the batch, later LLM requests replace each eligible result with a short marker.
- Original session entries and TUI history are not modified.
- Full text is copied to a session-scoped artifact before the outbound context is changed.
- Artifacts are temporary runtime copies and are removed when the session starts or shuts down cleanly.
- Results under 16 KB, errors, images, and the current work batch are protected.

Pruning happens only between completed batches to avoid repeatedly changing the provider's cached prefix during a tool loop. The first request after a prune may lose cache reuse from the earliest changed result onward; subsequent requests reuse the stable shortened context.

Artifacts are stored beside the session file:

```text
<session-directory>/context-gc/<session-id>/<tool>-<digest>.txt
```

Ephemeral sessions use the system temporary directory. Artifact files use owner-only permissions. The original output remains in session history; cleaning artifacts does not alter the session.

A crash can leave artifacts behind. Loading that session again removes its stale artifact directory before rebuilding any needed outputs. Cleanup is confined to the current session so concurrent Pi sessions are not disturbed.

## Commands

```text
/context-gc          # status
/context-gc status
/context-gc off      # disable until reload or session replacement
/context-gc on
/context-gc clean    # remove current-session artifacts while idle
```

Status reports estimated parent-context tokens removed from the latest outbound context build and the number of artifacts cached by the current runtime. It does not include metered nested work such as subagent usage.

`/context-gc clean` only affects the current session and leaves GC enabled. Outputs are recreated automatically on the next context build if they are still eligible.

## Development

```bash
npm install
npm test
npm run typecheck
```
