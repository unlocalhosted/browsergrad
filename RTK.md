# Repo Task Kit

This is the short operational runbook for future agents.

## First Five Minutes

1. Run `git status --short --branch`.
2. Read `AGENTS.md`, `AGENTS-MAP.md`, and the package README for the area you are touching.
3. Search with `rg` before editing.
4. Identify whether the change touches editable source, generated source, or published `dist/`.
5. Choose the smallest test command that exercises the change.

## Editing Rules

- Use `apply_patch` for manual file edits.
- Avoid unrelated refactors.
- Do not revert user changes.
- Keep Python package generated files synchronized through codegen.
- Keep public errors descriptive and explicit when functionality is unsupported.

## Pyodide Probe Pattern

When checking installed Python behavior from Node, run from a package directory that can resolve `pyodide`, for example:

```sh
cd packages/browsergrad-jit
node --input-type=module
```

Then import package dist installers and node adapters:

```js
import { loadPyodide } from "pyodide";
import { installJit } from "./dist/index.js";
import { createNodePyodideTarget } from "./dist/node-adapter.js";
```

For `browsergrad-grad` from the JIT directory, use sibling imports:

```js
import { installGrad } from "../browsergrad-grad/dist/index.js";
import { createNodePyodideTarget as createGradTarget } from "../browsergrad-grad/dist/node-adapter.js";
```

## Curriculum Compatibility Work

Before changing package APIs for a course/lab:

- Check whether need is reusable across curricula.
- Put lab-specific facts in `docs/internal/` or lab manifests.
- Add package APIs only when they match BrowserGrad's general PyTorch-shaped/browser-safe contract.
- Add focused package tests for reusable APIs.
- Keep native-only upstream harness assumptions out of root package code.
