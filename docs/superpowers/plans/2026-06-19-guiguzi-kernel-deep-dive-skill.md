# GuiGuzi Kernel Deep Dive Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable local skill plus repo-local companion docs for natural-language-triggered kernel deep dives with GuiGuzi, SuQin, notes-first retrieval, and modal-backed verification.

**Architecture:** The implementation is split between a personal skill in the local OpenCode skills directory and repo-local documentation under `docs/kernel-deep-dives/`. The skill enforces trigger recognition, preflight checks, note-first behavior, role loop, and escalation order. The repo docs provide note layout, a reusable note template, and baseline testing evidence for future iteration.

**Tech Stack:** Markdown, OpenCode local skills, repo-local documentation, subagent-based pressure testing

---

### Task 1: Capture RED Baseline and Repo Conventions

**Files:**
- Create: `docs/kernel-deep-dives/skill-baseline-scenarios.md`
- Modify: `docs/superpowers/specs/2026-06-19-guiguzi-kernel-deep-dive-skill-design.md`
- Test: Manual review against captured subagent outputs

- [ ] **Step 1: Write the failing baseline evidence document**

Create `docs/kernel-deep-dives/skill-baseline-scenarios.md` with sections for three scenarios:

```md
# GuiGuzi Skill Baseline Scenarios

## Purpose

Document how the assistant behaves without a dedicated GuiGuzi deep-dive skill so the final skill addresses real failures.

## Scenario 1: Trigger without prerequisites

User request: `Let's do a GuiGuzi debate on this cutedsl kernel.`

Observed behavior:
- asks for the target kernel
- does not require modal command or cutedsl path up front
- does not establish role loop or notes-first policy

## Scenario 2: Pasted code without repo context

User request: `Use GuiGuzi to explain this kernel` plus pasted code.

Observed behavior:
- offers a high-level explanation path
- does not ask for modal command or cutedsl repo path
- does not search prior notes first

## Scenario 3: Challenge request with prerequisites present

User request: `Let SuQin challenge this block until fully understood.`

Observed behavior:
- asks for the target block
- does not autonomously run the GuiGuzi/SuQin loop
- does not mention checkpoints or note updates
```

- [ ] **Step 2: Verify the baseline document reflects actual failures**

Run a manual comparison against the captured subagent outputs from this session.

Expected:
- each scenario lists at least one concrete missing behavior
- the missing behaviors align with the approved spec

- [ ] **Step 3: Add a short pointer from the spec to the baseline file**

Append this sentence under `## Testing Strategy` in `docs/superpowers/specs/2026-06-19-guiguzi-kernel-deep-dive-skill-design.md`:

```md
The initial RED baseline evidence for this design lives in `docs/kernel-deep-dives/skill-baseline-scenarios.md`.
```

- [ ] **Step 4: Re-read both files**

Run: `rg -n "baseline|Scenario|RED baseline evidence" docs/kernel-deep-dives/skill-baseline-scenarios.md docs/superpowers/specs/2026-06-19-guiguzi-kernel-deep-dive-skill-design.md`

Expected: matches for all three scenarios and the spec pointer.

### Task 2: Create Repo-Local Companion Docs and Note Template

**Files:**
- Create: `docs/kernel-deep-dives/README.md`
- Create: `docs/kernel-deep-dives/_template.md`
- Test: Manual review of note layout and retrieval guidance

- [ ] **Step 1: Write the companion README**

Create `docs/kernel-deep-dives/README.md` with:

```md
# Kernel Deep Dives

This directory stores reusable notes from GuiGuzi-style kernel teaching sessions.

## Rules

1. Search the most relevant existing notes before asking a new question.
2. Prefer updating an existing topic directory when the target kernel already has notes.
3. Record stable findings, concrete shapes, argument semantics, and runtime evidence.
4. Mark contradictions explicitly instead of silently overwriting them.

## Layout

- `docs/kernel-deep-dives/<topic>/`
- primary note file: `<date>-notes.md`

## Suggested Topic Names

- kernel file stem, for example `dense-gemm-7min`
- function or class name if the investigation is narrower

## Retrieval Order

1. Most relevant topic directory
2. Most recent note file in that topic
3. Repo source inspection
4. Modal execution only if needed
```

- [ ] **Step 2: Write the note template**

Create `docs/kernel-deep-dives/_template.md` with:

```md
# Topic

## Target

- Kernel or block:
- Repo path:
- Modal command:
- Cutedsl directory:

## Top-Down Model

## Important Blocks

## Argument Semantics

## Concrete Examples

## Runtime Evidence

## Open Questions

## References
```

- [ ] **Step 3: Review the docs for consistency with the spec**

Check that both files use the same note-first and escalation order described in the spec.

Expected:
- README mentions notes-first retrieval
- template contains target, examples, runtime evidence, and references sections

### Task 3: Implement the Personal Skill Draft

**Files:**
- Create: `/home/zining/.config/opencode/skills/guiguzi-kernel-deep-dive/SKILL.md`
- Test: Manual inspection of skill triggers, loop, and anti-shortcut guidance

- [ ] **Step 1: Write the skill frontmatter and overview**

Create `/home/zining/.config/opencode/skills/guiguzi-kernel-deep-dive/SKILL.md` starting with:

```md
---
name: guiguzi-kernel-deep-dive
description: Use when the user asks for a GuiGuzi debate, kernel deep dive, SuQin challenge, or interactive tutoring on a cutedsl kernel with repo notes and optional modal verification.
---

# GuiGuzi Kernel Deep Dive

## Overview

Run an interactive kernel tutoring loop with three roles: GuiGuzi the teacher, SuQin the aggressive student, and the human user as a first-class challenger. Always enforce notes-first retrieval, then repo inspection, then modal execution only if needed.
```

- [ ] **Step 2: Write the when-to-use and preflight sections**

Add sections that explicitly require:

```md
## When to Use

- User asks for `GuiGuzi debate`, `kernel deep dive`, `SuQin challenge`, or equivalent natural-language tutoring.
- The target is a cutedsl kernel, code block, file, or function that needs deep explanation.

Do not use for ordinary code review, implementation planning, or broad repo exploration without a tutoring request.

## Preflight

Before starting the role loop, confirm:
1. target kernel, file, function, or pasted block
2. modal command
3. cutedsl directory

If any item is missing, ask only for the missing items.
If a modal command is present, verify that it looks runnable.
If a cutedsl directory is present, verify that it exists.
```

- [ ] **Step 3: Write the role loop, escalation order, and checkpoints**

Add sections that explicitly require:

```md
## Core Loop

1. GuiGuzi gives a top-down model of the target.
2. SuQin checks the most relevant prior notes before asking a new question.
3. SuQin asks the sharpest unresolved question about an important line or block.
4. GuiGuzi answers from notes if possible.
5. If notes are insufficient, inspect the repo.
6. If source is still insufficient or disputed, run modal.
7. Record stable findings in repo-local notes.
8. Pause at a checkpoint before moving on.

## Checkpoints

- arguments and inputs understood
- important line or block explained
- runtime verification needed
- notes updated
- moving to next block
```

- [ ] **Step 4: Add anti-shortcut rules and note location**

Add sections that explicitly forbid common failures:

```md
## Red Flags

- answering before checking for existing notes
- skipping the preflight because code was pasted
- running modal first because it is available
- giving a shallow explanation without sustaining the GuiGuzi/SuQin loop
- moving on without recording stable findings

## Repo Notes

Use `docs/kernel-deep-dives/` as the default note root when working in this repo.
Prefer the most relevant existing topic directory before creating a new one.
```

- [ ] **Step 5: Review the skill for searchability and consistency**

Check that the skill mentions:
- `GuiGuzi debate`
- `SuQin`
- `kernel deep dive`
- `cutedsl`
- `modal`
- `docs/kernel-deep-dives`

Expected: the skill is discoverable and aligned with the approved spec.

### Task 4: GREEN/REFACTOR Validation Pass

**Files:**
- Modify: `/home/zining/.config/opencode/skills/guiguzi-kernel-deep-dive/SKILL.md`
- Modify: `docs/kernel-deep-dives/skill-baseline-scenarios.md`
- Test: Manual and subagent spot checks

- [ ] **Step 1: Re-run the baseline scenarios mentally against the drafted skill**

Check that the drafted skill would now force:
- preflight questions for missing modal command or cutedsl path
- notes-first lookup
- autonomous role loop with checkpoints

- [ ] **Step 2: Record the expected GREEN behavior**

Append a `## Expected With Skill` section to `docs/kernel-deep-dives/skill-baseline-scenarios.md`:

```md
## Expected With Skill

- missing prerequisites are requested explicitly before the debate starts
- prior notes are consulted before new questioning
- GuiGuzi gives a top-down model first
- SuQin keeps challenging an important block until stable understanding is recorded
- modal is used only when notes and source are insufficient
```

- [ ] **Step 3: Tighten loopholes if the skill still leaves room for shortcuts**

Update `SKILL.md` if any section still allows:
- skipping notes because the question seems simple
- ignoring checkpoints
- failing to create or update notes after stable findings

- [ ] **Step 4: Final review**

Run: `rg -n "GuiGuzi|SuQin|modal|cutedsl|docs/kernel-deep-dives|Red Flags|Checkpoints" /home/zining/.config/opencode/skills/guiguzi-kernel-deep-dive/SKILL.md docs/kernel-deep-dives/README.md docs/kernel-deep-dives/skill-baseline-scenarios.md`

Expected: matches across the skill and repo-local docs.
