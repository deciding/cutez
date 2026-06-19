# GuiGuzi Kernel Deep Dive Skill Design

## Goal

Design a reusable assistant skill and companion repo workflow for interactive, natural-language-triggered kernel deep dives. The workflow should let `GuiGuzi` act as a teacher, `SuQin` act as an aggressive student, and the human user interrupt at any time, while grounding explanations in existing notes, the `cutedsl` repository, and optional `modal` execution.

## Scope

This design covers:
- A personal skill that activates from natural language requests such as `GuiGuzi debate`
- Preflight checks for required session inputs such as the `modal` command and `cutedsl` directory
- A role-driven teaching loop across `GuiGuzi`, `SuQin`, and the human user
- A repo-local note system that accumulates prior findings and is consulted before asking new questions
- Rules for when the teacher should answer from notes, inspect the repo, or run `modal`
- Stopping conditions and checkpoint behavior for long deep-dive sessions
- A test strategy for validating the skill behavior itself

This design does not yet cover:
- Implementing the final personal skill files in a user-level skills directory
- Building a dedicated CLI, TUI, or web app for the workflow
- Generalizing the workflow beyond kernel deep dives into arbitrary code tutoring domains

## Context

The current repository already uses repo-local specs and plans under `docs/superpowers/`, and has a `modal/` directory with runnable scripts. The user wants a natural-language workflow rather than a slash-command entrypoint. The workflow must therefore infer intent from user phrasing, inspect the current conversation context for required inputs, and ask only for missing prerequisites.

The user also wants notes to accumulate in the repository so future sessions can reuse previous findings. That means note lookup is not a side effect; it is part of the core reasoning loop. `SuQin` should first look for the most relevant existing notes before asking a new question, and only escalate to repo or runtime inspection when the note layer is insufficient.

## Proposed Architecture

### Recommended Split: Personal Skill + Repo-Local Companion Docs

The recommended design uses two artifacts:

1. A personal skill that enforces behavior
2. A repo-local documentation area that stores reusable session notes and repo-specific conventions

This split keeps the behavior reusable while allowing the `cutedsl` and `modal` specifics to evolve inside the repository.

### Why This Split Is Preferred

A repo-only document would describe the workflow, but would not strongly shape assistant behavior in future sessions. A single monolithic skill with all repo specifics embedded would be harder to reuse and would become stale when paths or note conventions change. The split keeps the trigger and dialogue discipline in the skill, and keeps project-specific memory in the repo.

## Trigger Model

The skill should activate from natural-language requests that strongly imply the teaching workflow. Example triggers include:
- `Let's do a GuiGuzi debate`
- `Use GuiGuzi to explain this kernel`
- `Start a kernel deep dive`
- `Let SuQin challenge this block`

The skill should not require a slash command. On activation, it should inspect the current conversation context for:
- a `modal` command
- a `cutedsl` directory path
- a target kernel, file, or code block to study

If any required input is missing, the skill should ask only for the missing items before starting the teaching loop.

## Required Inputs

The minimum required inputs are:
- `modal` command
- `cutedsl` directory
- target kernel file, function, or pasted code block

The skill should accept these inputs from either:
- the current conversation context
- the latest user message
- follow-up clarification questions if context is incomplete

If the target is a pasted code block rather than a file path, the session may still proceed, but repo and note lookup should use any names or symbols recoverable from the pasted block.

## Repo-Local Note System

### Location

Store notes under a repo-local folder such as:

`docs/kernel-deep-dives/`

Each topic or session should live in its own subdirectory, for example:

`docs/kernel-deep-dives/dense-gemm-7min/`

### File Shape

Each topic directory should contain one primary self-contained markdown file, plus optional supporting artifacts if needed. A recommended default is:

`docs/kernel-deep-dives/<topic>/<date>-notes.md`

The primary note file should be self-contained enough that a future session can recover:
- what kernel or block was studied
- key lines or blocks that mattered
- the current understanding of arguments and APIs
- concrete examples, shapes, or execution traces
- open questions and contradictions
- references back to repo files or commands

### Notes-First Rule

Before `SuQin` asks a new question, the workflow should first search for the most relevant existing notes. Those notes are the first-pass memory layer.

Only if notes are missing, stale, contradictory, or insufficient should the teacher escalate to:
1. repo inspection
2. `modal` execution

This rule should be explicit in the skill because it is a central user requirement.

## Role Responsibilities

### GuiGuzi

`GuiGuzi` is the teacher and synthesis layer. He should:
- explain the target top-down before dropping to line-level details
- answer using existing notes when they are sufficient
- inspect the `cutedsl` repo when note coverage is incomplete
- run the provided `modal` command only when runtime evidence is needed
- connect local lines to the bigger system picture
- state uncertainty clearly when notes, source, and runtime evidence still do not settle a question

`GuiGuzi` should be able to use `modal` outputs and debug prints to ask deeper follow-up questions, not just to confirm success/failure.

### SuQin

`SuQin` is the aggressive challenger. He should:
- ask what each important line or block is for
- ask why an API requires each argument
- question whether each argument is actually useful or redundant
- propose alternative formulations and ask about correctness or speed impact
- ask for the bigger picture around surrounding variants or missing counterparts
- ask for concrete examples such as actual shapes, layouts, or argument values
- write down key findings in the repo-local note system

`SuQin` should continue pushing until an important line or block is understood well enough to summarize crisply, then move to the next important block.

### Human User

The human user can interrupt at any time to:
- ask a parallel question
- challenge the teacher or `SuQin`
- redirect attention to a different block
- ask for a more concrete example
- request that the session stop or summarize

The skill should treat human interventions as first-class inputs, not as interruptions that break the workflow.

## Session Workflow

### Phase 1: Preflight

On activation, the workflow should:
- confirm whether the `modal` command is present
- confirm whether the `cutedsl` directory exists
- confirm whether the target kernel, file, or code block is identified

If the `modal` command is present, the workflow should verify that it looks runnable. It does not need to execute immediately.

If the `cutedsl` directory path is present, the workflow should verify that it exists before proceeding.

### Phase 2: Topic Setup

The workflow should determine the topic name from the kernel or file path, then identify the most relevant note directory and note files. If no prior note exists, it should prepare to create a new note file when the first stable finding is produced.

### Phase 3: Top-Down Orientation

`GuiGuzi` should begin with a concise top-down explanation of the target kernel or code region:
- what problem it solves
- where the target block sits in the larger flow
- what the major phases are

This gives `SuQin` a map before the detailed questioning begins.

### Phase 4: Question Loop

For each important block:
1. `SuQin` checks relevant notes first
2. `SuQin` asks the sharpest unresolved question
3. `GuiGuzi` answers from notes if possible
4. If needed, `GuiGuzi` inspects repo code
5. If needed, `GuiGuzi` runs `modal` for runtime evidence
6. `SuQin` challenges the answer until the block is understood
7. The workflow records the stable findings in notes
8. The session advances to the next important block

The loop should bias toward important lines or blocks, not literal every-line commentary unless the user specifically asks for that granularity.

### Phase 5: Checkpoints

The session should run autonomously within a block, but pause at clear checkpoints:
- arguments and inputs understood
- important line or block explained
- runtime verification needed
- notes updated
- moving to next block

This matches the user preference for autonomy with intervention points.

### Phase 6: Stop Conditions

The workflow should stop when one of the following is true:
- the requested target has been fully covered
- the user redirects or stops the session
- a blocking prerequisite is missing and the user has not supplied it
- runtime verification cannot proceed and the missing requirement is external

On stop, the workflow should summarize:
- what was understood
- what was verified by notes, source, and runtime
- what remains open
- where the notes were written

## Teacher Escalation Policy

The escalation order should be rigid:

1. Existing notes
2. `cutedsl` repo inspection
3. `modal` execution only if needed

`modal` should be used only when needed to verify behavior, inspect outputs, resolve semantic disagreement, or investigate performance-sensitive alternatives. It should not be the default for every question.

## Note Content Template

The default note file should use sections like:
- `# Topic`
- `## Target`
- `## Top-Down Model`
- `## Important Blocks`
- `## Argument Semantics`
- `## Concrete Examples`
- `## Runtime Evidence`
- `## Open Questions`
- `## References`

This structure is intentionally compact but sufficient to support later note-first retrieval.

## Error Handling

The workflow should handle common failure modes explicitly:

- Missing `modal` command: ask the user for it before session start
- Missing `cutedsl` directory: ask the user for it before session start
- Invalid `cutedsl` directory: report that the path does not exist and ask for another
- Missing target kernel: ask the user for the file, function, or pasted code block
- No relevant prior notes found: continue with repo inspection and create a new note file later
- `modal` command fails: capture the failure, use it as teaching evidence if helpful, and continue source-level reasoning where possible
- Notes contradict runtime behavior: mark the notes stale, prefer verified evidence, and update notes accordingly

## Testing Strategy

The eventual skill should be validated using the `writing-skills` TDD workflow rather than written and trusted immediately.

The initial RED baseline evidence for this design lives in `docs/kernel-deep-dives/skill-baseline-scenarios.md`.

### RED: Baseline Failure Scenarios

Before writing the final skill, run pressure scenarios without the skill and document failures such as:
- the assistant answers immediately without checking whether `modal` or `cutedsl` context exists
- the assistant forgets to search prior notes before asking or answering
- the assistant gives a shallow summary instead of sustaining the `GuiGuzi` and `SuQin` loop
- the assistant jumps to `modal` execution too early
- the assistant fails to checkpoint or update notes

### GREEN: Minimal Skill Behavior

Write the minimal skill that forces:
- natural-language trigger recognition
- preflight input checks
- notes-first retrieval
- teacher escalation order
- role-specific questioning and answering behavior
- checkpoint-based progression

### REFACTOR: Close Loopholes

Re-test with scenarios that pressure common rationalizations, for example:
- `the user already pasted code, so notes can be skipped`
- `the question is simple, so no need for the SuQin loop`
- `modal is available, so run it first`
- `existing notes are probably stale, so ignore them`

The skill should explicitly forbid these shortcuts unless the evidence truly requires escalation.

## Recommended Deliverables

The final implementation should produce:

1. A personal skill for `GuiGuzi` deep-dive tutoring behavior
2. A repo-local note directory such as `docs/kernel-deep-dives/`
3. A short companion repo document that explains note layout and topic naming
4. At least three pressure scenarios used to test the skill before trusting it

## Future Work

Later extensions could add:
- automatic note-topic matching heuristics
- a note index file across all deep dives
- reusable prompts for concrete shape generation
- optional branch or commit awareness so notes can mention which code revision they describe
