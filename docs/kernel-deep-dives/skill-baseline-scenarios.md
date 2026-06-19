# GuiGuzi Skill Baseline Scenarios

## Purpose

Document how the assistant behaves without a dedicated GuiGuzi deep-dive skill so the final skill addresses real failures instead of guessed ones.

## Scenario 1: Trigger without prerequisites

User request: `Let's do a GuiGuzi debate on this cutedsl kernel.`

Observed behavior:
- asks for the target kernel
- does not require the `modal` command up front
- does not require the `cutedsl` path up front
- does not establish the `GuiGuzi` and `SuQin` role loop
- does not mention repo-local note reuse before questioning

Observed response:

```text
To do a useful GuiGuzi debate, I need the exact kernel in scope.

Which one should we dissect?
1. Paste the `cutedsl` kernel here
2. Give me the file path
3. Point me to the function name if it’s already in this repo

Once I have the target, I’ll argue both sides: what the kernel is doing well, where the design is brittle, and what I’d change first.
```

## Scenario 2: Pasted code without repo context

User request: `Use GuiGuzi to explain this kernel` plus pasted code.

Observed behavior:
- offers a high-level explanation path
- does not ask for the `modal` command
- does not ask for the `cutedsl` repo path
- does not search prior notes first
- does not define checkpoint behavior

Observed response:

```text
Paste the kernel snippet and I’ll explain it in a GuiGuzi style: first the kernel’s objective, then the “moves” it uses around indexing, memory movement, synchronization, and tiling, and finally the likely performance intent behind each part. If there’s a specific part you want decoded first, point to the lines.
```

## Scenario 3: Challenge request with prerequisites present

User request: `Let SuQin challenge this block until fully understood.`

Observed behavior:
- asks for the target block
- does not autonomously run the `GuiGuzi` and `SuQin` loop
- does not mention note updates
- does not mention the checkpoint model

Observed response:

```text
Paste the block you want challenged.

I’ll go line by line, question assumptions, force each symbol and transformation to justify itself, and keep pushing until the intent, mechanics, and edge cases are all explicit.
```

## Expected With Skill

- missing prerequisites are requested explicitly before the debate starts
- prior notes are consulted before new questioning
- `GuiGuzi` gives a top-down model first
- `SuQin` keeps challenging an important block until stable understanding is recorded
- `modal` is used only when notes and source are insufficient
- checkpoints are used before moving to the next important block
