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
