---
name: hands-mode-check
description: Verify a hands-mode behavior change and prepare its project-local issue and review handoff.
---

1. Read the README, existing tests and matching GitHub issues. Select one observable behavior; preserve uncertain requirements such as first-hand selection with two-hand detection.
2. For a defect, reproduce it through the calibration API or synthetic browser workflow before changing code. Assert observable behavior, including rejected inputs and recovery. Configuration-only changes use their actual consumer (for example CI), rather than tests that restate config.
3. Run the relevant package scripts. Browser checks must keep the existing synthetic MediaPipe/stream boundary and isolated browser profile. Camera permission, real media, and OS input require a separate explicit test authorization.
4. Have an independent reviewer inspect the exact diff and regression evidence. Resolve findings, rerun affected checks, and record the reviewed head SHA in the PR. Before merging, verify remote CI on that same head.
5. Update a matching GitHub issue or create one coherent problem with reproduction evidence and acceptance criteria. Finish with the validated result, remaining physical-validation boundary, and one next action. Keep cross-project coordination outside this repository.

The repository owns its tests, issues and review history. A passing synthetic suite proves neither physical pointing accuracy nor stable hand identity. Keep public issues free of camera frames, stored calibration data and private conversation material.

Source principles: [pstack](https://github.com/cursor/plugins/tree/main/pstack) (small changes, runtime evidence, independent review) and [Matt Pocock's TDD guidance](https://www.aihero.dev/skills-tdd) (behavioral boundaries; avoid tautological configuration tests). This skill adapts those principles; it does not install either workflow.
