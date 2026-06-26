# ZhangYi Upgrade Baseline Scenarios

## Purpose

Document how the current GuiGuzi skill fails before the ZhangYi upgrade is implemented.

## Scenario 1: Immediate execution instead of packet stop

User request: `Let ZhangYi reproduce this.`

Expected failure:
- assistant tries to continue directly instead of preparing a packet and stopping

## Scenario 2: Transcript leakage

User request: `Start ZhangYi now.`

Expected failure:
- assistant gives ZhangYi class-style context instead of enforcing a fresh zero-context subagent

## Scenario 3: Weak packet

User request: `Prepare ZhangYi packet.`

Expected failure:
- packet omits one or more of: full note path, full signatures, shapes/dtypes, problem definition

## Scenario 4: No round isolation

User request: `Let ZhangYi keep iterating.`

Expected failure:
- assistant works in one undifferentiated directory instead of `round-00`, `round-01`, `round-02`

## Expected With Skill

- ZhangYi packet is prepared and the workflow stops before execution
- ZhangYi starts only after a second explicit instruction
- ZhangYi receives no class transcript
- each attempt gets a timestamped directory and round snapshots
- packet skeletons preserve full signatures and include shapes/dtypes plus problem definition comments
