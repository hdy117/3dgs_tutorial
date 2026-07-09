# Documentation Language And Structure Contract

All durable Markdown documentation produced by this harness must be written in **Chinese prose**. Keep professional technical terms in English when that is the common engineering form, such as `API`, `CLI`, `schema`, `data contract`, `pipeline`, `module`, `architecture`, `interface`, `tensor`, `dtype`, `checkpoint`, `runtime`, `deployment`, `DAG`, `cache`, `adapter`, and `service`.

## Language

- Use Chinese for explanations, section prose, acceptance criteria explanations, risks, decisions, and handoff notes.
- Preserve English technical terms instead of forcing awkward translations.
- Do not produce mostly-English durable docs unless the user explicitly requested English-only docs.

## Required Sections

- Every PM feature breakdown and architect design document must include a section named **`设计思想`** that explains:
  - the module/software goal
  - core abstraction
  - responsibility split
  - dependency direction
  - why this design is chosen
- Every substantive design document must include either **`总体流程图`** or **`总体架构图`** near the beginning.
  - Use Mermaid by default.
  - Use `flowchart` for module/software architecture and data flow.
  - Use `sequenceDiagram` for runtime interactions.
- If a task has both architecture and runtime concerns, include both an architecture/process diagram and a runtime/sequence diagram, or explain in Chinese why one diagram is not applicable.

## Diagram Quality

- Diagrams must use real module names, data artifacts, services, commands, or interfaces from the repo after exploration.
- Do not use generic boxes such as "Component A" unless the design is intentionally abstract.
- Prefer Mermaid in Markdown so the docs stay reviewable in git.

## Per-Doc Expectations

### Feature breakdown (PM)

- Feature scope, non-goals, deliverables, acceptance criteria
- Task DAG, role schedule, dependency risks
- Approval state and downstream handoff packets
- Module/software design idea (with `设计思想`)

### Architecture design

- System context, module boundaries, ownership, dependency direction
- Runtime/deployment shape
- Overall architecture or process diagram
- Major alternatives rejected

### Class/interface design

- Public interfaces/protocols, implementation classes, method signatures
- Inputs/outputs, errors, ownership/lifecycle, dependency injection, extension points
- Interface/implementation separation or Pimpl/Impl boundaries
- How the object model supports the design idea

### Data-flow design

- Source data, schemas/contracts, transformations, artifact paths
- Validation checks, error handling, provenance
- Total data-flow diagram when data movement is part of the work

### Sequence/runtime flow

- User/operator flow, async jobs, state transitions, background workers
- Service calls, failure/retry behavior
- Total runtime/process diagram when execution order matters

### Configuration/API docs

- Config files, environment variables, CLI/API entrypoints
- Defaults, override rules, migration notes

## Locations

- If the repo has a docs convention, follow it for durable project docs.
- Otherwise use `docs/` for durable project docs.
- Use `.grok/sdlc-runs/<run-id>/docs/` for run-local design notes, especially before implementation is accepted.
- For user-requested durable requirements, update the requested document in place and cross-reference run-local notes.

## Acceptance Bar

Tester and PM acceptance must reject documentation that is:

- mostly English (when Chinese is required)
- missing `设计思想`
- missing a total process/architecture diagram when applicable
- using a diagram that does not match the design or implementation
