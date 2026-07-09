# Source Code Comment Contract

Require source-code comments as an implementation deliverable, not optional polish. Comments must explain purpose, contracts, invariants, and non-obvious steps while staying concise enough to maintain.

## File Header

Every source code file must start with a module/file docstring or leading comment that states:

- the file's purpose
- main responsibilities
- important entrypoints
- major external contracts

Use the idiomatic comment style for the language.

## Types And Members

- Every class, interface, protocol, enum, dataclass/model, and public type alias must include a docstring or adjacent comment that explains its role, lifecycle/ownership, important invariants, and extension points when relevant.
- Every member variable must be documented near its declaration, including:
  - class attributes
  - dataclass/Pydantic/model fields
  - struct fields
  - enum values with non-obvious semantics
  - persistent instance attributes initialized in constructors
  - cached state, mutable containers, handles/resources, configuration fields

## Functions And Methods

Every function and method must include a docstring or adjacent/leading concise comment. This includes:

- public APIs and private helpers
- nested functions and closures assigned as reusable callbacks
- CLI handlers, test fixtures/helpers
- public properties and language-equivalent function constructs

The comment must explain:

- what it does
- important inputs/outputs
- side effects
- error behavior
- assumptions
- non-obvious pre/postconditions

No function is exempt merely because its name or signature seems clear. Trivial one-line helpers may use a short purpose comment, but must still be documented.

## Key Code Blocks

Add comments for key algorithm steps and boundaries:

- validation gates
- external format conversion
- state transitions
- async/thread/process boundaries
- resource ownership
- retries, caching, batching
- numerical logic, tensor shape changes, coordinate transforms
- error handling

Also document:

- service boundaries, config schemas, data contracts
- coordinate frames, units, timestamp conventions
- device/dtype assumptions, schema versions, error-code semantics
- adapter boundaries where external formats convert into internal contracts

## Style

- Keep inline comments short and local; put longer reasoning in design docs.
- Do not write comments that merely repeat syntax such as "increment i" or "return result".
- Explain why the code exists or what contract it preserves.

## Acceptance Bar

Tester must flag as acceptance issues:

- any undocumented function or method
- a source file lacking a purpose header
- a class/interface/member variable lacking useful documentation
- key code steps that are not explained

Do not demand line-by-line narration of obvious syntax.
