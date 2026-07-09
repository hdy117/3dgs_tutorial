# Object-Oriented Implementation Contract

Prefer object-oriented implementation for non-trivial code changes. Use functions for small pure utilities, simple glue, or framework-required callbacks, but organize core behavior around explicit classes, protocols/interfaces, dataclasses/models, and services when state, lifecycle, external resources, configuration, or extension points are involved.

Default class design should separate interface from implementation whenever the repo language and framework make that practical. Treat public contracts as stable surfaces and keep implementation details behind private implementation classes, adapters, service internals, or Pimpl/Impl-style handles.

## Architect Obligations

- Define the object model before implementation: public interfaces/protocols, implementation classes, responsibilities, ownership, lifecycle, collaborators, constructor dependencies, public methods, hidden implementation state, error contracts, extension points, and where simple functions are intentionally used.
- Document interface/implementation separation for each non-trivial class:
  - what belongs in the public contract
  - what remains private
  - which dependencies are injected through the interface
  - whether Pimpl/Impl is appropriate for ABI stability, compile-time isolation, dependency hiding, or implementation privacy

## Developer Obligations

- Implement core workflows through cohesive classes or services with clear boundaries.
- Avoid large procedural scripts, global mutable state, hidden singletons, and long functions that mix configuration, I/O, transformation, validation, and orchestration.
- Keep public methods and constructors small and contract-focused; delegate details to private methods, private implementation classes, adapters, or language-idiomatic Pimpl/Impl structures.
- Prefer composition and dependency injection over hard-coded dependencies.
- Keep adapters at external boundaries, domain/data contracts in explicit models, and orchestration in a narrow service or runner class.

## Language Guidance

- **Python**: prefer `Protocol` or abstract public contracts plus concrete implementation classes, `dataclass`, typed config/model classes, and small service classes when they fit the repo style. Use leading-underscore/private implementation helpers where a full interface class would add noise.
- **C++**: prefer interface headers that expose minimal public contracts and keep heavy dependencies, mutable state, and algorithm details in `.cc/.cpp` files or `Impl`/Pimpl classes when ABI stability, rebuild cost, or dependency isolation matters.
- **Other languages**: use the idiomatic equivalent instead of forcing Python-specific or C++-specific patterns.

## Pragmatism

- Do not add abstract base classes, inheritance hierarchies, factories, or manager classes unless they reduce real complexity, isolate change, or match an existing repo pattern.
- Follow existing repo patterns when they already define a clear architecture.

## Acceptance Bar

PM acceptance criteria and tester checks must include whether:

- the implementation follows the agreed object model
- public interfaces stay separate from implementation details
- Pimpl/Impl-style choices are justified where used or omitted
- any procedural exceptions are justified
- class responsibilities remain cohesive
