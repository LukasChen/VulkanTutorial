# Engine Project TODO

## Foundation

- [ ] Split the current code into shared engine modules and thin app executables.
- [ ] Create separate `Runtime` and `Editor` targets in CMake.
- [ ] Define a stable folder structure for engine, game, editor, and assets.
- [ ] Add project-wide logging, assertions, and error reporting.
- [ ] Add configuration files for build, runtime, and editor settings.
- [ ] Set up third-party dependency management and version pinning.

## Core Engine

- [ ] Implement a scene system with entities, components, and serialization.
- [x] Build an ECS architecture with clear ownership and update ordering.
- [ ] Add a transform hierarchy for parent-child scene relationships.
- [ ] Implement a game loop with fixed timestep simulation support.
- [ ] Add an event or messaging system for engine subsystems.
- [ ] Add a basic service or subsystem registry only where needed.

## Rendering

- [x] Add camera
- [x] Refactor rendering code out of the monolithic engine class.
- [ ] Create abstractions for render passes, pipelines, buffers, and materials.
- [ ] Add mesh, texture, and shader asset management.
- [ ] Support multiple cameras and viewport rendering.
- [ ] Add material instances and parameter editing.
- [ ] Implement frustum culling.
- [ ] Add instanced rendering.
- [ ] Add skybox or environment lighting support.
- [ ] Add shadow mapping.
- [ ] Add post-processing pipeline support.
- [ ] Add debug rendering for bounds, lights, and physics shapes.

## Asset Pipeline

- [ ] Define source asset and cooked asset formats.
- [ ] Add asset importers for meshes, textures, materials, and scenes.
- [ ] Add metadata files for imported assets.
- [ ] Implement asset GUIDs and reference tracking.
- [ ] Add hot-reload for shaders and selected asset types.
- [ ] Build an asset database or registry.
- [ ] Add background asset processing jobs.

## Scene and Gameplay

- [ ] Create scene save/load in a human-readable format first.
- [ ] Add prefabs or reusable scene templates.
- [ ] Add scripting integration plan and runtime boundaries.
- [ ] Define game-specific components separately from engine components.
- [ ] Add spawn/despawn workflows and scene transitions.
- [ ] Support runtime editing hooks where useful for debugging.

## Physics

- [ ] Choose between integrating a physics library or building a minimal custom layer first.
- [ ] Add collider components for box, sphere, and capsule shapes.
- [ ] Add rigid body simulation support.
- [ ] Add collision filtering and layers.
- [ ] Add trigger volumes.
- [ ] Add character controller support.
- [ ] Add physics debug drawing.
- [ ] Keep physics data synchronized with transforms and scene entities.

## Animation

- [ ] Add skeleton and animation clip asset support.
- [ ] Implement animation playback and blending.
- [ ] Add animation state machines.
- [ ] Add root motion support if needed by the game.
- [ ] Add animation preview tools in the editor.

## Particles and VFX

- [ ] Implement a particle system with CPU simulation first.
- [ ] Add emitter components and reusable particle presets.
- [ ] Support burst, looping, and one-shot emitters.
- [ ] Add color, size, lifetime, velocity, and spawn-rate curves.
- [ ] Add editor preview and live tuning for particle effects.
- [ ] Evaluate later whether GPU-driven particles are worth the complexity.

## Audio

- [ ] Integrate an audio backend.
- [ ] Add audio source and listener components.
- [ ] Support 2D and 3D playback.
- [ ] Add streaming for large audio assets.
- [ ] Add mixer buses, volume controls, and mute/solo tools.
- [ ] Add editor preview for sound assets.

## Editor

- [ ] Create a separate editor executable linked against shared engine code.
- [ ] Add dockable editor UI.
- [ ] Add a scene hierarchy panel.
- [ ] Add an inspector panel for component editing.
- [ ] Add a content browser for assets.
- [ ] Add viewport controls for orbit, fly, and selection navigation.
- [ ] Add transform gizmos for move, rotate, and scale.
- [ ] Add undo/redo support.
- [ ] Add play-in-editor and stop/reset workflows.
- [ ] Add prefab editing support.
- [ ] Add scene save prompts and dirty state tracking.
- [ ] Add editor camera bookmarks and layout persistence.

## Tooling

- [ ] Add unit tests for engine utilities and serialization.
- [ ] Add integration tests for scene loading and asset import.
- [ ] Add formatting and linting rules.
- [ ] Add shader build automation and validation.
- [ ] Add profiling instrumentation for CPU and GPU work.
- [ ] Add crash dump or failure capture support.
- [ ] Set up CI builds for debug and release configurations.

## Performance

- [ ] Add frame timing and subsystem profiling overlays.
- [ ] Track allocations and memory usage.
- [ ] Add resource lifetime tracking for GPU objects.
- [ ] Reduce per-frame allocations in the renderer and scene update.
- [ ] Add job system groundwork for future parallelism.

## Save/Load and Data

- [ ] Define save-game data separately from scene data.
- [ ] Add versioning for scene and asset formats.
- [ ] Add migration support for older serialized files.
- [ ] Validate asset and scene references on load.

## Stretch Goals

- [ ] Networking foundation if multiplayer is planned.
- [ ] Navigation or pathfinding system.
- [ ] Terrain tools.
- [ ] UI framework for in-game interfaces.
- [ ] Visual scripting support if the project needs non-programmer workflows.
