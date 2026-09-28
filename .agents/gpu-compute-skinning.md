# GPU Compute Skinning Implementation Instructions

## Objective

Implement glTF skeletal animation for this Vulkan/Slang project using a dedicated
compute-skinning pass. The compute shader must deform each skinned mesh once per
frame into a GPU output vertex buffer. The rasterization vertex and shadow
vertex shaders must consume that output buffer. Do not add CPU vertex deformation
as the runtime rendering path.

This instruction is based on:

- glTF inverse-bind and joint-matrix semantics: `gltfTutorial_019_SimpleSkin.md`
  and `gltfTutorial_020_Skins.md`.
- The Vulkan Tutorial compute-skinning design:
  [Introduction](https://docs.vulkan.org/tutorial/latest/Advanced_glTF/Skeletal_Compute_Skinning/01_introduction.html)
  and
  [Compute Skinning](https://docs.vulkan.org/tutorial/latest/Advanced_glTF/Skeletal_Compute_Skinning/03_compute_skinning.html).

## Scope and responsibility split

The first implementation is intentionally hybrid, matching the referenced Vulkan
tutorial:

1. The CPU advances animation time, samples glTF channels, and propagates the
   joint hierarchy.
2. The CPU computes one joint matrix per glTF joint and uploads the palette:

   ```text
   jointMatrix[j] = currentGlobalJointTransform[j]
                     * inverseBindMatrix[j]
   ```

3. A Vulkan compute shader performs linear blend skinning (LBS), one invocation
   per vertex.
4. The compute output is read by the normal and shadow rasterization passes.

Do not invert the current animated joint transform. `inverseBindMatrices` are
bind-pose data supplied by glTF. Do not move animation curve sampling to a GPU
compute shader in the first milestone; that is a separate optimization after
correctness is established.

## Existing project constraints

- `Model` in `model.h`/`model.cpp` currently flattens OBJ and glTF geometry and
  discards skins, joints, animations, and primitive boundaries. Extend the glTF
  representation rather than trying to reconstruct that data in `Renderer`.
- `Vertex` in `vertex_layout.h` currently contains only position, normal, and UV.
  Preserve this static mesh path for OBJ and non-skinned glTF assets.
- `Renderer` currently uses per-frame resources, two frames in flight, an
  instanced static draw path, set 0 for frame uniforms, and set 1 for materials.
- `shaders/shader.slang` and `shaders/shadow.slang` currently read position and
  normal from vertex input and apply an instance model matrix.
- `CMakeLists.txt` already compiles Slang to SPIR-V. Add the compute shader to
  that existing target; do not introduce a second shader build system.
- The existing graphics queue supports compute operations. Use it for the first
  implementation so that no separate compute-queue ownership transfer is
  needed. A dedicated compute queue is an optional later optimization.
- Follow the repository's direct failure style for required data. Report invalid
  glTF data with a useful exception or validation diagnostic; do not silently
  skip required skin components.

## Asset representation

Refactor the glTF side of `Model` into an asset representation that retains, at
minimum:

- one or more mesh primitives with their own vertex/index ranges;
- skinned vertex attributes `JOINTS_0` and `WEIGHTS_0`;
- node local transforms, parent/child relationships, and scene roots;
- skin joint-node indices and one inverse bind matrix per joint;
- animation clips, samplers, channels, interpolation mode, and duration.

Keep OBJ loading working. A glTF primitive with no skin remains a static mesh.
Validate accessor component types, byte strides, counts, and normalized weights
while loading. Normalize weights only when the source requires it; preserve the
glTF values and validate that the final sum is usable for LBS.

Use a stable joint index convention: `JOINTS_0` values index the skin's `joints`
array, not arbitrary node indices. Store the node index alongside each skin
joint so hierarchy traversal and inverse-bind lookup cannot be confused.

## Animation and ECS design

Add an `Animator` component and an `AnimationSystem` that are registered from
`main.cpp` through the existing `Engine::BindSystem` mechanism. The component
should own or reference:

- the selected model/skin and clip;
- playback time, speed, looping, paused/stopped state;
- optional transition state for a later cross-fade;
- a handle or index for the renderer's joint-palette allocation.

Sample translation, rotation, and scale channels independently. Use the bind
local transform for a joint when a channel is absent. Use linear interpolation
for translation/scale and normalized quaternion interpolation for rotation in
the first milestone. Clamp or wrap time according to the clip and looping
flags. Compute global joint transforms in parent-before-child order.

Keep the glTF skeleton separate from the ECS `Transform` hierarchy. Define one
explicit model-root transform for placing the animated entity in the scene. If
root motion is later enabled, extract it once and do not also apply that motion
through the joint palette.

The CPU-side palette must be in the same coordinate convention as the mesh and
the current Slang shaders. Verify the matrix multiplication order with a bind
pose test before adding animation playback.

## GPU buffer ABI

Use explicit GPU-facing structs. Do not rely on `glm::vec3` packing matching
storage-buffer layout. Prefer 16-byte fields (`float4`/`uint4`) and document the
matching C++ and Slang layouts in one place.

The recommended resources per skinned primitive are:

- immutable bind-pose input vertex buffer, usable as a storage buffer;
- immutable index buffer;
- immutable joint-index buffer (`uint4` per vertex);
- immutable weight buffer (`float4` per vertex);
- per-frame joint-matrix storage buffer;
- per-frame output vertex buffer, usable as both storage and vertex input.

The compute shader descriptor set should match this ABI:

```text
binding 0: StructuredBuffer<InputVertex>       bind-pose vertices
binding 1: RWStructuredBuffer<OutputVertex>    skinned output vertices
binding 2: StructuredBuffer<float4x4>          joint matrices
binding 3: StructuredBuffer<uint4>              joint indices
binding 4: StructuredBuffer<float4>             joint weights
```

Use a push-constant struct containing at least `vertexCount`. If several
primitives share a palette, include a joint-base/palette offset rather than
copying matrices unnecessarily. Allocate output buffers per frame in flight or
use a ring allocation; never overwrite a buffer still used by an earlier frame.

The output vertex must contain position, normal, and UV. Positions use the full
blended 4x4 matrix. Normals use the rotational 3x3 portion and are normalized.
UVs are copied unchanged. If tangents are added later, transform and normalize
their xyz while preserving handedness.

## Compute shader

Add `shaders/skinning.slang` with a compute entry point named `main` and a
conservative `[numthreads(64, 1, 1)]` declaration. Each invocation:

1. Returns when `SV_DispatchThreadID.x >= vertexCount`.
2. Loads one bind-pose vertex and its four indices/weights.
3. Builds `skinMatrix = sum(weight[i] * jointMatrices[index[i]])`.
4. Transforms position and normal and writes exactly one output element.

Keep the shader's matrix convention consistent with the existing Slang code,
which transposes instance matrices before using `mul`. Confirm the convention
with a one-joint translation and a 90-degree rotation test.

## Vulkan compute pipeline

Add compute-specific resources to `Renderer`:

- compute descriptor-set layout and pool capacity;
- compute pipeline layout with a compute-stage push-constant range;
- compute shader module and pipeline;
- per-frame joint and output buffers plus mapped upload/storage metadata;
- a `SkinComputeResources` record attached to each skinned primitive.

Create the descriptor set layout with five storage-buffer bindings as listed
above. Use `VK_SHADER_STAGE_COMPUTE_BIT` for these bindings. The output buffer
must be created with both `eStorageBuffer` and `eVertexBuffer` usage flags.

Record commands in this order for each frame:

1. Update ECS animation state and global joint transforms.
2. Upload the current joint palette to the frame's joint buffer.
3. Bind the compute pipeline and the primitive's compute descriptor set.
4. Push `vertexCount` and dispatch `(vertexCount + 63) / 64` groups.
5. Insert a `vk::DependencyInfo`/`vk::BufferMemoryBarrier2` for every output
   buffer before any draw reads it:

   ```text
   source stage:  COMPUTE_SHADER
   source access: SHADER_WRITE
   destination stage: VERTEX_INPUT
   destination access: VERTEX_ATTRIBUTE_READ
   ```

   Include ray-tracing or other consumer stages only when those consumers are
   actually added. Do not use a host-visible readback to synchronize the pass.
6. Bind the output buffer as the vertex buffer and issue the graphics draw.

Use the existing graphics command buffer and submission path initially. If the
compute pass ever moves to a separate queue, add explicit queue-family ownership
transfers and semaphores; do not assume a pipeline barrier alone synchronizes
different queues.

## Graphics integration

Add a dedicated skinned draw path before attempting to merge it with the current
instanced batching. Static OBJ/tree/box meshes must continue using their current
vertex buffer and instance buffer.

For skinned draws:

- bind the compute-produced output buffer instead of the bind-pose vertex buffer;
- retain the existing frame UBO and material descriptor sets;
- apply the entity/model transform exactly once;
- update `shader.slang` and `shadow.slang` input layouts only as needed for the
  output vertex ABI;
- do not apply the inverse bind matrix again in the graphics shader;
- ensure shadow rendering consumes the same deformed output buffer after the
  compute-to-vertex-input barrier.

Start with one non-instanced animated entity. Add instancing or palette offsets
only after the single-entity path is correct. A later batched design may use a
large output buffer with per-draw vertex offsets and a palette range.

## Build and diagnostics

Add `skinning.slang` to `SHADER_FILES` in `CMakeLists.txt`, with `main` as its
compute entry point. Keep the existing Slang command flags and copy behavior.
Make shader compilation part of the normal target dependency.

Add diagnostics for:

- missing skin attributes or inverse bind matrices;
- invalid joint indices and non-finite weights;
- clip duration and active playback state;
- joint count, vertex count, dispatch count, and palette byte size;
- missing compute output resources or descriptor updates.

Use validation layers and GPU-assisted validation when available. Label compute
buffers and the compute pipeline with debug names if the renderer already has a
debug-label helper.

## Milestones and acceptance tests

Implement and verify in this order:

1. Load a known animated `.glb` and print skin/joint/clip metadata.
2. Render the bind pose through the existing static path.
3. Create the compute pipeline and write output vertices without animation.
4. Dispatch compute and render the output buffer in the main pass and shadow
   pass. Verify the output is identical to bind pose.
5. Animate one joint with a translation and a rotation; compare against a small
   CPU reference calculation in a test or debug mode.
6. Add clip looping, pause, stop, speed, and missing-channel bind-pose fallback.
7. Add cross-fade blending only after single-clip playback is stable.
8. Test two animated entities, resize, two frames in flight, parent transforms,
   shadows, and destruction/recreation of resources.

The minimum correctness checks are:

- bind pose has no visible displacement;
- a joint's inverse bind matrix is applied exactly once;
- weights and joint indices affect the intended vertices;
- compute writes are visible to both main and shadow vertex input;
- normals remain normalized and lighting follows the deformed surface;
- no validation-layer synchronization or descriptor-layout errors occur;
- static OBJ rendering remains unchanged.

Do not mark the feature complete until the compute output buffer is the shared
source for rasterization and the CPU is no longer deforming vertices per frame.
