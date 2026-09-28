# Ray tracing

This example builds BLAS/TLAS trees and traces primary and shadow rays in Slang compute shaders.
Nodes, geometry, instance headers, function tables, and scratch storage use GPU pointers through the shared root ABI.
Run `example_raytrace` to write `raytrace.png`: two boxes share one indexed BLAS, alongside a room and a moving procedural sphere.

## Build and update

`build.hpp` records GPU builds and refits into a `ComputeBatch`, using caller-owned BVH storage and reusable scratch.
`begin_compute` borrows a kernel array and a mapped root allocator. `launch<Root>(batch, kernel, arguments, x, y, z)`
copies arguments into a distinct GPU allocation, binds the kernel, and rounds invocation counts up to threadgroups.
Launches execute serially with compute barriers between them; consecutive launches of one kernel reuse its binding.
`readback_and_wait(batch, {{.source = ..., .destination = ...}, ...})` copies independent outputs, submits once, waits,
and releases the batch's command pool and semaphore. Destinations must not overlap other copies' sources or destinations.
An empty copy list submits and waits without readback.
Keep kernel storage, roots, and referenced resources alive until it returns. Kernels must guard padded invocations.
`get_bvh_build_sizes(count)` reports persistent BVH, reusable build/refit scratch, and per-build/per-refit root bytes, including allocator padding.
Primitive data and the `Bvh` header are separate. `allocate_bvh` and `allocate_bvh_scratch` reserve these layouts from caller-owned arenas.
`allocate_bvhs(allocator, inputs, outputs)` reserves multiple BVHs and writes their headers into a caller-owned host span of equal length.
`BvhAllocationInput.count` is the primitive count for builds or leaf count for imports. Storage is owned by the allocator's backing heap.
Build or refit children before their parent scenes.
Input trees must be nonempty and scene graphs acyclic, with at most four instance levels along any path.

`build_bvhs(batch, requests, scratch)` and `refit_bvhs(batch, requests)` accept spans or initializer lists of `BvhBuildRequest`.
Requests in one call must be independent, with disjoint output storage. Submit child builds/refits in an earlier call than their parent scenes.
Each request names a host `Bvh` header, count, and Morton normalization bounds; refit ignores those bounds and retains topology.
Headers/requests are consumed during recording. Referenced GPU data must remain alive until the batch completes.
Trees with up to 512 primitives use one 128-thread workgroup each in a fused LBVH build/refit dispatch; larger trees use multi-dispatch PLOC.
The small builder uses stable 4-bit Morton sorting and binary radix topology, with sorted positions resolving duplicate keys.
This matches HIPRT's small-tree algorithm family and 512-primitive limit; node layout, binary leaves, and scheduling remain this module's own.
Calls with more than 65,535 small trees split into multiple dispatches. Imported trees still use the import/refit-import APIs.
`get_bvh_batch_sizes(requests)` sums persistent storage and root bytes and reports the largest reusable scratch requirement.
Reserve BVHs with `allocate_bvh`; for any large builds, use `allocate_bvh_scratch` with their largest count. All-small batches need no scratch.
Empty request spans record nothing and report zero capacity. Both operations remain inside the caller's `ComputeBatch`, with no additional submission or wait.

Geometry accepts float3 vertices with optional uint32 index triples, or custom primitive AABBs.
`PrimitiveInput.stride` is the vertex, AABB, or instance byte stride; `index_stride` is the byte stride between index triples.
AABB stride is 24 for two float3 values or 32 for two float4 values; max starts at half the stride and w is ignored, matching HIPRT.
Use at least 4-byte alignment for vertices, indices, and AABBs, and 8-byte alignment for instance records and BVH headers.
Refit preserves topology and requires unchanged primitive count, order, kind, and vertex indexing.
Update vertices, custom bounds, or instance transforms, then refit affected trees from children to parents.

### Custom BVH import

`import_bvh(batch, bvh, input)` imports a GPU-resident binary tree for triangles, custom primitives, or instances.
Provide n leaves and n-1 internal nodes, with internal node zero as root; a singleton uses leaf zero and no internal array.
Child indices address the array selected by `import_internal`/`import_leaf`. Every non-root node has one parent, with no cycles or unreachable nodes.
Bounds must enclose the referenced geometry and child bounds. Leaves retain their primitive IDs, including repeated IDs.
`BvhImportInternal` and `BvhImportLeaf` preserve HIPRT's 64/32-byte input record layouts; the resulting BVH uses this module's own layout.
Use `get_bvh_import_sizes(n)` and `allocate_bvh(..., n)` with the leaf count, which may exceed the primitive count. Import needs no scratch.
Keep source arrays separate from output and alive until completion. Source arrays are not modified.
`refit_imported_bvh(batch, bvh, scratch, n)` updates leaf and internal bounds from the current geometry or child BVHs without reading the import arrays.
Keep topology, leaf IDs/count and primitive kind unchanged. Size the disjoint scratch range with `get_bvh_import_sizes(n).refit_scratch`.
Refit preserves repeated IDs and recomputes full primitive bounds; spatially clipped references may therefore become less tightly bounded.
Topology changes require importing again. Use `refit_bvh` for trees emitted by `build_bvh` or `build_bvhs`.
The example imports the procedural sphere's singleton BLAS.

`import_bvhs(batch, requests)` accepts `BvhImportRequest` spans or initializer lists and combines all tree workgroups into one ordered range.
Each workgroup locates its tree in a prefix table; large trees use multiple workgroups. Outputs must not overlap other outputs or any import source.
`get_bvh_import_batch_sizes(requests)` sums storage and root bytes and reports the maximum reusable refit scratch. Import needs no scratch.
`refit_imported_bvhs(batch, requests, scratch)` records ordered updates with that shared scratch; only each BVH and leaf count are consumed.
The source import arrays need not survive the completed import. Update children in earlier calls than their parent scenes.
Empty spans record nothing. Import and imported-refit dispatches split at 65,535 workgroups; capacity queries include those extra roots.

### Bounds export

`export_bvh_aabbs(batch, {geometry, scene, ...}, gpu_bounds)` collects current root AABBs in one dispatch after builds or refits.
It accepts built and imported trees, including singletons. Results follow the input order and use 24-byte `Aabb` records.
Geometry bounds are in object space; scene bounds include instance transforms and the conservative motion interval.
As in HIPRT's AABB export, this reads stored root bounds; it does not recompute tighter bounds or apply query masks/time.
Each nonempty call uses `16 * count + 32` bytes in the batch's root arena and requires `24 * count` output bytes, with no scratch.
Input headers are consumed immediately; GPU output must be separate from BVH storage and batch roots.
Pass the output to `readback_and_wait` for CPU access. The example reads all four bounds alongside the image with one submission and wait.

Instances reference GPU-resident `Bvh` headers and support masks, affine matrices, and SRT transforms.
Motion keys have strictly increasing times; queries clamp outside their range, with time zero selecting the first key as in HIPRT.
SRT rotation uses an axis in xyz and an angle in radians in w. Matrix keys are decomposed into rotation, scale, shear, and translation.
Translation, scale, and shear interpolate linearly; rotation uses shortest-path quaternion interpolation, matching HIPRT's default transform mode.
Transforms must remain invertible throughout interpolation. Swept bounds include intermediate motion;
Motion uses a conservative sphere bound, which can increase traversal work.

## Query

Include `query.slang` and supply the application functions declared there; `ray_functions.slang` provides sample bindings.
`trace_bvh` selects closest or first accepted hit. `begin_query` followed by `next_hit` resumes across accepted hits,
including nested scenes. Iteration is in traversal order, not sorted by distance; `commit_hit` optionally narrows the ray interval.
`closest_hit` consumes the remaining cursor, commits the closest accepted hit, and finishes; another call returns false.
Exhaustion sets `query_finished`; exceeding four instance levels sets `query_stack_overflow` without accessing beyond the cursor arrays.
Both states are terminal. Closest traversal returns no hit on overflow; callers needing the reason should inspect the cursor state.
A miss has `primitive == invalid_index`.

`QueryOptions` selects time, instance mask, ray type, function table, and an application payload GPU pointer.
Every instance along a path must overlap the query mask. The table index is
`ray_type * geometry_type_count + geometry_type`; both types must be within the supplied table dimensions when accessing the table.
Triangle geometry with `geometry_type == invalid_index` bypasses callbacks even when a table is supplied.
Entries select application intersection/filter IDs and data pointers. `invalid_index` disables an entry's callback;
triangles use built-in intersection, while AABBs require a custom intersection callback.
Callbacks receive the object-space ray and original primitive/instance IDs. Intersection returns true with a distance
inside the ray interval, object-space normal, and optional barycentrics; filters return true to discard a hit, as in HIPRT.
Callbacks preserve IDs and own payload semantics. Give concurrent rays separate mutable payload storage.

Hits contain distance, barycentrics, unnormalized object-space normal, and the instance path from outermost to innermost.
`hit_object_to_world` reconstructs the path transform at the query time; `inverse_transform`, `transform_point`,
`transform_vector`, and `transform_normal` handle space conversion. `transform_normal` takes the inverse transform
and returns an unnormalized inverse-transpose result. Transformed rays retain direction length so their distance parameter stays unchanged.

## Implementation

The builder adapts HIPRT's locally ordered clustering: Morton sorting, surface-area neighbour search,
and parallel merging of mutual neighbours. This version uses fixed Morton tiles and one dispatch per
level, retaining clusters for further merging across tile boundaries. It is a multi-dispatch PLOC
implementation; HIPRT's single-kernel H-PLOC scheduling and performance claims do not apply here.
Morton keys use stable 4-bit radix sorting: eight digit passes, with group histograms, hierarchical exclusive scans, and scatter.
Up to 128 entries sort in one workgroup/dispatch; a singleton skips sorting. No power-of-two padding is needed.
Sorting reuses both PLOC cluster arrays for temporary keys/IDs; the additional histogram hierarchy is included in the scratch size query.
Ranks use shared-memory integer atomics and bit counts without assuming a hardware subgroup size. Equal Morton keys retain primitive order.
Leaves contain one primitive; traversal uses parent links without a fixed tree-depth stack.
The four-level instance cursor is separate. Small-tree refit handles LBVH and PLOC through shared-memory ancestor doubling and integer bound reduction.
Each round snapshots sources into registers before any atomic writes; group barriers separate rounds. Shared storage is 28,644 bytes per workgroup.
Larger PLOC refit replays build levels and updates each tile's nodes in child-before-parent order.
Imported refit doubles ancestor distance each round and merges ordered float encodings with 32-bit integer atomics.
It uses two state arrays and `2 * ceil(log2(n)) + 2` ordered phases for n leaves, with O(n log n) work even for unbalanced trees.
Each phase reads a stable source array; dispatch barriers separate initialization, copying, atomic merging, and publishing final bounds.
This is a source module in the example, with application-compiled callbacks; it does not provide HIPRT's binary API or runtime compilation.

Build and run the kernel regression test with:

```powershell
cmake --build --preset msvc-debug --target test_raytrace
ctest --preset msvc-debug -R '^test_raytrace$'
```

The Vulkan-only regression test runs the same kernels with buffer device addresses and the validation layer,
independently of the wrapper's descriptor-heap and device-address-command requirements.
It checks tree structure and compares closest-hit and any-hit results against CPU brute force,
including partial workgroups, identical bounds, degenerate triangles, and deep trees. Additional cases cover four-level scenes,
strided indexed meshes, masks, matrix/SRT motion, callbacks and payload, resumable traversal, space conversion, and bottom-up refit.
Import coverage includes 18 balanced/skewed trees across three geometry states, shuffled internal indices, duplicate primitive IDs,
per-node bounds/topology checks, float4 AABBs, imported scene refit, and terminal instance-overflow/closest-cursor states.
Sorting coverage includes 80 cases through 131,073 entries, full uint32 keys, stable duplicate ordering, buffer guards, and three scan levels.
Batch coverage includes 48 triangle/AABB/scene trees through 512 primitives, independent CPU radix topology, three geometry states,
PLOC/LBVH refit compatibility, queries, root-bounds export, and a 512-leaf chain with depth 511 and a nonzero root.
Bulk import covers mixed tree sizes, exact allocation capacity, shuffled nodes, repeated primitive IDs, segmented import/refit and poisoned source arrays.
The harness calls the storage-size/allocation APIs directly but records GPU kernels through Vulkan; it does not exercise `ComputeBatch` submission.
The native example has also completed on an RTX 3080 with NVIDIA driver 617.14, producing its PNG and bounds readback with no Debug validation messages.
That run covers the example's build, import, export, trace, and `ComputeBatch` path; the broader regression cases still use the raw Vulkan harness.

## Remaining HIPRT alignment

| Area | Remaining work |
| --- | --- |
| Build algorithms | H-PLOC, fast/balanced/high-quality modes, spatial splits, triangle pairing and oriented bounds |
| SDK operations | Compaction and geometry/scene serialization |
| Traversal integration | Configurable traversal stacks/hints and runtime compilation/linking of application callbacks |
| Verification | Broader native-wrapper regression coverage and direct comparison against HIPRT |

These are implementation and verification gaps; successful kernel regressions do not establish full HIPRT parity.

See [HIPRT's MIT license](LICENSE-HIPRT.txt) and the repository's [third-party notices](../../THIRD_PARTY_NOTICES.md).
