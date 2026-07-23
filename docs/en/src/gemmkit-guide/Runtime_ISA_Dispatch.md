# Runtime ISA Dispatch

gemmkit ships one engine and picks the instruction set it runs on when your program starts, not when you compile it. A build made on a laptop and copied to a server will use the server's AVX-512 if the server has it; the same binary run on an older machine quietly falls back to a narrower kernel. You do not select a backend, gate on a `cfg`, or rebuild per host: the first GEMM call detects the CPU's features, caches the winning kernel, and every later call is a plain indirect call through that cached pointer.

## The backend roster

The set of kernels a build carries depends on the target architecture, and which one runs depends on the CPU. From fastest to slowest, the candidates are:

- **AVX-512F** on x86-64, the widest float kernel. Two dot-product specializations sit alongside it for the narrow element types: **AVX-512 VNNI** (`vpdpbusd`) for `i8 -> i32`, and **AVX-512 BF16** (`vdpbf16ps`) for `bf16`. These require the `int8` and `half` features respectively, and the CPU must report the matching feature bit.
- **FMA / AVX2** on x86-64, the widen-FMA kernel for machines without AVX-512.
- **NEON** on aarch64, where SIMD is baseline (every aarch64 CPU has it), so there is nothing to detect at runtime.
- **simd128** on wasm32, chosen at compile time rather than runtime (see below).
- **scalar**, the portable floor. It exists on every target and is what runs when nothing better is available. A correct, if unaccelerated, result is always reachable.

Tile geometry is the one thing that changes per `(element type, ISA)` pair: the microkernel computes an `MR x NR` register tile whose size is tuned to the ISA's vector width. For `f32` the shipped tiles are:

| ISA | f32 tile (MR x NR) |
| --- | --- |
| AVX-512F | 32 x 12 |
| FMA / AVX2 | 16 x 6 |
| NEON | 16 x 4 |
| simd128 | 8 x 4 |
| scalar | 4 x 4 |

All five run the same generic float microkernel; only the tile shape differs. `MR` is `MR_REG * LANES`, so a wider vector buys a taller tile. `f64` halves the lane count and therefore halves `MR` (AVX-512F `f64` is `16 x 12`, and so on). The VNNI and BF16 dot kernels use their own depth-grouped geometry and are covered under [Element Types](Element_Types.md); this table is background for reasoning about why a kernel packs and blocks the way it does, not a knob you set.

## Automatic selection

Each element type owns a single dispatch slot, a `OnceLock` holding a typed function pointer. On the first call for that type the selection ladder runs feature detection once, picks the best available kernel, and stores its monomorphized entry points (plain, prepacked, fused) plus the tile geometry. The one-time cost, the `is_x86_feature_detected!` probe and the `OnceLock` initialization, is paid by whichever call happens first; from then on dispatch is a cached pointer load and an indirect call, with no per-call branching on the ISA. There is no `transmute` and no atomic pointer juggling behind this, just a typed slot per type.

A consequence worth stating plainly: **there is no public API that reports which ISA was selected.** The choice is an internal detail of the memoized slot. If you need to be certain a specific kernel is live, do not try to read it back, pin it (next section) and let a mismatch fail loudly instead.

## Pinning a kernel with `GEMMKIT_REQUIRE_ISA`

Setting the environment variable `GEMMKIT_REQUIRE_ISA` forces exactly one kernel end to end instead of auto-selecting. The accepted values (case-insensitive, surrounding whitespace trimmed) are:

| Value | Forces | Also accepts |
| --- | --- | --- |
| `scalar` | the portable scalar kernel | |
| `fma` | the FMA / AVX2 widen kernel | `avx2` |
| `avx512f` | the AVX-512F widen kernel | |
| `avx512vnni` | the `i8` `vpdpbusd` dot kernel (plain AVX-512F for other types) | `vnni` |
| `avx512bf16` | the `bf16` `vdpbf16ps` dot kernel (plain AVX-512F for other types) | `bf16` |
| `neon` | the aarch64 NEON kernel | |
| `simd128` | the wasm32 `simd128` kernel | `wasm` |
| `auto` | normal auto-selection (also the default when unset or empty) | |

The `avx512vnni` and `avx512bf16` pins select the dot kernel for their one narrow type and the plain AVX-512F path for everything else, so a mixed workload under one of those pins still runs correctly for its other types.

The contract is **panic, not fallback.** If the requested ISA is unavailable, because the CPU does not report the feature, because the value names an ISA that does not exist on this target architecture (`neon` on x86, `avx512f` on aarch64), or because the value is an outright typo, dispatch panics rather than silently running a different kernel. That is deliberate, and it is exactly what you want for CI: a job whose whole purpose is to exercise the AVX-512 VNNI path must not pass by quietly testing the scalar fallback because a feature flag was misspelled or an emulator was misconfigured. gemmkit's own CI pins each kernel this way, running the x86 dot kernels under Intel SDE, NEON on aarch64, and `simd128` on wasm; a broken pin turns into a red build instead of a false green.

The value is **read once, before the first dispatch,** and memoized alongside the kernel choice. Set it in the process environment before any GEMM runs; changing it mid-process has no effect, because the slot is already populated. An unrecognized value is a hard error precisely so it cannot be mistaken for `auto` and slip through.

## WebAssembly is compile-time

wasm32 has no runtime feature detection, so `simd128` is not chosen by probing the machine; it is selected by a compile-time `cfg`, and the build must actually enable it with `-C target-feature=+simd128`. Forget the flag and the wasm build silently uses the scalar floor. Pinning `GEMMKIT_REQUIRE_ISA=simd128` turns that silent degradation into an assertion: the build panics if the SIMD path is not live, which is why the wasm CI jobs pin it. The full wasm build story, including the threaded target, is in [no_std and WebAssembly](no_std_and_WebAssembly.md).
