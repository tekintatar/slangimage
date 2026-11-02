# Texture parameter contracts

The functional `si.ops` boundary rejects invalid numeric controls with
`ValueError`, before passing them to shaders. Scalar/vector checks happen on the
host; image-dependent dimensions and coordinate bounds are checked before output
allocation/dispatch. This policy concerns controls, not the values stored inside
an input image or LUT.

## Common rules

- Numeric controls must be finite and fit fp32. NaN, either infinity, overflow
  beyond fp32, and a nonzero value that rounds all the way to fp32 zero are
  rejected. Float64 input values are still allowed; exact fp32 representation is
  not required. Compute/storage output precision remains a separate contract.
- Integer controls must be integer-valued and fit their declared bounds.
  Values such as `4.0` are accepted; `4.5` is not silently truncated. Boolean
  controls accept booleans or numeric 0/1, not arbitrary truthy numeric values.
- Dimensions must be positive integers, except affine's documented zero sentinel
  for the input canvas. Generated/resized/tiled/affine output extents are checked
  against the selected device's reported 2D texture limit and signed RGBA
  indexing capacity. Tiny positive resize scales still produce at least 1x1.
- RGB controls (`false_color`, `normalize`) require exactly three finite
  components. CLAHE requires exactly two positive integer tile counts; their
  256-bin histogram product must fit signed indexing.
- Enum/mode errors are rejected; supplying both Gaussian parameter forms or both
  resize parameter forms is an error rather than silently choosing one.

## Operator-specific boundaries

| Controls | Contract |
|---|---|
| Gaussian, surface, bilateral, Kuwahara radii | Integer range 1..16383. The upper bound protects signed squared-offset/loop arithmetic, **not** execution time. Surface's radius is its legacy sample-count control. |
| Unsharp / EP unsharp radius | Their float-radius conventions remain: unsharp truncates before Gaussian validation; EP rounds ties-to-even before sample-count validation. The resulting radius/count must be 1..16383. |
| NLM search/patch radii | Integers 0..16383; zero search is identity and zero patch is a single-pixel patch. This can still be an extremely expensive filter. |
| Gaussian/surface/bilateral weight sigmas | Positive, from `2**-60` through `2**60`. These bounds keep squared sigmas and reciprocal weight factors finite and normal in fp32 fast math. |
| Bilateral spatial/range exceptions | Finite spatial sigma <=0 still selects box weighting. A zero range is permitted for `range_type='cutoff'`, meaning exact color matches; Gaussian range sigma must be positive. |
| NLM `h` / noise `sigma` | 0..`2**60`. Zero `h` retains the existing shader epsilon floor; zero noise sigma disables noise compensation. |
| Normalize `std` | Nonzero normal fp32 components: absolute value >=`2**-126`. Negative standard divisors remain accepted algebraically. |
| Exposure EV | -126..127 keeps `exp2(ev)` normal and finite. It does not guarantee finite intermediate products for arbitrary HDR image values. |
| Color controls | Hue/saturation/brightness clamp to [-1, 1]. Public signed contrast maps to a factor clamped to [0.001, 5]; values outside the usual [-1, 1] remain accepted. Direct `ColorControlsOp` callers instead use factor-style contrast, with the legacy <=1e-4 bypass. |
| Highlight/shadow | Highlights clamp to [0, 1]. Shadows accept finite fp32-range controls without a [-1, 1] clamp; that interval is the usual range, not validation. Positive luma follows the curve; zero/clipped-negative luma bypasses division. |
| Hue rotation | Finite fp32-range radians, including negative/multiple turns, are accepted without host modulo reduction. fp32 angle rounding loses phase precision at large magnitudes; final RGB is clipped. |
| False color | Each endpoint is a finite fp32-range RGB triple, passed as fp32 even with fp16 texture storage. Endpoints/output RGB are unclipped; only encoded-RGB luma is clipped to [0, 1] before interpolation. Extreme endpoints can still overflow arithmetic/storage. |
| Soft threshold | Positive `smooth_amount`, with strictly ordered clipped edges **after fp32 rounding**. Collapsed/reversed intervals are rejected, not left to undefined `smoothstep` behavior. |
| Two-point threshold | White must remain greater than black after fp32 rounding, and the resulting gap must be finite and normal in fp32. Valid narrow ranges use their actual gap; there is no shader epsilon floor that changes the requested slope. |
| Tile | Finite positive scales, finite phase offsets, valid extent/mode. The existing modulo phase normalization remains; inverse-scaled coordinates must fit signed shader indexing with rounding headroom. |
| Affine | Finite, nonsingular 2x3 forward matrix; finite fp32 inverse. A conservative absolute bound on inverse-coordinate intermediates prevents overflow and unsafe float-to-int conversion. Ordinary off-image coordinates still produce transparent fill. |

Intentional finite clamps are not turned into blanket positivity rules: gamma,
vibrance, sharpen-luma, white balance, pointillize's minimum cell size, CMYK's
minimum width/softness, and other clamped controls retain their existing behavior.
Pointillize additionally rejects positive cell radii too large for signed sample
coordinates. See individual API docstrings for color-space/clipping conventions.

## Scope and limits

The shared checks also strengthen Gaussian apply/reuse/amount paths, LUT size
validation, two-point threshold validation, and geometry tile/affine wrappers.
This does **not** certify every directly imported kernel/effect class: code
that calls kernel or effect classes directly bypasses the functional API, and
the remaining low-level contracts are not yet specified. No CUDA autograd
parameter policy was changed.

Representable parameters are not a promise of bounded latency, affordable
allocation, or finite output for arbitrary pixel data. Input pixel NaNs/Infs are
not scanned, and valid large images/radii/CLAHE grids can exhaust resources.
Service-facing request budgets and deployment timeouts remain separate work. The current checks do not make GPU kernels a
sandbox for hostile workloads.

Evidence: `tests/test_parameter_validation.py` uses host stubs to prove rejection
without submitting dangerous values to a GPU, and bounded fp16/fp32 GPU cases to
check preserved sentinels and weight/threshold boundaries. A signature inventory
enforces coverage of every public scalar numeric/boolean control (the CPU LUT
generator has separate dimension tests). See [TEST_COVERAGE.md](TEST_COVERAGE.md)
for recorded backend runs and remaining numerical gaps.
