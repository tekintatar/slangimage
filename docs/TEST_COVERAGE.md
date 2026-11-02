# Public operator test evidence

## How to read this document

This page maps every public operator to the tests that check it and records
what each validation run actually covered. Use it to judge how strongly an
operator, precision or backend has been verified; it is not a guarantee for
arbitrary inputs.

- **Start with the tables.** The texture-operator table below and the CUDA
  tensor/`nn` tables further down are the current evidence map.
- **Dated sections are validation records**, kept for provenance. Each names
  the tests it added, its error budgets, and the backends, dependency versions
  and pass counts of the runs. Later records supersede earlier counts.
- **Fixed defects** are summarized where they were found.
- **Remaining gaps** are listed at the end, under
  [Known gaps and next priorities](#known-gaps-and-next-priorities).

Status (2026-09-13): the planned numerical-reference work is complete through
the blend and same-input cross-backend records. Dates describe individual
validation runs, not a certification of the published release.

Source inventory: 2026-09-08; backend/prewarm evidence, geometry
precision/reference coverage and gamma/Gaussian/unsharp references were
updated afterwards (see the dated records below).
This maps existing tests, not exhaustive numerical coverage. The prewarm check
now enforces operator execution on the selected device at both precisions;
it remains dispatch/storage evidence, not an independent mathematical oracle.
Scope: all **44 image operators** in `ops.__all__`, its three helpers/re-exports,
the nine differentiable operators, nine `nn` wrappers, and three exported effects.
Runtime ownership and threading, installed-artifact and entry-point checks, and
performance are recorded in the later sections of this page.

## Reading the evidence

- **Reference** means a NumPy/torch formula comparison for the stated cases.
  **Invariant** means a narrower mathematical property; neither implies all
  parameter, alpha, border, shape, or backend combinations are covered.
- **S** in the fp16 column means finite/range smoke, not a numerical oracle.
  **Q** means same-backend fp16/fp32 RGB PSNR >40 dB, plus finite/range checks,
  in [precision:test_fp16_matches_fp32][precision]. Only four ops have Q.
- **P** means only prewarm dispatch/storage checks were found beyond any
  evidence listed. These run at both fp16/fp32 on the selected texture backend,
  without checking numerical outputs against a reference.
  **API** means fp16 handle/shape/chain checks in [public][public], on AUTO.
- Test labels below are exact function names within the linked source file.
  Grouped smoke functions exercise several operators in one pytest case.
  Tests through kernel classes cover the corresponding shader path, not every
  functional-wrapper parameter conversion.

## Texture operators (`si.ops`)

All rows have a prewarm path. **D** = direct registry case; **I** = indirect.
The [prewarm registry](../src/slangimage/ops.py) has 44 direct cases for 43
distinct operators (bilateral appears twice); `affine` is reached through
`rotate`. [prewarm:test_prewarm_covers_all_ops][prewarm] runs the registry at
both fp16 and fp32 using the selected `ctx` device. It wraps every exported
operator (excluding the three helpers), verifies that each actually completes,
and checks its output device and storage precision. Adding an export without a
prewarm path fails the set comparison; nested calls count as real execution.
The fixture also asserts that an explicitly requested backend matches the
device returned by the library. No default-device lookup is allowed inside the
explicit-device prewarm check.

This still does not cover every mode or numerical result: nearest tile/affine,
alternate blend modes, and alternate resize modes are not all prewarmed.

Validation on 2026-09-08:

| Device/backend | Prewarm evidence | Regression selection |
|---|---|---|
| Apple M3 Max / Metal | All exports completed at fp16 and fp32 | Full suite: 219 selected-backend/host + 10 AUTO cases passed; 2 CUDA modules excluded |
| NVIDIA L40S / CUDA | All exports completed at fp16 and fp32 | Targeted: 36 passed, 10 AUTO cases deselected; autograd not selected |
| NVIDIA L40S / Vulkan | All exports completed at fp16 and fp32 | Targeted: 36 passed, 10 AUTO cases deselected; 2 CUDA modules excluded |

Both environments used SlangPy 0.42.0 and pytest 9.1.1. Python was 3.12.12 on
Metal and 3.12.3 on Linux; the L40S used driver 580.126.16 and the unchanged
PyTorch 2.13.0 environment. Remote tests ran from an isolated source copy on
the remote host, not a newly built artifact.
The targeted selection was `test_prewarm.py`, `test_public_api.py`,
`test_backend_selection.py`, `test_test_reporting.py`, and the release-device
lifecycle case, with `-m 'not auto_device'`; Vulkan also collected `test_diff.py`
and `test_nn.py` to verify their exclusion. These results add no new numerical
references or CUDA autograd evidence.

| Operator | fp32 reference / invariant / weaker evidence | fp16 checks beyond registry | Prewarm |
|---|---|---|---|
| `gaussian_blur` | [core][core]: `test_gaussian_matches_numpy`, `test_gaussian_amount_matches_numpy`; float64 normalized edge-replicated RGBA convolution, with a closed-form corner-weight check | Same references with both texture-store errors propagated; Q also remains | D |
| `surface_blur` | [complex][complex]: `test_surface_matches_numpy`, independent RGB bilateral sums, sample-count footprint and replicated borders | Same references; Q also remains | D |
| `bilateral` | [bilateral][bilateral]: independent full/1D weighted sums, hand calculations, Gaussian/cutoff and box/Gaussian spatial weights, cutoff neighbors; [neighborhood][neighborhood]: existing invariants | fp16/fp32 full/separable, nonlinear intermediate-storage error propagation, alpha/HDR, tiny images/borders and zero cutoff | D: full + separable |
| `nl_means` | [NLM][nlm]: independent flattened-patch reference, hand calculations, noise compensation, center/h floors, virtual border patches and defaults; [neighborhood][neighborhood]: existing invariants | fp16/fp32, alpha/HDR, search 0/1/2/5, patch 0/1/2, tiny images and repeated calls | D |
| `unsharp_mask` | [core][core]: `test_unsharp_matches_numpy` (RGB and linear-light luma modes, clipping and exact alpha retention) | Same references with blur-storage and nonlinear luma error propagation; Q also remains | D |
| `ep_unsharp_mask` | [complex][complex]: `test_ep_unsharp_matches_numpy`, bilateral base + linear-light YCbCr reference; default detail and public/low-level/preset agreement regressions | Same references with intermediate-storage intervals; default defect fixed | D |
| `sharpen_luma` | [complex][complex]: `test_sharpen_matches_hls_and_cubic`, independent separable cubic weights and stdlib HLS, RGB/HSL modes | Same references, clamping and alpha checks | D |
| `vibrance` | [complex][complex]: `test_vibrance_matches_hls`, `test_vibrance_color_boundaries`, stdlib HLS plus independent skin-mask formula | Same references, including color/amount boundaries; Q remains | D |
| `gamma` | [core][core]: `test_gamma_matches_numpy`, `test_gamma_endpoints_and_direction` (clipped power law, exponent bounds, alpha) | Same reference on quantized input plus one output-storage ULP | D |
| `color_controls` | [color controls][color-controls]: independent linear-light matrix/chroma reference, individual/combined controls, hand calculations; [extra][extra]/[regressions][regressions]: approximate neutral and contrast direction | fp16/fp32, chroma/luma caps, saturated and negative/HDR RGB, alpha, tiny images, public clamps and low-level factor compatibility | D |
| `srgb_to_linear` | [sRGB][srgb]: independent 50-digit Decimal decode, known values, threshold neighbors and tighter linear-branch checks; [extra][extra]: roundtrip/order | fp16/fp32, dark/negative/HDR RGB, exact alpha, channel independence, tiny images | D |
| `linear_to_srgb` | [sRGB][srgb]: independent 50-digit Decimal encode, known values, threshold neighbors and tighter linear-branch checks; [extra][extra]: roundtrip/order | fp16/fp32, dark/negative/HDR RGB, exact alpha, channel independence, tiny images | D |
| `exposure` | [simple color][simple-color]: independent fp32-control exp2/clipping reference and hand values; [adjust][adjust]: existing stops | fp16/fp32 palette/ramp/HDR/random, alpha, tiny images and direct parity | D |
| `sepia` | [simple color][simple-color]: independent clipped matrix/interpolation reference and hand values; [adjust][adjust]: existing matrix check | fp16/fp32 intensity clamps, negative/HDR RGB, alpha, tiny images and direct parity | D |
| `hue_rotate` | [hue][hue]: independent matrix/complex rotation, hand values, signed/multiple turns and clipping boundaries; [adjust][adjust]: approximate neutral/luma | fp16/fp32, saturated/negative/HDR RGB, fp32 angle neighbors, alpha, tiny images, low-level parity | D |
| `false_color` | [false color][false-color]: independent weighted-endpoint reference, hand values, custom/reversed/equal maps and clipping boundaries; [adjust][adjust]: endpoints | fp16/fp32, fp32 endpoint uniforms, saturated/negative/HDR RGB, exact alpha, tiny images and low-level parity | D |
| `highlight_shadow` | [highlight/shadow][highlight-shadow]: independent Decimal curve, hand values, individual/combined controls and activation neighbors; [adjust][adjust]: identity/direction | fp16/fp32, clipping, alpha, tiny images, parameter branches; near-black continuity and dark-curve regressions now pass | D |
| `solarize` | [augmentation][color-aug-numeric]: independent per-channel branch reference; [aug][aug]: existing identity | fp16/fp32 signed/HDR RGB, fp32 thresholds, alpha, tiny images and direct parity | D |
| `posterize` | [augmentation][color-aug-numeric]: independent fp32 8-bit/mask reference and rounding neighbors; [aug][aug]: existing bits | fp16/fp32 bits 1/2/4/7/8, HDR clipping, alpha, tiny images and direct parity | D |
| `grayscale` | [augmentation][color-aug-numeric]: independent rounded-weight dot reference/hand value; [aug][aug]: existing luma | fp16/fp32 negative/HDR RGB, alpha, tiny images and direct parity | D |
| `normalize` | [augmentation][color-aug-numeric]: independent fp32-uniform affine reference | fp16/fp32 signed means/std, negative/HDR RGB, alpha, tiny images and direct parity | D |
| `white_balance` | [white balance numeric][white-balance-numeric]: independent locus, fixed inverse matrices, Bradford adaptation and clipped sRGB pipeline; [white balance][wb]: legacy reference/invariants | fp16/fp32 temperature/tint branches and clamps, palette/gray/HDR/random RGB, exact alpha, tiny images, repeated calls and direct parity | D |
| `equalize` | [equalize][equalize]: rank-based CDF reference, sparse hand calculation, constant/single-bin channels, RGB independence, repeated calls | fp16/fp32, ramps, sparse bins, HDR RGB clipping, exact alpha preservation, tiny images and selected midpoint neighbors | D |
| `autocontrast` | [autocontrast][autocontrast]: per-channel min/max reference, hand calculation, degenerate cutoff, transparent extrema and repeated calls; [aug][aug]: existing NumPy reference | fp16/fp32, constant/narrow ranges, negative/HDR RGB, exact alpha, tiny images; fp32 cutoff neighbors | D |
| `clahe` | [clahe][clahe]: weighted empirical CDF and separable interpolation reference, hand calculations, strength/HDR/alpha checks; [neighborhood][neighborhood]: existing invariants | fp16/fp32 divisible/non-divisible grids and tiny images; all 8 tile-grid regressions now pass | D |
| `resize` | [resampling][resampling]: `test_resize_matches_reference`, `test_resize_large_reduction_and_ringing`, `test_resize_dimension_forms`; independent Lanczos3/Catmull-Rom weights, mode-specific borders and antialiasing | Same fp16/fp32 references, alpha/HDR, tiny images, up/down/mixed scaling; Lanczos intermediate-storage error propagated with absolute weights | D: lanczos3 |
| `flip` | [geometry][geometry]: `test_flip_matches_numpy`, `test_exact_remaps_at_small_image_borders`, `test_functional_geometry_matches_reference` (exact remaps, including alpha) | Same exact references on quantized fp16 input | D |
| `rot90` | [geometry][geometry]: `test_rot90_matches_numpy` (negative/wrapped k), small-image borders, functional references | Same exact references on quantized fp16 input | D |
| `crop` | [geometry][geometry]: `test_crop_matches_slice`, small-image/full-frame/last-pixel crops, functional references | Same exact references on quantized fp16 input | D |
| `tile` | [geometry][geometry]: `test_tile_matches_numpy_with_scale_and_phase`, `test_tile_is_periodic_at_integer_scale`, `test_tile_small_images_and_seams`, functional references (nearest/bilinear) | Same references, including signed/large phases and seam interpolation; per-value storage budget | D |
| `affine` | [geometry][geometry]: `test_affine_identity_and_translation`, `test_affine_matches_numpy`, `test_affine_small_images_and_zero_fill`, functional references (nearest/bilinear); inverse oracle independent of production helper | Same references, including half-pixel borders, scale/shear/reflection, and zero fill; per-value storage budget | I: rotate |
| `rotate` | [resampling][resampling]: `test_rotate_matches_reference` (17/-33/123 degrees, bilinear/nearest, expand on/off), `test_rotate_zero_is_exact`, exact expanded/unexpanded quarter-turn remaps; direct coordinate/tent-function oracle | Same fp16/fp32 references, alpha/HDR and tiny images; expanded-canvas bug fixed, no expected-failure marker | D |
| `kuwahara` | [Kuwahara][kuwahara]: isolated eight-sector reference, angular boundaries, centered variance, color ties and spatial weights; [regressions][regressions]: composed-effect constant invariant | fp16/fp32, borders/tiny images, defaults, alpha/HDR; large-power constant/nonconstant regressions now pass | D |
| `pointillize` | [stylize numeric][stylize-numeric]: independent fp32 hash/nearest-cell sample map away from ties; [neighborhood][neighborhood]: legacy determinism/quantization | fp16/fp32 radius/jitter/seed, clamps, negative/HDR RGB, current-pixel alpha, borders/tiny images, repeated calls and direct parity | D |
| `cmyk_halftone` | [stylize numeric][stylize-numeric]: independent normalized-CMYK separation, rotated screens and multiplicative recombination; [neighborhood][neighborhood]: legacy constant/extreme cases | fp16/fp32 width/angle/sharpness, input/control clamps, HDR inputs, exact alpha, tiny images, repeated calls and direct parity | D |
| `smooth_threshold` | [augmentation][color-aug-numeric]: independent luma/smoothstep reference with propagated luma error; [point][point]: monotonic invariant | fp16/fp32 clipped/control edges, negative/HDR RGB, alpha, tiny images and direct parity | D |
| `smooth_threshold_bw` | [augmentation][color-aug-numeric]: independent two-point luma reference and narrow-gap endpoint regression; [extra][extra]: existing ramp | fp16/fp32 signed/HDR RGB, alpha, tiny images, direct parity; gap defect fixed | D |
| `invert` | [point numeric][point-numeric]: independent unclipped subtraction reference; [point][point]/[pool][pool]: involution/recycling | fp16/fp32 negative/HDR RGB, alpha, tiny images and direct parity | D |
| `extract_channel` | [point numeric][point-numeric]: exact string/integer channel broadcast references | fp16/fp32 negative/HDR values, alpha, tiny images and direct parity | D |
| `blend` | [blend numeric][blend-numeric]: all 12 formulas and shared source-over oracle; earlier focused [alpha][blend-alpha], [soft-light][soft-light], [color][color-mode] and [opaque][point] references | fp16/fp32 and all mixed-input pairs; normalized alpha/opacity, smaller foreground, HDR RGB, branch neighbors, `1e-6` alpha cutoff, repeated/direct parity | D: all modes captured |
| `clamp_multiply` | [point numeric][point-numeric]: independent clipped product and smaller-second border replication references | fp16/fp32 negative/HDR RGB, alpha independence, tiny images and direct parity | D |
| `high_pass_contrast` | [point numeric][point-numeric]: independent unclipped signed-strength formula; [point][point]: self-gray invariant | fp16/fp32 negative/HDR RGB, alpha, tiny images, repeated calls and direct parity | D |
| `lut` | [lut][lut]: `test_lut_matches_volume_reference`, `test_lut_nonconstant_small_images`, `test_lut_ignores_table_alpha_and_preserves_unclipped_rgb`; independent logical-volume trilinear oracle | All four image/table fp16/fp32 combinations; sizes 16/24/40/128/512, slice/cell boundaries, clamping and alpha/HDR; output precision follows image | D |
| `random_noise` | [noise numeric][noise-numeric]: backend-local repeatability, fp32 seed/storage invariants, channel/range/origin, distribution and neighbor correlation; [point][point]: legacy bounds | fp16/fp32 shapes/tiny axes, signed/fractional seeds and direct parity; deliberately no cross-backend bit equality | D |

### Geometry precision batch

The five operators above now run the same deterministic NumPy references at
fp16/fp32, through kernel classes and the public functional API. Coverage includes
non-square and 1x1/1xN/Nx1 images, variable/zero alpha, RGB outside [0, 1], exact
remaps, wrapped bilinear taps, positive/negative and million-pixel phases,
fractional affine borders, half-pixel translations, scale/shear, reflection,
and fully outside canvases. Output device and storage format are asserted.

Reference/error policy in [test_geometry.py][geometry]:

- Quantize the input to its actual storage dtype first, then promote it to
  float64 for reference computation. Do not include input quantization in the
  shader-error budget or perform interpolation in NumPy fp16 arithmetic.
- Exact remaps and nearest sampling require exact values, including alpha.
  Identity/integer affine translations are also exact in both modes.
- Bilinear error is bounded per output value: `3e-5` for fp32, plus one fp16
  output ULP for fp16. No relative tolerance or image-average error metric.
  This budget is for the bounded image values and small canvases in these tests,
  not an error guarantee for arbitrary image sizes or transforms.
- One output ULP, rather than half an ULP, accounts for observed Metal storage
  rounding. In the 1xN tile seam fixture, the fp16 result is `1.0693359375` for
  a reference of `1.070068359375`; the fp32 run on identical quantized input is
  exact. The fp16 value matches rounding toward zero, not rounding to nearest.
- For non-binary inverse coefficients, general nearest-neighbor cases avoid
  coordinates exactly on sampling boundaries. A mathematical zero can become
  slightly negative in fp32 and select zero fill instead of the first pixel.
  Exactly representable transforms separately test the boundary rule; no broad
  pixel tolerance is used to conceal a different selected sample.

This original batch was not a full affine/tiling parameter audit, general-angle
`rotate`/resize reference, or serialized cross-backend golden corpus. The later
parameter and [resampling batch](#resampling-texture-batch) extend that evidence.

Validation on 2026-09-08: the full Metal suite
passed 396 cases (386 selected-backend/host and 10 AUTO), with both CUDA modules
excluded. The geometry file passed all 181 cases on each L40S backend, CUDA and
Vulkan, using `SLANGIMAGE_BACKEND=<backend> python -m pytest -q -ra
tests/test_geometry.py`. Those remote runs selected neither AUTO nor autograd
tests. Environments match the prewarm validation above (SlangPy 0.42.0, unchanged
PyTorch 2.13.0 on Linux); remote validation ran from an isolated source copy.
No kernel or dependency changes were needed for this batch.

### Gamma / Gaussian / unsharp numerical batch

[test_core_numeric.py][core] tests the public texture functions at both fp16
and fp32. Nine input patterns cover constants, ramps, step edges, center/corner
impulses, signed/HDR RGB, and 1x1/1xN/Nx1 images with varying alpha. These are
NumPy references, not comparisons against `si.diff` or shared production helpers.

- **Gamma:** `clip(rgb, 0, 1) ** clip(power, 0.01, 10)`, with exact source alpha.
  Eight finite powers cover both clamp boundaries, brightening, identity within
  [0, 1], and darkening; endpoint/direction invariants are separate checks.
- **Gaussian:** normalized Gaussian convolution of stored RGBA, edge replication,
  horizontal then vertical, without a color-space conversion or clipping.
  Four radius/sigma pairs include a narrow kernel and radius exceeding tiny
  image dimensions; two amount values check the size-aware parameter conversion
  independently. A CPU-only corner-impulse check anchors the reference to the
  closed-form `[1/4, 1/2, 1/4]` convolution rather than another GPU result.
- **Unsharp:** RGB uses `clip(src + amount * (src - blur), 0, 1)`. Luma decodes
  stored source and blurred RGB to linear light, applies the library's rounded
  BT.601-style YCbCr transform, sharpens/clamps Y, retains Cb/Cr, then encodes and
  clips RGB. Both modes retain source alpha exactly. Four strengths include
  negative, zero, moderate, and strong sharpening; radius/sigma differ from the
  defaults to check parameter forwarding.

The new docstrings state previously omitted behavior: gamma clips RGB and bounds
the exponent; Gaussian also blurs alpha; texture unsharp clips RGB whereas
`si.diff.unsharp_mask` does not. The rounded luma color transform is not an exact
identity even with amount zero. These are clarified existing semantics, not
algorithm changes.

Error policy is local to these small, bounded fixtures, not a blanket library
guarantee. References quantize the input first and then compute in float64.
Each Gaussian pass allows `5e-6` fp32 compute error plus one storage ULP; the
horizontal uncertainty is convolved through the vertical pass before its output
store is included. RGB unsharp scales blur uncertainty by `abs(amount)`.
Luma unsharp propagates blur intervals through the monotone sRGB transfer curves
and positive luma weights, with a `2e-6` linear-light arithmetic allowance scaled
by strength, before clipping and final storage. This captures near-black
amplification without an image-wide loose tolerance. Alpha preservation for
gamma/unsharp is exact. All output shapes, devices, precisions, and finite values
are checked, with absolute per-value bounds and no PSNR/relative-error fallback.

Remaining scope: non-finite/extreme parameter policy, arbitrary image/radius
scales, generalized higher-order/differentiable behavior, and wider color-science
contracts are not certified by this batch. The later complex batch below covers
surface blur, vibrance, edge-preserving unsharp and sharpen-luma, and records
defects that were subsequently fixed.

Validation on 2026-09-08: the new file passed
399 cases on each of Metal, CUDA, and Vulkan (398 GPU cases plus one CPU oracle
check). The full Metal suite passed 795 cases: 785 selected-backend/host and 10
AUTO, with the two CUDA modules excluded. Remote runs selected only
`tests/test_core_numeric.py`, not CUDA autograd tests, using an isolated source
copy and the unchanged SlangPy 0.42.0 / PyTorch 2.13.0 L40S environment described
above. No algorithm mismatches were found in these fixtures; runtime changes
are docstrings only. Lint and coverage-link checks also passed.

### Complex texture batch

[test_complex_numeric.py][complex] adds independent references for the
remaining operators in this batch, using the preceding batch's quantized
fp16/fp32 fixtures: constants, ramps, steps, impulses/corners, signed/HDR RGB,
variable alpha, non-square images and 1x1/1xN/Nx1 shapes.

- Surface blur: float64 RGB/spatial Gaussian weights and edge-replicated sums;
  odd/even sample counts, tiny footprints, range sigmas and footprints exceeding
  image dimensions. The legacy `radius` is a sample-count control, not a pixel
  radius. The 1/2 cases now assert exact identity via an explicit shader return,
  not undefined-weight fallback. Broader invalid-parameter policy is not certified.
- EP unsharp: independent surface base and rounded linear-light YCbCr transform;
  zero/negative/positive strengths and propagated intermediate-storage error.
  Only Y is clipped, not returned RGB. The default activity check compares
  against amount zero, avoiding a false pass from color-roundtrip drift, and
  also checks the new default against the independent 3x3-base formula.
  Public, explicit-radius, low-level and preset defaults agree in both precisions.
- Vibrance: standard-library HLS with a separately specified RGB skin mask;
  negative/zero/positive/clamped amounts, primaries, grays, dark/near-white and
  low-chroma boundaries. Sharpen-luma uses independently factored cubic weights
  and separable NumPy sums, testing both RGB and HSL lightness modes, fractional
  radii, footprint cap, and strength/radius clamps.

Arithmetic budgets are fixture-scoped absolute bounds: surface/vibrance use
`5e-6` and cubic sharpening `5e-6*(1+abs(strength))` (twice that in HSL mode),
plus one per-value output storage ULP. EP propagates surface-storage intervals
through monotone transfers and positive luma weights, as in the earlier
luma-unsharp reference. These are not arbitrary-resolution or
arbitrary-magnitude guarantees.

The defects found in this batch are now fixed. Default EP radius is
now 3 (a 3x3 footprint), with explicit-radius semantics unchanged. Surface
one-pixel/empty footprints return the center before computing weights. The
former expected failures are ordinary required regressions now; no xfail
remains here.

Remaining gaps include non-finite/invalid-parameter policy, extreme scales and
general color science. CUDA autograd is outside this texture batch.

Initial pre-fix validation on 2026-09-08 (historical results, superseded by the
post-fix validation below):

- Full Metal suite: **1777 passed, 4 xfailed, 2 skipped collection nodes**.
  Groups: 1767 selected-backend/host passes plus the four known xfails, 10 AUTO
  passes, and the two CUDA autograd modules excluded. Apple M3 Max, Python
  3.12.12, SlangPy 0.42.0.
- Pre-fix targeted CUDA and Vulkan runs: **982 passed, 4 xfailed each**, selecting
  only `tests/test_complex_numeric.py` (981 passing GPU cases and one passing
  CPU oracle check; all four known xfails are GPU cases). NVIDIA L40S, driver
  580.126.16, Python 3.12.3, SlangPy 0.42.0, PyTorch 2.13.0+cu130 unchanged.
  Requested/actual devices agree. No CUDA autograd tests ran in either selection.
- Remote runs used an isolated source copy on the remote host, with imports
  verified. No editable install, dependency upgrade, or installed-artifact
  claim.
- Lint, diff whitespace checks, and coverage-link checks passed. At that stage,
  runtime changes were documentation/comments only and both defects remained
  unresolved. Those xfails were not a clean release gate.

Post-fix validation on 2026-09-08, with the corrections:

- **1044 complex-reference cases pass on Metal**, including all four former
  expected failures and 58 added scale/boundary/entry-point cases. No xfails.
- **Full Metal suite: 1839 passed, 2 skipped collection nodes** — 1829
  selected-backend/host passes, 10 AUTO passes, CUDA autograd modules excluded.
- **CUDA and Vulkan: 1110 passed, 1 deselected each**, selecting
  `test_complex_numeric.py`, `test_prewarm.py`, `test_colorspace_and_extra_ops.py`,
  `test_regressions.py`, `test_pool.py`, and `test_smoke.py`, with
  `-m 'not auto_device'`. This includes all 1044 complex cases plus affected
  prewarm, effect, parameter and lifetime checks. The deselected test is an
  intentional AUTO-device lifecycle case. No CUDA autograd was selected.
- Remote stack unchanged: L40S / driver 580.126.16, SlangPy 0.42.0,
  PyTorch 2.13.0+cu130. Run from an isolated source copy, with imports
  verified. No installed-artifact claim.
- Lint, diff whitespace and coverage-reference link checks pass. These two
  defects are resolved; broader gaps are listed at the end of this page.

### Public parameter-validation batch

[test_parameter_validation.py][parameters] enforces the
[texture parameter contract](PARAMETER_CONTRACTS.md): every public scalar
numeric/boolean control has host-only NaN/Inf/fp32-overflow rejection checks,
with a signature inventory that fails when a newly added control is omitted.
Additional checks cover vector/grid shape, fractional integer controls,
conflicting parameter forms, zero/negative domains, weight-sigma extremes,
collapsed fp32 threshold edges, oversized extents/histogram indices, singular
or non-finite affine matrices, and unsafe inverse-scaled coordinates.

The file has **544 cases: 540 host checks and 4 selected-backend GPU cases**.
The GPU cases each run several bounded operations at fp16/fp32, checking that
intentional clamps, box/cutoff/zero-search sentinels, negative normalization
divisors and minimum-one-pixel resize behavior remain valid. The preceding
numerical references continue to run unchanged.

The numerical/encoding bounds are not wall-clock or memory budgets. Host stubs
ensure dangerous values do not trigger output allocation/dispatch; these tests
never try an enormous allocation or hung GPU loop to prove that it fails.
Remaining scope: image/LUT pixel NaN policy, directly imported kernel/effect
entry points not sharing these validators, resource budgets and CUDA autograd
contracts. These remain separate from functional-API parameter validation.

Validation on 2026-09-08, with the preceding fixes:

- **Full Metal: 2383 passed, 2 skipped collection nodes** (2373 selected-backend/
  host, 10 AUTO; CUDA autograd modules excluded). Apple M3 Max, Python 3.12.12,
  SlangPy 0.42.0.
- **CUDA and Vulkan: 2373 passed, 10 deselected each**, running all `tests/` with
  `-m 'not auto_device' --ignore=tests/test_diff.py --ignore=tests/test_nn.py`.
  These are full selected texture/host selections, not CUDA gradient evidence.
  NVIDIA L40S / driver 580.126.16, Python 3.12.3, SlangPy 0.42.0,
  PyTorch 2.13.0+cu130 unchanged. No expected failures in any run.
- Remote runs used an isolated source copy, with imports verified. No
  installed-artifact claim or remote dependency upgrade.
- Ruff, the configured mypy typing check, whitespace and coverage-link checks
  passed. Existing numerical references remain passing alongside the new guards.

### Resampling texture batch

Added 2026-09-08 in
[test_resampling_numeric.py][resampling]. These are texture-path tests, not
`si.diff.resize` or CUDA gradient coverage. The initial pass changed only tests
and comments; the follow-up fixes exact-quarter-turn matrix coefficients.
Shader sampling and arbitrary-angle behavior remain unchanged.

- **Resize:** independent dense float64 weight matrices for Lanczos3 and
  Catmull-Rom bicubic. Lanczos widens support when reducing and normalizes only
  in-bounds taps; bicubic has a fixed 4x4 footprint with repeated edge pixels.
  Nine input patterns cover constants, ramps, steps, center/corner impulses,
  HDR, 1x1, 1xN and Nx1. Output cases include up/down/mixed scaling, 1x1 and
  identity; a 95x63 checkerboard reduced to 11x7 tests widened support.
  Lanczos alone is asserted to antialias. Separate step tests require negative
  and above-one ringing, including alpha: neither filter clips or premultiplies.
  Scale/single-dimension forms test ties-to-even and minimum-one-pixel rounding.
- **Rotation:** direct trigonometric inverse coordinates, without production
  rotation-matrix/inversion helpers. Bilinear uses a dense source-center tent
  sum, nearest requires exact selected values. Angles 17, -33 and 123 degrees
  test both modes and expansion settings across all nine patterns/precisions.
  Expanded dimensions and sampling anchor follow the exact rotated rectangle's
  bounds, not the center of the integer-rounded canvas. Zero angle is exact.
  Arbitrary-angle nearest cases assert a >2e-5 coordinate distance from integer
  sampling boundaries; they do not certify fp32 tie-breaking at those boundaries.
- **Error policy:** input is quantized before float64 reference evaluation.
  A `3e-5` fp32 arithmetic allowance per pass applies to these bounded inputs
  and dimensions, not arbitrary images. One output-storage ULP includes Metal's
  observed toward-zero behavior. Lanczos horizontal arithmetic/storage errors
  propagate through the **absolute** vertical weights, since negative lobes
  could otherwise cancel uncertainty. Bicubic has no intermediate store.
  No relative tolerance, clipped reference or image-wide average metric.
  Host closed-form checks pin identity, single-pixel, cubic half-phase/edge
  weights and rotation direction.

**Resolved defect — expanded exact-quarter-turn canvas:** for a 17x13 input, `rotate`
at 90 degrees with `expand=True` returned 14x17, not 13x17. Float64 trigonometric
residuals inflated a bound before `ceil`; 180/270/360/-90/450 degrees were affected
too. Initially deferred, this is now fixed using exact coefficients only for
exact multiples of 90 degrees. The strict expected-failure
marker has been removed. The expanded NumPy `rot90` regression now covers 240
cases: 12 angles (including negative/periodic/large turns), five shapes (odd/even
non-square, 1x1, 1xN, Nx1), both modes and both precisions. Another 48 cases pin
unexpanded square remaps, and 10 host cases verify unchanged arbitrary/nearby
angle coefficients.

Initial, pre-fix validation on 2026-09-08, with the resampling tests/docs
(historical counts; post-fix results follow below):

- **New batch: 490 passed, 24 strict expected failures on each backend**
  (488 texture cases and two host oracle checks pass). The 24 failures all
  reproduce the deferred expanded-canvas defect; there are no other failures.
- **Full Metal: 2873 passed, 24 xfailed, two skipped CUDA collection nodes**;
  2863 selected texture/host checks plus 10 intentional AUTO checks.
  Apple M3 Max, Python 3.12.12, SlangPy 0.42.0.
- **Full selected CUDA and Vulkan: 2863 passed, 24 xfailed, 10 deselected each**.
  Both ran `python -m pytest -q --tb=short tests/ -m 'not auto_device'
  --ignore=tests/test_diff.py --ignore=tests/test_nn.py`, with explicit
  `SLANGIMAGE_BACKEND` and a 180-second process timeout. NVIDIA L40S, driver
  580.126.16, Python 3.12.3, SlangPy 0.42.0, PyTorch 2.13.0+cu130 unchanged.
  Neither run selected CUDA autograd; Vulkan results are texture/host only.
- Remote runs used an isolated source copy. This is source-tree validation,
  not an installed-artifact or dependency-version-matrix claim.
- Ruff, configured mypy, whitespace and coverage links passed. AST comparison
  (ignoring docstrings) and shader comment stripping confirm no executable
  production changes. Remote dependencies were unchanged.

Post-fix validation on 2026-09-08, with the quarter-turn fix and expanded
regressions:

- **Resampling file: 788 passed on each backend**, including 288 exact
  quarter-turn remaps and 10 non-quarter-angle compatibility checks. The former
  24 expected failures are replaced by passing, broader coverage; no xfails remain.
- **Full Metal: 3171 passed, two skipped CUDA collection nodes** (3161 selected
  texture/host checks, 10 AUTO). **Full selected CUDA and Vulkan: 3161 passed,
  10 deselected each**. Same selection commands and backend stacks as above;
  CUDA autograd is excluded from both remote runs, with PyTorch kept at 2.13.
- Remote runs used an isolated source copy, with imports verified. No
  dependency upgrades or installed-artifact validation were performed.
- Ruff, configured mypy, whitespace and coverage links passed. The only
  executable production change is exact-quarter-turn coefficient selection;
  arbitrary-angle math, shader sampling and texture allocation are unchanged.

### LUT texture batch

Added 2026-09-08 in
[test_lut_numeric.py][lut]. These checks cover the texture `si.ops.lut`, not
`si.diff.lut`, `si.nn.Lut` or gradients with respect to the table. The initial
LUT pass changed only comments/docstrings; the later upload normalization fix
is recorded below. The LUT shader algorithm is unchanged.

- **Independent oracle:** a logical `[blue, green, red, channel]` volume,
  evaluated by eight-corner product weights. It does not reuse the shader's
  normalized 2D atlas-coordinate or nested bilinear calculations, the public
  identity generator, or the tensor implementation. Host checks pin a
  cross-channel multiaffine polynomial, a 1/8-weight corner impulse and atlas
  row-wrap placement. Five additional host checks verify the public identity
  generator against direct atlas pixel coordinates.
- **Tables/storage:** cross-channel multiaffine, nonlinear and high-contrast
  random tables at atlas sizes 16, 24, 40, 128 and 512 (including non-power-of-two
  sizes). There are always 64 blue slices; red/green resolution is `size/8`.
  Every combination of fp16/fp32 image and table storage is checked. Both are
  quantized before float64 reference computation; output precision/device must
  follow the source image, not the table.
- **Samples:** cube corners, every blue node/midpoint, atlas row wraps,
  representable neighbors of internal R/G/blue nodes, random/HDR coordinates,
  and alpha below zero/inside/above one. Intensities -2/0/.25/.7/1/2 pin the
  control clamp. Nonconstant 1x1, Nx1 and non-square images supplement the
  boundary corpus's 1xN image. The raw-texture API is checked too.
- **Actual contract:** clamp input RGBA and intensity to [0,1], linearly
  interpolate mapped RGB, ignore table alpha, and retain the clamped source
  alpha. Intensity zero is exact **clamped-input** passthrough, not preservation
  of HDR/negative source values. Table RGB/output RGB are not clipped; a
  constant table verifies negative and above-one output values with
  different table alphas (changing table alpha must have exactly zero effect).
  These tests document existing behavior, not a new
  alpha policy.
- **Error policy:** per-channel absolute bounds. With `eps = float32 epsilon`,
  coordinate uncertainty is budgeted as `4*eps*atlas_size` texels in R/G and
  `2*eps*63` in blue. Multiply by each axis's maximum adjacent-table-node
  difference and the clamped intensity, add `5e-6` arithmetic allowance and
  one output-storage ULP. Smooth tables have tighter bounds than random ones.
  These allowances apply to this bounded-value/size corpus, not arbitrary
  HDR tables or atlas dimensions; no PSNR, relative tolerance or input
  quantization double-counting. Alpha and zero-intensity cases require exact
  equality. Pixel NaN/Inf policy and same-input cross-backend comparisons
  remain separate work.
  Constant-table RGB also allows subtraction/addition rounding in the final
  intensity-one mix `c + (mapped-c)*1`: CUDA showed a maximum `5.96e-8` error
  against -0.25 in the initial exact-value assertion. The corrected test uses
  `2*eps*(abs(mapped-c)+abs(mapped))` plus one output ULP, not exact RGB equality;
  alpha and independence from table alpha still require exact equality.

The initial fixture exposed a separate fp32 zero-stride singleton upload
failure, subsequently reproduced on Metal, CUDA and Vulkan before LUT dispatch.
The initial LUT corpus used nonzero strides to isolate interpolation. The
[upload correction](#singleton-stride-upload-correction) below resolves the
finding, and LUT
tests now use the original zero-stride view directly.

Initial LUT validation on 2026-09-08, with the LUT tests/docs
(before the upload correction):

- **New LUT file: 378 passed on each backend** (372 texture cases, six host
  checks). No expected failures. No LUT interpolation defect was found within
  this bounded corpus.
- **Full Metal: 3549 passed, two skipped CUDA collection nodes** (3539 selected
  texture/host, 10 AUTO). Apple M3 Max, Python 3.12.12, SlangPy 0.42.0.
- **Full selected CUDA and Vulkan: 3539 passed, 10 deselected each**. Both ran
  `python -m pytest -q --tb=short tests/ -m 'not auto_device'
  --ignore=tests/test_diff.py --ignore=tests/test_nn.py`, with explicit backend
  selection and a 180-second timeout per process. NVIDIA L40S, driver
  580.126.16, Python 3.12.3, SlangPy 0.42.0, PyTorch 2.13.0+cu130 unchanged.
  No CUDA autograd tests selected, including in the Vulkan run.
- Remote runs used an isolated source copy, with imports verified. This is
  source-tree validation, not an installed-artifact claim.
- Ruff, configured mypy, whitespace and coverage links passed. Python AST
  comparison ignoring docstrings and shader comment stripping confirm no
  executable production changes. Dependencies were unchanged.

### Singleton-stride upload correction

The follow-up fix normalizes byte strides after RGB expansion and dtype
conversion. NumPy permits arbitrary strides on length-one axes even when an
array is C-contiguous, but SlangPy needs canonical strides. Flattening and
reshaping the contiguous buffer fixes zero, negative and oversized singleton
strides without copying writable fp32 pixel storage or changing caller metadata.
Already canonical writable fp32 uploads retain their original array object.
Read-only buffers receive a defensive copy because SlangPy's ndarray binding
requires writable data. Existing fp16 conversion behavior is unchanged.

[test_textures.py][textures] adds 192 regressions:

- 176 GPU checks: row/column/pixel new-axis views, negative/oversized singleton
  strides, ordinary contiguous inputs, transposes, reversals and stepped slices;
  both input dtypes, RGB/RGBA, both output precisions, writable/read-only. Exact
  quantized HDR/alpha roundtrips and RGB opaque-alpha expansion are required.
- 16 host checks: capture the buffer passed to SlangPy and assert canonical
  byte strides, writable upload storage, sharing for writable contiguous views,
  defensive copying for read-only views, and unchanged caller values/flags/
  strides. Ordinary writable fp32 input must retain object identity.
- The 378-case LUT corpus now uses the formerly failing zero-stride input view
  directly, with no workaround or expected-failure marker.

Validation on 2026-09-08, with the LUT batch and upload fix:

- **Full Metal: 3741 passed, two skipped CUDA collection nodes** (3731 selected
  texture/host, 10 AUTO). **Full selected CUDA and Vulkan: 3731 passed,
  10 deselected each**, using the same explicit-backend selection/180-second
  timeout and Python/SlangPy/GPU stacks recorded for the LUT batch above.
  PyTorch remains 2.13; neither remote run selected CUDA autograd. No expected
  failures. All 192 new upload regressions and the 378 LUT cases pass.
- Remote runs used an isolated source copy.
  Source-tree validation only, without dependency upgrades or an installed-
  artifact claim. Ruff, configured mypy, whitespace and coverage links passed.

### Blend alpha/opacity references

[test_blend_alpha.py][blend-alpha] adds 22 cases: two hand-calculated oracle
checks and 20 texture cases (multiply/screen × fp16/fp32 × five opacities).
Every texture case covers all 16 combinations of base/foreground alpha
0/.25/.75/1, with nonzero hidden RGB on transparent pixels. The independent
reference partitions coverage into background-only, foreground-only and overlap,
then converts the weighted premultiplied sum back to straight RGB. Opacity
-1/0/.375/1/2 tests clamping and foreground-alpha scaling, not RGB scaling.
Inputs are quantized before float64 evaluation; alpha and all-transparent output
are exact. RGB allows `5e-6` arithmetic error plus one output-storage ULP.
The budget is limited to this [0,1] corpus (positive output alpha >=.09375).

Scope deliberately excludes other blend formulas (including soft-light/color),
HDR/out-of-range alpha, the <=1e-6 output-alpha cutoff, different input sizes,
mixed precision, autograd and performance. Only the public docstring changes;
no shader/runtime behavior changes. This is not a complete blend audit.

Validation 2026-09-08: **46 passed on each of Metal, CUDA and Vulkan**, selecting
only `tests/test_blend_alpha.py tests/test_blend_and_point_ops.py` (22 new checks
plus 24 existing tests). No failures, expected failures or autograd selection.
Same SlangPy 0.42.0 / PyTorch 2.13 stacks as the preceding pass; remote runs
used an isolated source copy, with a 120-second timeout per backend. Ruff,
configured mypy and whitespace/link checks passed. No full-suite or
installed-artifact rerun in this pass.

### Soft-light references

[test_blend_soft_light.py][soft-light] adds 16 cases: two host hand-calculated
oracle checks, eight fp16/fp32 boundary/compositing cases and six neutral-gray
checks. The independent float64 reference interpolates the foreground black,
gray and white anchors; the white anchor uses a polynomial below base .25 and
square root above it. The corpus includes base .25 / foreground .5 and their
immediate neighbors in each input storage dtype, with channel permutations.
It covers opaque layers, one partial-alpha/opacity combination, transparent
background and transparent foreground. Neutral-gray identity is asserted only
over an opaque base; partial backgrounds also receive foreground-only gray.

Inputs are quantized before reference evaluation. Output alpha is exact; RGB
allows `5e-6` arithmetic error plus one storage ULP for this [0,1] corpus, where
positive output alpha is >=.5. No production changes. Color mode, HDR, tiny-alpha
cutoff, mixed storage, different image sizes, autograd and performance remain
out of scope; this does not certify the whole blend implementation.

Validation 2026-09-09, with these tests/docs: **62 passed on each of
Metal, CUDA and Vulkan**, selecting `test_blend_soft_light.py`,
`test_blend_alpha.py` and `test_blend_and_point_ops.py` (16 new + 46 existing).
No failures, expected failures or CUDA autograd selection. SlangPy 0.42.0,
PyTorch 2.13.0+cu130 unchanged; remote runs used an isolated source copy,
120-second timeout per backend.
Ruff, configured mypy and whitespace/link checks passed. No full-suite,
installed-artifact or performance rerun in this pass.

### Blend color-mode references

[test_blend_color.py][color-mode] adds 14 cases: two host hand calculations and
12 fp16/fp32 texture cases. The independent reference keeps background weighted
luminosity (.30/.59/.11), then moves along the foreground chroma direction from
neutral gray as far as all channels permit in [0,1]. This avoids the shader's
sequential lower/upper ClipColor implementation. It is equivalent on normalized
inputs, not a promise for arbitrary HDR or a literal HSL conversion.

The palette spans black/white/gray, primaries/secondaries, non-neutral backgrounds
and intermediate colors. Gray backgrounds sample both gamut boundaries of
(.75,.25,.25), including adjacent representable fp16/fp32 levels around .15/.65.
Cases cover opaque inputs, one partial-alpha/opacity combination, transparent
background/foreground, zero opacity and both layers transparent. Opaque output
luminosity is independently checked; partial-alpha output is checked against
source-over, not incorrectly required to retain background luminosity.

Both inputs are quantized before float64 evaluation. Alpha and fully transparent
output are exact; RGB allows `5e-6` plus one output-storage ULP on this [0,1]
corpus (positive alpha >=.5). Only comments/docstrings change. HDR/out-of-range
alpha, near-degenerate epsilon behavior, tiny-alpha cutoff, mixed precision,
different image sizes, autograd and performance remain excluded.

Validation 2026-09-09, with the soft-light/color changes:
**76 passed on each of Metal, CUDA and Vulkan**, selecting `test_blend_color.py`,
`test_blend_soft_light.py`, `test_blend_alpha.py` and `test_blend_and_point_ops.py`
(14 new + 62 existing). No failures, expected failures or autograd selection.
SlangPy 0.42.0 / PyTorch 2.13.0+cu130 unchanged; remote runs used an isolated
source copy, with a 120-second timeout per backend.
Ruff, configured mypy and whitespace/link checks passed. No full-suite,
installed-artifact or performance rerun in this pass.

### Global equalization references

[test_equalize_numeric.py][equalize] adds 24 cases: two host hand calculations,
20 fp16/fp32 pattern cases and two alpha/repeated-call cases. The independent
reference sorts per-channel bin assignments and uses sample ranks to obtain
the CDF, rather than reimplementing the shader's histogram/prefix-sum pipeline.
It subtracts the first occupied-bin count; a single occupied bin maps to
`bin/255`, not exact input passthrough. RGB is clamped before binning, and all
pixels count regardless of alpha. Alpha itself is preserved without clipping.

Patterns cover constants, multiple values in one bin, ramps, uneven sparse
histograms, channel-independent distributions, negative/above-one RGB, 1x1,
1xN and Nx1. The exactly representable .5 midpoint and its stored neighbors
pin half-up binning. Inputs are quantized before float64 evaluation; fixture
bin assignments must agree under ideal, fused-fp32 and unfused-fp32 arithmetic.
Other rounding-sensitive bin boundaries are not certified or hidden with a
pixel-error tolerance. Output uses `5e-7` arithmetic allowance plus one storage
ULP; alpha is exact, and RGB must stay in [0,1]. Counts here are small/exact.

No algorithm changes; only the equalize docstring/comments are clarified.
CLAHE, a deeper autocontrast pass, nonfinite pixels, very large counts and
performance remain outside this pass.

Validation 2026-09-09, with these tests/docs: **35 passed on each of
Metal, CUDA and Vulkan**, selecting `test_equalize_numeric.py` and
`test_color_aug.py` (24 new + 11 existing). No failures, expected failures or
autograd selection. SlangPy 0.42.0 / PyTorch 2.13.0+cu130 unchanged; remote
runs used an isolated source copy, 120-second timeout per backend. Ruff,
configured mypy and whitespace/link checks passed. No full-suite,
installed-artifact or performance rerun.

### Autocontrast references

[test_autocontrast_numeric.py][autocontrast] adds 24 cases: one host hand
calculation, 20 fp16/fp32 pattern cases, one fp32 cutoff case and two
transparent-extrema/repeated-call cases. The reference evaluates per-channel
min/max scaling in float64 after input-storage quantization. Constants,
negative/HDR values, independent channel ranges, a few storage ULPs around .5,
outliers, ramps, 1x1, 1xN and Nx1 are covered. All pixels contribute to extrema,
regardless of alpha; input RGB is not clamped.

Ranges at or below the fp32 representation of `1e-12` pass through exactly.
The cutoff case checks its adjacent representable spans; fixtures require
float64 and fp32 subtraction to agree on the branch. Alpha and degenerate RGB
are exact, including out-of-range values. Active RGB allows `5e-7` arithmetic
error plus one output-storage ULP. Endpoint checks also allow arithmetic error:
CUDA fast division produced one fp32 step below 1 at the upper endpoint of
the just-above-cutoff span; the initial bit-exact assertion was corrected,
without changing the reference error budget or production algorithm.

Only tests/docs/comments change. Nonfinite pixels, overflow-scale fp32 ranges,
subnormal behavior, CLAHE and performance remain outside this pass.

Validation 2026-09-09, with the equalize/autocontrast changes:
**59 passed on each of Metal, CUDA and Vulkan**, selecting
`test_autocontrast_numeric.py`, `test_equalize_numeric.py` and `test_color_aug.py`
(24 new + 35 existing). Final runs have no failures, expected failures or
autograd selection. SlangPy 0.42.0 / PyTorch 2.13.0+cu130 unchanged; remote
runs used an isolated source copy, with a 120-second timeout per backend.
Ruff, configured mypy and whitespace/link checks passed. No full-suite,
installed-artifact or performance rerun in this pass.

### CLAHE references

[test_clahe_numeric.py][clahe] adds 49 cases: three host hand calculations and
46 fp16/fp32 texture cases, including eight regressions for the now-fixed
tile-grid defect. The reference evaluates the weighted empirical CDF
of clipped sample multiplicities plus uniform redistributed mass, avoiding the
shader's sequential prefix loop. Spatial mapping uses NumPy's separable linear
interpolation at nominal histogram-tile centers with constant edge extension.
Hand calculations pin clipping, fractional bin caps and interpolation weights.

The passing corpus covers constants, black/near-black, chromatic and spatially
distinct populations, negative/HDR RGB, grids (1,1)/(3,2), clip limits
0.25/3/32/256, strengths -1/0/.375/1/2, 1x1/1xN/Nx1 with fitting grids, and
alpha-independent histograms across repeated calls. Luminance is clamped before
nearest-bin assignment; clipped counts are fractional, redistributed once, and
the CDF is not offset by its first occupied-bin count. Final RGB is clamped even
at zero strength; alpha is preserved exactly, including out-of-range values.
This is weighted RGB luminance, not a Lab/OpenCV parity contract.

Inputs are quantized before float64 evaluation. A conservative fp32 arithmetic
interval must leave each chosen luminance bin unambiguous; discrete bin errors
are not covered by RGB tolerance. Passing RGB permits `1e-5` arithmetic error
plus one storage ULP on this small-count corpus. Nonfinite/overflow-scale data,
large-count precision, ambiguous half-bin boundaries, exact dark-ratio cutoff
neighbors, performance and exhaustive parameter combinations remain excluded.

**Resolved defect:** the shader previously used ceil-sized histogram footprints
but fractional interpolation spacing and could blend empty tiles. The fix uses
the same ceil-sized nominal grid for both passes, with occupied-tile edge
extension. Mapping buffer strides, histogram clipping, CDF math and RGB/alpha
policy remain unchanged. The eight regressions now pass without
expected-failure markers.

Initial pass 2026-09-09, with these tests: **44 passed, 8 xfailed on each
of Metal, CUDA and Vulkan**, selecting `test_clahe_numeric.py` and
`test_neighborhood_ops.py -k clahe` (10 unrelated cases deselected). No autograd
selection. SlangPy 0.42.0 / PyTorch 2.13.0+cu130 unchanged; remote runs used
an isolated source copy, 120-second timeout per backend.
Ruff, configured mypy and whitespace/link checks passed.

After the fix, the same selection is **52 passed on each of Metal,
CUDA and Vulkan**, with 10 unrelated cases deselected and no xfails/autograd
selection. Remote runs used an isolated source copy, with the same
dependencies and timeout. No full-suite, installed-artifact or
performance rerun in this pass.

Visual follow-up used the pre-fix shader versus the fixed shader on
the same Metal device and stored inputs: three local photos (landscape
2048x1365, building 2720x3400, portrait 1024x1024), native and <=63-pixel-long-edge
thumbnails, grids 8x8/24x24, clip limit 3, strength 1, both fp16/fp32 (24 pairs).
Metrics are computed on lossless float readback before display conversion;
PNGs are display-only and difference panels amplify max-channel error by 8.
Representative fp16 mean absolute RGB differences, in 1/255 units:

| Sample | Native 8x8 | Native 24x24 | Thumbnail 24x24 |
|---|---:|---:|---:|
| Landscape | 0.167 | 1.467 | 55.996 |
| Building | 0 (bit-identical) | 0.731 | 45.267 |
| Portrait | 0 (bit-identical) | 2.706 | 44.356 |

Both precisions gave the same qualitative conclusions; the divisible controls
were bit-identical at both precisions. At native 24x24, 14.19% of landscape and
25.69% of portrait pixels changed by >5/255 in at least one channel. Visual
inspection found localized tonal changes on those images, subtler changes on
the building, and conspicuous thumbnail changes. This is an intentional output
compatibility change, not a promise of universal perceptual improvement.
Visual checks were on Metal, not a cross-backend image parity claim. Other
backends ran the numerical corpus.

### sRGB transfer references

[test_srgb_numeric.py][srgb] adds 50 cases: two host known-value checks and
48 fp16/fp32 texture cases across both directions. Each direction is evaluated
independently using scalar Decimal arithmetic at 50 digits, then rounded to
float64 for comparison. Inputs are quantized first; no production conversion
helper or opposite-direction roundtrip is used as the oracle. The normalized
equations are sourced from the [ICC sRGB specification notes](https://registry.color.org/rgb-registry/files/sRGB.pdf).
The reference selects branches at the shader literal's fp32 cutoff; it does
not assume that a rounded fp16 threshold is the exact mathematical join.

Cases cover both cutoffs and two adjacent stored values on each side,
black/near-white/above-white endpoints, dark values (including fp16 subnormal
outputs), negative and above-one RGB, ramps, deterministic mixed-channel
samples, and 1x1/1xN/Nx1. Alpha is exact even outside [0,1], transparent pixels
have unchanged RGB processing, and channel permutation gives exact matching
permutations. Negative values use the linear branch at every magnitude in
the corpus; HDR values follow the unbounded power branch. This extension is
library behavior, not a sign-mirrored extended-sRGB standard claim.

For bounded RGB [-2,4], general output error is limited per value to
`2e-7 + 2e-6*abs(reference)` plus one output-storage ULP. Dedicated low-branch
cases, including the inclusive fp32 cutoff and the highest fp16 value on its
low side, use two fp32 arithmetic ULPs plus one storage ULP; zero is exact.
General threshold-neighbor comparisons allow power-function error, not
bit-exact high-branch results. Rounded joins/storage mean these are approximate
inverses, not globally exact roundtrips. Nonfinite/overflow-scale inputs, fp32
subnormals, exhaustive half-value enumeration, full color-management/profile
conversion, other internal conversion helpers, autograd and performance are
outside this pass. Only tests/docs/docstrings change; no shader math changes.

Validation 2026-09-09, with this batch: **57 passed on each of Metal,
CUDA and Vulkan**, selecting `test_srgb_numeric.py` and
`test_colorspace_and_extra_ops.py` (50 new + 7 existing). No failures, expected
failures or autograd selection. SlangPy 0.42.0 / PyTorch 2.13.0+cu130 unchanged;
remote runs used an isolated source copy, 120-second timeout per backend.
Ruff, configured mypy and whitespace/link checks passed. No full-suite,
installed-artifact, visual or performance rerun in this pass.

### Bilateral references

[test_bilateral_numeric.py][bilateral] adds 103 cases: three host oracle checks,
56 full/separable pattern cases, 24 tiny/border cases, four alpha-independence
cases, 12 cutoff-neighbor cases and four zero-cutoff cases. Both fp16 and fp32
use independent float64 neighborhood tensors and normalized RGB-weighted sums.
Hand calculations pin replicated corner multiplicities and combined spatial/
range Gaussian weights. A nontrivial fixture also ensures the full-2D and
separable references differ by much more than their allowed error.

The corpus covers radii 1/2/4, box and sigma_spatial=1.5 Gaussian weighting,
sigma_range .35 Gaussian and .25 cutoff, constants, impulses, ramps,
deterministic random samples, separated low-variation color clusters, negative/
above-one RGB, 1x1/1xN/Nx1 and independent alpha values in [-.25,1.25]. Alpha is
exact and ignored by range weights. The dedicated binary fixtures use a .5
red-channel distance and adjacent stored values to pin inclusive cutoff
behavior; the rejected zero pixel is exact. Repeated normalized averaging of
the nonzero pixel is not required to be bit-exact (Metal fp32 showed one ULP
of rounding in the initial over-strict assertion). Zero cutoff accepts only
equal RGB samples and is checked against identity with per-pass error bounds.

Separable is horizontal then vertical, with a same-precision texture store in
between and vertical range weights computed from the stored intermediate.
The reference does not pretend this equals full 2D or use a GPU intermediate
as its oracle. First-pass arithmetic (`5e-6`) plus one storage ULP gives an RGB
uncertainty interval. This bounds vertical RGB distances and Gaussian weights;
the output bound accounts for weight changes times deviations from the nominal
mean, sample perturbations and the lower normalization bound. A further `5e-6`
plus one output ULP covers the final pass. The largest propagated fp16 budget
in the main corpus is approximately .00331; fp32 uses its own smaller storage
intervals. Single-pass cases use `5e-6` plus one output ULP.

General cutoff cases assert stable membership under intermediate uncertainty
and a conservative fp32 squared-distance/threshold margin; an ambiguous fixture
would fail rather than hide a discrete membership change inside RGB tolerance.
Exact binary boundary fixtures disable that margin only where their decisions
are unambiguous and the vertical footprint is one replicated row. Radius and
sigma extremes, broad near-cutoff separable distributions, nonfinite pixels,
overflow/underflow-scale RGB, performance and autograd remain outside scope.
Only tests/docs/docstrings change; no bilateral shader math changes.

Validation 2026-09-09, with the sRGB/bilateral changes:
**107 passed on each of Metal, CUDA and Vulkan**, selecting
`test_bilateral_numeric.py` and `test_neighborhood_ops.py -k bilateral`
(103 new + 4 existing; 9 unrelated cases deselected). No failures, expected
failures or CUDA autograd selection. SlangPy 0.42.0 / PyTorch 2.13.0+cu130
unchanged; remote runs used an isolated source copy, 120-second timeout per
backend. Ruff, configured mypy and whitespace/link
checks passed. No full-suite, installed-artifact, visual or performance
rerun in this pass.

### Non-local means references

[test_nl_means_numeric.py][nlm] adds 71 cases: three host hand calculations,
30 pattern/patch cases, two public-default checks, eight noise-compensation
cases, six tiny/border cases, four zero-search cases, eight strength-floor
cases, eight center-weight cases and two alpha/repeated-call cases. GPU cases
cover both fp16/fp32. The reference forms sliding-window patch vectors on an
edge-extended image lattice, computes mean squared feature distances in float64,
and reduces normalized candidate/center weights. It uses no production helper
or GPU output as the oracle, and quantizes input/control values first.

The three hand calculations pin uniform-weight corner replication, the center
weight floor, and an out-of-image patch center: samples clamp after patch and
search offsets are added, not after clamping the candidate center first. Border
search offsets are retained even if their candidate RGB samples coincide.
Distance is averaged over patch area **and three RGB channels**. Noise
compensation subtracts `2*sigma**2` before clamping distance to zero. Effective
`h**2` is floored at fp32 `1e-8`; the center takes the maximum neighbor weight
with a fp32 `1e-6` floor. Consequently h=0 is not a general exact-copy promise.

Corpus: finite stored RGB in [-.25,1.25], constants, impulses, edges, repeated
texture, deterministic random/HDR inputs and microcontrast; search radii
0/1/2/5, patch radii 0/1/2, h=0/5e-5/1e-4/2e-4/.1/.2, and sigma=0/.05/.1/.2/2.
The no-argument public call is checked against search=5, patch=1, h=.1, sigma=0.
The center-weight fixtures sample both sides of its floor without requiring an
exact transcendental tie. Alpha is exact, independent of RGB
weights, and can lie outside [0,1]. RGB is not clipped; channel-extrema checks
allow only the arithmetic/store budget. Zero search is identity within that
budget rather than falsely requiring bit-exact multiply/divide cancellation.

Per-value tolerance is `1e-5` fp32 arithmetic plus one output-storage ULP, with
no relative or image-average tolerance and no intermediate GPU texture store.
Large radii/workloads, extreme h/sigma combinations, exact floor-neighbor
enumeration, nonfinite/overflow-scale pixel data, performance and autograd
remain outside scope. Only tests/docs/docstrings change; no NLM shader math
changes.

Validation 2026-09-09, with the sRGB/bilateral/NLM changes:
**73 passed on each of Metal, CUDA and Vulkan**, selecting
`test_nl_means_numeric.py` and `test_neighborhood_ops.py -k 'nlm or nl_means'`
(71 new + 2 existing; 11 unrelated cases deselected). No failures, expected
failures or autograd selection. SlangPy 0.42.0 / PyTorch 2.13.0+cu130 unchanged;
remote runs used an isolated source copy, 120-second timeout per backend.
Ruff, configured mypy and whitespace/link checks passed. No full-suite,
installed-artifact, visual or performance rerun in this pass.

### Kuwahara references

[test_kuwahara_numeric.py][kuwahara] adds 74 cases: two host reference checks,
30 pattern/radius cases, eight variance-power cases, eight spatial-weight cases,
six tiny/border cases, two default-parameter cases, two alpha/repeated-call cases
and 16 large-power regressions (ten constant, six nonconstant). GPU cases cover
fp16/fp32. The reference uses
atan2 interval lookup rather than the shader's integer sector classifier,
centered weighted variance rather than subtraction of raw moments, and
log-normalized sector weights rather than overflowing raw inverse powers.

Boundary hand checks pin all axes/diagonals and center membership in sector 4.
A three-color equal-norm hand calculation shows that tied variances blend
sector means, not choose one winner or turn the result grayscale. The variance
signal is Euclidean RGB magnitude, not weighted luminance. Spatial weights are
`exp(-distance_squared/(sigma_spatial*radius))`, with sector weight sums <=1e-5
skipped; variance is floored at 1e-5. Alpha is exact and ignored by all weights.
RGB is unclipped and evaluated in stored value space.

Corpus: radius 1/2/4 and default 7, sigma_spatial .01/.1/1/3/12, variance powers
.5/1/3/3.5/7; constants, equal-norm chromatic palettes, impulse, a stepped
magnitude ramp and deterministic random dyadic-norm palettes, plus 1x1/1xN/Nx1.
Selected negative/HDR colors reach [-.25,1.25]. Inputs and controls are quantized
before float64 evaluation. A conservative fp32 moment-error budget and spatial
weight approximation allowance are propagated through the variance floor and
normalized sector weights; direct RGB arithmetic adds `2e-5` plus one output
storage ULP. Tests reject fixtures whose propagated variance contribution is
>=.001 instead of accepting weak bounds. The largest main-corpus contribution
is approximately .000907. Exact default equal-norm cases use only the direct
RGB arithmetic/storage budget. These checks do not certify arbitrary continuous
near-floor distributions, HDR overflow scales, maximum radii, performance,
sector-skip threshold neighbors or all variance powers.

**Resolved defect:** constant RGBA=.25 originally failed at var_power=7.5/8/12
in fp16/fp32 due to unstable raw inverse-variance normalization. The fix
scales weights relative to the smallest eligible sector variance, so the
largest weight is 1 and no overflowing absolute inverse power is formed.
Sector classification, spatial weighting, raw-moment variance/floor and alpha
are unchanged. All six original expected failures are now required passes;
additional cases cover constant powers 64/1e6 and nonconstant powers 8/12/64.

Validation 2026-09-09, with the earlier work and this fix:
the initial selection was **61 passed, 6 xfailed** on each backend. After the
fix and expanded regressions, **77 passed on each of Metal, CUDA and Vulkan**,
selecting `test_kuwahara_numeric.py` and `test_regressions.py -k kuwahara`
(74 new + 3 composed-effect checks; 40 unrelated cases deselected). No xfails
or CUDA autograd selection remain in this batch. SlangPy 0.42.0 /
PyTorch 2.13.0+cu130 unchanged; remote runs used an isolated source copy,
120-second timeout per backend. Ruff,
configured mypy and whitespace/link checks passed. No full-suite,
installed-artifact or performance rerun.

Visual follow-up: 24 isolated old/new shader pairs on Metal using three local
photos resized to longest edge 769, fp16/fp32, radius 7, sigma_spatial 12 and
powers 3.5/7/7.5/8. Normal powers 3.5/7 have no visible difference in inspected
images and no channel delta >1/255 across all pairs; maximum fp16 difference
is 0.124512/255. Higher powers remove black patches and nonfinite output
(up to 18.62% of pixels invalid before, zero after). All alpha checks are exact.
Metrics use unclipped readback and separately count nonfinite old pixels;
numeric deltas exclude those pixels. Six representative fp16 sheets were
visually inspected; all 24 pairs were measured. The comparison images were
local artifacts and are not part of this repository. This does not certify
native-resolution photos or other-backend visual parity.

### Color-controls references

[test_color_controls_numeric.py][color-controls] adds 140 cases: two host hand
calculations, 100 individual/combined-control references, eight clamp cases,
six tiny-image/alpha/repeated-call cases, two luma-clipping/unclipped-RGB cases,
20 direct-kernel factor cases and two public contrast-floor cases. Inputs are
quantized to fp16/fp32 before a float64 oracle. Independent matrix multiplication
and complex rotation implement the declared rounded coefficients and radial
projection; test-only sRGB references are reused, not production math helpers.

The operator decodes sRGB before linear-light luma/chroma manipulation, clips
luma to [0, 1], rotates/scales chroma and limits its magnitude to 0.6, then
encodes sRGB and applies affine RGB contrast around 0.5. Final RGB is unclipped;
alpha is exact and does not affect RGB. The rounded conversion matrices are not
exact inverses, so even below the chroma cap neutral controls are approximately,
not exactly, identity. These existing semantics are documented, not redesigned.

Public contrast is signed and maps to `clamp(1+contrast, .001, 5)`. Direct
`ColorControlsOp` takes a factor and retains its legacy `<=1e-4` bypass; adjacent
fp32 controls around the bypass and public/direct agreement for factors
.001/.5/1/2/5 are checked. The bypass discontinuity is characterized, not endorsed
as a continuous numeric control. Public mapping cannot reach it.

The bounded corpus uses normalized palettes/grays/random colors, selected
negative/HDR colors in [-.25, 1.5] and magenta-ray chroma-cap neighbors. An fp32
linear-space allowance `2e-6*(1+max(abs(decoded_input)))` is propagated through
the monotone encoder and contrast factor, plus `2e-6*(1+abs(expected))` final
arithmetic and one output storage ULP. The tests reject a compute allowance
>=.001. This is a per-channel bound, not an image-average tolerance, and rounded
matrix non-identity is part of the reference rather than hidden in that bound.
No arbitrary HDR overflow, pixel NaN/Inf, exhaustive parameter combinations,
performance or low-level API redesign is certified.

Validation 2026-09-09, with the earlier work and this pass:
**142 passed on each of Metal, CUDA and Vulkan**, selecting
`test_color_controls_numeric.py`, `test_colorspace_and_extra_ops.py` and
`test_regressions.py -k color_controls` (140 new + two existing; 48 unrelated
cases deselected). No skips/xfails or CUDA autograd selection. Metal used Apple
M3 Max; CUDA/Vulkan used NVIDIA L40S, unchanged SlangPy 0.42.0 and PyTorch
2.13.0+cu130. Remote runs used an isolated source copy; each backend had a
120-second timeout.
Ruff, configured mypy and whitespace checks passed. No shader math changed;
no full-suite, installed-artifact or visual/performance rerun was needed for
these test/documentation-only changes.

### Highlight/shadow references

[test_highlight_shadow_numeric.py][highlight-shadow] adds 105 cases: one host
hand-calculation case, 72 curve references, six tiny-image/alpha/repeated-call/
low-level parity cases, two clamp/extended-shadow cases, two default RGB-clipping
cases, four curve-activation neighbor cases, two shadow-zero neighbor cases,
four near-black continuity regressions and 12 dark-curve gray/chromatic cases.

The independent scalar oracle uses 50-digit Decimal arithmetic and algebraically
factored power terms. Inputs/controls are quantized first; luma uses declared
(0.299, 0.587, 0.114) weights on encoded RGB, not sRGB-decoded linear RGB. Both
shadow/highlight deltas use the original luma and are added before clipping;
the operation is not a sequential pair of curves. Luma and final RGB are clipped
to [0, 1], whereas alpha is retained exactly and cannot affect RGB. Defaults are
approximately identity on normalized RGB, not on arbitrary HDR/negative pixels.

The initial pass found a <=1e-4 ratio bypass causing a hard jump between gray
.000099/.000101 with shadow_amount=.5/1. On Metal, the largest pre-fix jump was
~.009874 (2.52/255). The fix changes the denominator guard to positive luma,
retaining bypass only at zero/clipped-negative luma. All four initial strict
xfails are now required passes. Tone equations and RGB clipping are unchanged.

For bounded RGB [-.25, 1.5] and shadows [-2, 2], fp32 luma uncertainty propagates
through power/coefficient intervals, exponent rounding, delta addition, ratio
and clipping, then one output storage ULP. Pow uses an eight-epsilon relative
allowance, and subtraction from 1 has an absolute allowance. Every compute bound
must be <.001; cutoff-ambiguous luma intervals are rejected rather than masked
with a loose tolerance. Selected activation roots use adjacent stored fp16/fp32
values. No exact Core Image/GPUImage parity, arbitrary HDR overflow, pixel NaN/Inf,
subnormal behavior, exhaustive controls or visual/performance claim is made.

Dark-curve tests additionally check zero and levels 1e-6 through 2e-4 along
gray and two nonnegative chromatic rays, for shadow amounts -1/-.5/0/.5/1/2.
These use a tighter `2e-7 + 5e-5*abs(expected)` compute allowance plus one
storage ULP, so a broad interval cannot hide an incorrect near-black curve.

Pre-fix validation 2026-09-09, with the earlier work: **92 passed,
4 xfailed on each of Metal, CUDA and Vulkan**, selecting
`test_highlight_shadow_numeric.py` and `test_color_adjust.py -k 'highlight or shadow'`
(89 new passing cases + three existing, four new strict expected failures;
five unrelated cases deselected). All four near-black regressions reproduce on
all three backends. No CUDA autograd or AUTO-device tests were selected.
Metal used Apple M3 Max; CUDA/Vulkan used NVIDIA L40S. SlangPy 0.42.0 and remote
PyTorch 2.13.0+cu130 remain unchanged. Remote runs used an isolated source
copy, with a 120-second timeout per backend. Ruff, configured mypy and
whitespace checks pass. No full-suite, installed-artifact, visual or
performance rerun was performed in that initial pass.

After the fix and expanded dark-curve regressions: **108 passed on
each of Metal, CUDA and Vulkan**, same selection (105 new + three existing,
five unrelated deselected), no xfails/skips or autograd selection. Remote runs
used an isolated source copy; same devices, dependencies and 120-second backend
timeout. The original Metal fp32 example
now differs by .00009848 (~.0251/255), versus ~2.52/255 before. Ruff, configured
mypy and whitespace checks pass. No full-suite, installed-artifact, visual or
performance rerun.

### Hue-rotation references

[test_hue_numeric.py][hue] adds 104 cases: two host reference checks, 80
pattern/angle references, ten fp32 angle-neighbor cases, six tiny-image/alpha/
repeated-call/direct-kernel parity cases, four clipping-boundary cases and two
full-turn/clipped-luma cases. Palette/grayscale/random and selected negative/HDR
RGB span [-.25, 1.5]. Tested angles cover zero, .37/-.73, signed quarter/half/full
turns and 8*pi+.37; neighbors around zero use +/-1e-6, not subnormal controls.

The independent reference uses float64 matrix multiplication and complex chroma
rotation on quantized inputs/angles. It includes the library's rounded matrices,
not an exact inverse or external color-science implementation. A CPU red/zero
hand calculation gives (.999807, .000371, 0) after clipping, explicitly checking
the non-identity conversion. Gray has zero chroma in the ideal rounded-coefficient
reference. RGB is clipped to [0, 1], alpha stays exact and cannot affect RGB.

The operator does not decode sRGB, reduce angles on the host or preserve final
luma exactly: internal Y is constant but reconstruction coefficients are rounded,
and clipping can shift output luma substantially. These existing semantics are
now documented. No hue algorithm or parameter behavior changed.

Each channel uses compute allowance `2e-6*(1+max(abs(stored_rgb)))` plus one
output storage ULP for two small matrices and fp32 trig/rotation. Full-turn
comparison allows both output stores plus 1e-6, since fp32 2*pi is not an exact
turn. Selected clipping planes are straddled with offsets +/- .002, large enough
to survive fp16 input quantization; these are not exhaustive boundary sweeps.
No huge-angle accuracy, arbitrary HDR overflow, pixel NaN/Inf, subnormal controls,
exact Core Image parity, inverse-roundtrip guarantee or performance claim is made.

Validation 2026-09-09, with the earlier work: **105 passed on
each of Metal, CUDA and Vulkan**, selecting `test_hue_numeric.py` and
`test_color_adjust.py -k hue` (104 new + one existing; seven unrelated deselected).
No xfails/skips or CUDA autograd/AUTO-device selection. Metal used Apple M3 Max;
CUDA/Vulkan used NVIDIA L40S, unchanged SlangPy 0.42.0 and PyTorch 2.13.0+cu130.
Remote runs used an isolated source copy, with a 120-second timeout per
backend. Ruff, configured mypy and whitespace checks
passed. No full-suite, installed-artifact or visual/performance rerun; this
pass changes only tests and documentation.

### False-color references

[test_false_color_numeric.py][false-color] adds 64 cases: two host hand checks,
48 pattern/endpoint references, six tiny/default/alpha/repeated-call/direct-kernel
parity cases, two constant-map/unclipped-output cases, four luma-boundary cases
and two endpoint-uniform precision cases. The reference uses float64 luma and
barycentric endpoint weighting, independently of production math helpers; image
storage and fp32 endpoint quantization are applied before reference computation.

Luma weights (0.299, 0.587, 0.114) operate on encoded RGB, without sRGB decoding
or per-channel input clipping. Only the interpolation coordinate is clipped to
[0, 1]. Custom endpoint and output RGB remain unclipped; alpha is retained exactly
and cannot affect the mapped RGB. Equal dyadic endpoints produce an exact
constant map. Endpoint controls remain fp32 even for fp16 textures; selected
controls straddle an fp16 midpoint to check this distinction.

The corpus covers RGB approximately [-.26, 1.5], endpoint components [-.5, 2],
default/gray/reversed/chromatic/equal/HDR maps, tiny/non-square images, and
non-gray cancellation fixtures straddling luma 0/1. A four-fp32-epsilon dot
allowance propagates through coordinate clipping and endpoint separation, plus
four epsilon times the summed absolute endpoints/output for interpolation
arithmetic and one storage ULP. Every compute allowance must be <1e-5. No
image-average tolerance hides boundary errors. No algorithm change or defect was
found in this bounded corpus. Extreme endpoint/input overflow, subnormal behavior,
pixel NaN/Inf, exact Core Image parity and visual/performance claims are outside
scope.

Validation 2026-09-09, with the earlier work: **65 passed on each
of Metal, CUDA and Vulkan**, selecting `test_false_color_numeric.py` and
`test_color_adjust.py -k false_color` (64 new + one existing; seven unrelated
deselected). No xfails/skips or CUDA autograd/AUTO-device selection. Metal used
Apple M3 Max; CUDA/Vulkan used NVIDIA L40S, unchanged SlangPy 0.42.0 and PyTorch
2.13.0+cu130. Remote runs used an isolated source copy, with a 120-second
timeout per backend. Ruff, configured mypy and whitespace
checks passed. No full-suite, installed-artifact or visual/performance rerun;
only tests/documentation changed in this pass.

### Simple color, augmentation, threshold and pointwise references

Three new files add **322 cases** for twelve operators:
[test_simple_color_numeric.py][simple-color] has 98,
[test_color_aug_numeric.py][color-aug-numeric] has 138, and
[test_point_ops_numeric.py][point-numeric] has 86. Inputs are quantized to
fp16/fp32 storage before float64 reference evaluation. Controls are independently
rounded to fp32. Public outputs retain the source device/precision; alpha is
exact, and direct-kernel/public parity plus repeated calls are checked on tiny
1x1/1xN/Nx1 images. The corpus spans palettes, ramps, deterministic random data
and selected RGB in [-.25, 1.5], not arbitrary overflow/nonfinite pixel data.

Exposure references fp32 `2**ev`, RGB multiplication and [0,1] clipping for six
bounded stops. Sepia independently evaluates the rounded classic matrix, clips
its matrix endpoint and clamps intensity before interpolation; partial amounts
can retain negative/HDR source RGB. Solarize checks per-channel inclusive fp32
threshold branches without implicit RGB clipping. Posterize models the fp32
0..255 rounding decision and power-of-two low-bit mask for bits 1/2/4/7/8,
including adjacent fp32 values at a rounding boundary. Grayscale uses an
independent rounded (.299,.587,.114) dot. Normalize checks three fp32 mean/std
triples, including signed std, and unclipped outputs.

Smooth threshold independently evaluates (.2125,.7154,.0721) luma and the
clipped-edge smoothstep polynomial for four transition placements. The two-point
variant uses (.299,.587,.114), signed/HDR inputs and three ordinary ranges plus
the narrow-range regression below. Luma uncertainty is propagated through the
monotone curves before adding a small fp32 arithmetic allowance and one output
storage ULP; branch errors are not hidden by a broad RGB tolerance.

Invert and channel extraction cover unclipped negative/HDR values and all
string/integer channel selectors. Clamp-multiply independently clips each RGB
input and multiplies it, including last-row/column replication when the second
image is smaller and independence from its alpha. High-pass contrast evaluates
`0.5 + strength*(image-blur)/2` for five signed strengths with unclipped output
and same-size inputs. Simple arithmetic uses operation-count-scaled fp32 epsilon
allowances plus one output-storage ULP; no image-average or relative tolerance.

**Resolved finding:** the two-point shader's `max(gap, 1e-6)` denominator made
valid narrower ramps non-linear and returned .0596046448 rather than 1 at the
exact white endpoint for black=.5 and next-fp32 white=.5000000596046448. One
strict expected failure reproduced on all three backends. The shader now
divides by the actual gap and host validation rejects only gaps that become
subnormal/nonfinite in fp32. The expected-failure marker is removed.

Validation 2026-09-12: before correction,
**321 passed, one xfailed** on each of Metal, CUDA and Vulkan. After correction,
**322 passed** on each, selecting the three new files. A separate pre-existing
operator selection passed 16 cases per backend (34 unrelated cases deselected),
and `test_parameter_validation.py -k smooth_threshold_bw` passed 13 host cases
per remote invocation (533 unrelated cases deselected). No xfails/skips or
CUDA-autograd/AUTO-device cases remain in these final selections. Metal used an
Apple M3 Max, Python 3.12.12 and SlangPy 0.42.0. CUDA/Vulkan used an NVIDIA L40S,
driver 580.126.16, Python 3.12.3, SlangPy 0.42.0 and PyTorch 2.13.0+cu130, with
120-second bounds and verified imports from an isolated source copy.
No final full suite, rebuilt artifact, direct cross-backend golden comparison,
autograd, visual or performance rerun is claimed for this pass.

### White balance, stylization and noise references

Three new files add **225 cases** for four operators:
[test_white_balance_numeric.py][white-balance-numeric] has 82,
[test_stylize_numeric.py][stylize-numeric] has 95, and
[test_random_noise_numeric.py][noise-numeric] has 48.

White balance uses independent blackbody/daylight locus equations, fixed
Bradford/sRGB matrices and a clipped decode/adapt/encode pipeline; it does not
import the production matrix builder. The host matrix is rounded to fp32 before
the oracle applies it. Its absolute allowance is `2e-5*(1+abs(expected))` plus
one output-storage ULP. Temperatures exercise the 2000/4000/7000/25000 K loci
and clamps; tint exercises both signs and +/-150 clamps.

Pointillize independently evaluates the fp32 cell hash, searches the same 3x3
neighbor set, truncates/clamps the chosen center and compares the resulting
sample map exactly. Fixtures require the closest two squared distances to differ
by more than `2e-4`, so compiler rounding at a genuine/tiny tie is not certified.
The tests verify that RGB comes from the sampled center while alpha remains at
the output coordinate. CMYK halftone independently performs input clipping,
normalized C/M/Y plus K separation, four rotated smoothstep dot screens and
multiplicative recombination. Its bounded absolute compute allowance is `4e-4`
plus one output-storage ULP.

Random noise intentionally has no CPU or cross-backend golden image: fp32
`sin`/`fmod` hash results can diverge across GPU targets. The suite instead checks
exact repeatability within a backend, identical grayscale channels, opaque alpha,
fp32 seed quantization, seed-independent first-column behavior, fp16 storage
rounding relative to the same backend's fp32 result, and broad mean/deviation/
histogram/neighbor-correlation bounds on 256x256 samples. These are bounded
quality regressions, not cryptographic randomness or cross-platform identity.

Validation 2026-09-12: **225 passed** on each
of Metal, CUDA and Vulkan. A separate pre-existing selection from
`test_white_balance.py`, `test_neighborhood_ops.py` and
`test_blend_and_point_ops.py` passed 10 cases per backend (33 unrelated cases
deselected). No xfails/skips or CUDA-autograd/AUTO-device cases were selected.
Metal used an Apple M3 Max, Python 3.12.12 and SlangPy 0.42.0. CUDA/Vulkan used
an NVIDIA L40S, driver 580.126.16, Python 3.12.3, SlangPy 0.42.0 and PyTorch
2.13.0+cu130, with 120-second bounds and verified imports from an isolated
source copy. No full suite, artifact rebuild/install, direct cross-backend
golden comparison, visual or performance run is claimed for this pass; only
tests and documentation/docstrings changed.

### All blend modes and same-input cross-backend comparison

[test_blend_numeric.py][blend-numeric] adds **110 cases**. An independent
float64 oracle implements all twelve public formulas and the shared source-over
coverage equation. Forty-eight cases cross every mode with all four base/
foreground fp16/fp32 combinations, partial alpha/opacity and a 7x9 base against
a 4x5 foreground to verify final-row/column replication. Twenty-four cases use
opaque RGB values from -.5 through 1.5; twelve probe representable neighbors of
formula branch points; two isolate output alpha immediately below/at/above
`1e-6`; and 24 verify opacity clamps, repeated calls and direct/public parity.
The base controls output precision. The standard absolute compute allowance is
`2e-5*(1+abs(reference))` plus one output-storage ULP (`4e-5` for the HDR
lattice); the isolated cutoff additionally asserts its zero/nonzero sides.
Arbitrary out-of-range alpha, nonfinite/overflow pixels, tensor autograd and
performance remain outside this bounded texture pass.

Validation 2026-09-12: **190 passed** on each
of Metal, CUDA and Vulkan, selecting the new file, all three earlier focused
blend files, `test_blend_and_point_ops.py` and four host harness-contract tests.
The total is 110 new blend cases plus 80 earlier/host cases. No xfails/skips or
CUDA-autograd/AUTO-device cases were selected. Environments match the immediately
preceding pass, with 180-second remote bounds and verified source imports. No
production shader/kernel/validation code changed.

[test_cross_backend_numeric.py][cross-backend] adds four host manifest/policy
contracts. [cross_backend_numeric.py][cross-backend-script] captures a reusable, lossless
NPZ corpus rather than committing one machine as a permanent golden baseline.
On identical structured 24x24 inputs it covered every texture operator exported
at the time, both storage precisions, all blend modes and important bilateral/unsharp/resize/
tile/affine variants: **126 arrays per backend, 63 per precision**. The Metal,
CUDA and Vulkan archives produced **378 pairwise case comparisons**, all passing
their declared policy. Exact remaps require bit equality. Default arithmetic
bounds are .002 fp16 and `8e-5 + 2e-5*abs(value)` fp32; known multi-stage/
transcendental cases use .005 fp16/.0008 fp32. Kuwahara uses .005/.0021, the sum
of its independent propagated-error envelopes. These are disagreement alarms,
not substitutes for operator-specific mathematical oracles.

The largest ordinary bounded difference was Kuwahara: Metal versus CUDA/Vulkan
was .001220703125 fp16 and .00154030323 fp32; CUDA and Vulkan were identical.
Pointillize is evaluated as a discrete sample map: Metal/CUDA matched, while
Vulkan selected the CPU-oracle candidate at one of 576 pixels and the other two
selected its close competing cell, a 0.173611% mismatch at both precisions with
exact alpha. Its policy permits at most 1% changed pixels and records maximum RGB
change separately. Random noise correctly does not compare pixels: pairwise
maximum differences reached .87539059, while mean/std deltas stayed below .04.
The comparator verifies grayscale/opaque/range invariants before those aggregate
checks. For every other case the largest difference was within the explicit
numeric bound; across blend captures specifically, maximum difference was one
fp32 ULP (`1.1920929e-7`). Generated archives/report remained temporary and are
not release artifacts. No full suite, installed-artifact, visual or
performance rerun is claimed.

### Texture-pool ownership, teardown and threading checks

[test_texture_pool_lifetime.py][pool-lifetime] adds six host-side ownership
invariants using texture-shaped fakes, while [test_pool.py][pool] adds two
real-texture cases to its six existing integration checks. The combined scope
covers exact-key reuse, retained outputs remaining unchanged, raw texture aliases
blocking reuse, reuse after alias release, functional-output steady state,
explicit recycle idempotence, explicit/frame overlap, explicit/finalizer overlap,
duplicate auto-recycle registrations, queue/accounting consistency and clear.

**Resolved finding:** the free-list identity guard did not also deduplicate the
pending-finalizer queue. If a texture had multiple auto-recycle registrations,
or was explicitly recycled before its wrapper finalizer ran, the same texture
could remain represented in both ownership queues. The free-list guard prevented
an immediate duplicate entry, but the stale pending reference could pin the
texture or prevent later reuse. The pool now tracks pending identities, keeps
free and pending ownership disjoint, removes stale pending ownership on explicit
recycle, and clears both representations together. Image data and dispatch math
are unchanged.

Validation 2026-09-12: **14 passed** on each of
Metal, CUDA and Vulkan (eight real-device integration cases plus six host
bookkeeping cases). Metal used Apple M3 Max/Python 3.12.12/SlangPy 0.42.0;
CUDA and Vulkan used NVIDIA L40S/driver 580.126.16/Python 3.12.3/SlangPy 0.42.0/
PyTorch 2.13.0+cu130 with 120-second bounds and verified source imports. Focused
ruff, mypy and whitespace checks passed. The full Metal suite at the same tree
passed **5272 tests** (5262 selected-backend/host plus 10 AUTO-device lifecycle),
with the two CUDA-autograd files skipped at collection as intended. This is the
first bounded runtime/lifetime pass; it does not cover frame exceptions/
escape, sustained memory, budget eviction, cache/device teardown or threading,
and is not a full CUDA/Vulkan-suite or installed-artifact claim.

The frame escape/exception extension adds seven more host invariants and two
real-device effect cases. It covers ordinary and exceptional top-level escape,
two-level exception unwinding, nested escaped intermediates, outer failure after
an inner escape, passthrough textures that the inner frame never owned, a nested
`Effect.apply()` chain and an effect that allocates twice before raising.

**Resolved finding:** an output escaped from an inner frame was removed from
that frame without being transferred to its active parent. A nested effect's
intermediate therefore bypassed the outer frame and could not be pooled after a
normal outer return or exception. Frames now retain their parent and promote
only textures actually allocated by the inner frame. Escaping an external input
remains a no-op, and a top-level escaped result remains caller-owned.

Validation 2026-09-12: the expanded **23-case**
selection passed on each of Metal, CUDA and Vulkan (10 real-device integration
cases plus 13 host ownership cases). The full Metal suite passed **5281 tests**
(5271 selected-backend/host plus 10 AUTO-device lifecycle), with two CUDA-
autograd files skipped at collection as intended. Focused/full runs used the
same environments above; remote runs retained 120-second bounds and verified
source imports. Ruff, mypy, whitespace and documentation-link checks passed.
Sustained memory, budget eviction, cache/device teardown and threading remain
outside this pass; no full CUDA/Vulkan suite or installed-artifact validation is
claimed.

The sustained-memory/budget extension adds 12 host cases (six format-accounting
parameters plus six eviction/cap/workload cases) and one real-device sustained
effect case. It verifies exact free-byte accounting for RGBA/RG/R fp16/fp32,
old-key-first eviction, same-key trimming, the documented one-hot-texture floor,
the 16-entry per-key policy, the eight-entry pending-alias policy, a 100-step
changing-size workload and a 64-iteration post-warmup effect plateau.

[pool_memory_stress.py][pool-stress] separately runs eight warmups followed by
two equal 64-iteration steady windows at 256x256, reporting pool counters,
process RSS high-water and NVML device-wide memory when available. This is a
bounded lifetime diagnostic, not a benchmark-protocol result. Re-run
2026-11-02 with `SoftGlowEffect` (Metal on the M3 Max; CUDA/Vulkan on an L40S,
SlangPy 0.42.0): across Metal, CUDA and Vulkan at fp16/fp32, every run created
two textures during warmup and none afterward; each phase held two free
textures (1.0 MB fp16/2.1 MB fp32). Second-window RSS growth was at most
1.0 MB fp16 and .131 MB fp32 on Metal (two runs each) and 0 MB on CUDA and
Vulkan. NVML growth was 0 MB for all four CUDA/Vulkan runs.

Validation 2026-09-12: the expanded **36-case**
selection passed on each backend (11 real-device integration cases plus 25 host
cases). The full Metal suite passed **5294 tests** (5284 selected-backend/host
plus 10 AUTO-device lifecycle), with two CUDA-autograd files skipped at
collection as intended. Ruff, mypy, whitespace and documentation-link checks
passed. No production allocation policy changed and no defect was found in this
extension. Cache/device teardown and threading remain outside it; no full CUDA/
Vulkan suite or installed-artifact validation is claimed.

The cache/device teardown extension adds four host-only cache-root cases and one
real-device surviving-Image case in [test_device_teardown.py][device-teardown].
It verifies that `release_device()` is idempotent, isolated to the exact device,
clears the pool plus module/operator/default-manager roots, permits collection
once package roots are gone, recreates a released default device, and leaves a
caller-owned device/Image usable with freshly rebuilt caches. The function does
not close the device: final teardown remains drop derived objects, call
`release_device(device)`, then call `device.close()`.

[device_teardown_stress.py][device-teardown-stress] separately performs four
create/dispatch/release/close/GC cycles in one process. SlangImage's three cache
maps were empty after every cycle on Metal, CUDA and Vulkan. Used device wrappers
nevertheless remained in SlangPy's created-device registry. Local Metal RSS
high-water rose from 445.497 MB after cycle one to 966.115 MB after cycle four.
On the L40S, CUDA NVML residency after teardown rose from 1112.146 to 1164.575
MB (640.090 MB baseline), while RSS high-water rose from 1389.654 to 3073.274
MB; Vulkan rose from 727.187 to 874.709 MB NVML and 1110.036 to 1987.879 MB RSS.
This diagnostic identifies native/driver retention after the package roots are
gone; it does not attribute that residency to SlangImage's cleared caches or
claim that process RSS high-water should fall.

Validation 2026-09-12: the combined pool and
teardown selection passed **41 cases** on each backend (12 real-device cases and
29 host cases). The teardown diagnostic completed with no SlangImage cache-root
failures on all three backends. The full Metal suite passed **5299 tests** (5289
selected-backend/host plus 10 AUTO-device lifecycle), with two CUDA-autograd
files skipped at collection as intended. Repeated device recreation is therefore
not a supported memory-bounding pattern on the validated SlangPy 0.42 stack;
reuse a long-lived device or use process boundaries for hard reclamation.
Threading remains outside this pass; no full CUDA/Vulkan suite or installed-
artifact validation is claimed.

The threading extension adds nine cases in
[test_threading_contracts.py][threading-contracts]. Before correction, the
process-wide default returned the same device to another thread, and sequential
cross-thread functional operations completed because the pool's frame owner was
reset after each outer frame. Only overlapping frames failed. The new affinity
registry binds each `DeviceManager` result to its construction thread and a
direct SlangPy device to its first SlangImage caller. Default lookup, upload,
readback, Image access, functional dispatch, texture pools, module caches and
release now reject another thread before GPU work or cache mutation. Real
SlangPy devices are weak-referenceable; non-weak-referenceable host test doubles
remain outside affinity tracking.

The host cases cover a foreign default-device lookup, two simultaneous first
default lookups (one owner, one rejection and one construction), serialized
explicit construction, sequential and overlapping shared-pool rejection,
unmodified owner-frame state after rejection, concurrent distinct pools,
module-cache rejection, and a foreign finalizer. Weakref callbacks cannot
propagate a useful exception, so a finalizer on the wrong thread deliberately
drops that texture's recycle opportunity without touching free/pending state.
The real-device boundary case checks Image metadata/readback, functional
dispatch, pool access and release from a foreign thread, then verifies the Image
is still exact and usable on its owner.

[threading_stress.py][threading-stress] creates an explicit device inside each
of two workers, synchronizes them before eight invert dispatches, then releases
and closes each device on its owner thread. Both workers returned exact fp32
results on Metal, CUDA and Vulkan; each pool created two outputs and recorded
six reuses. This is a bounded supported-pattern diagnostic, not evidence that
one device, Image, effect or cache may be shared or that arbitrary worker/device
counts are certified.

Validation 2026-09-12: all **nine new cases**
and the combined **50-case** lifetime/threading selection passed on each backend
(13 real-device cases and 37 host cases). The full Metal suite passed **5308
tests** (5298 selected-backend/host plus 10 AUTO-device lifecycle), with two
CUDA-autograd files skipped at collection as intended. Ruff, mypy, whitespace
and documentation-link checks passed. No full CUDA/Vulkan suite or installed-
artifact validation is claimed; tensor stream/device threading is outside these
checks (see the CUDA tensor section below).

### Device creation and backend selection

[`test_device_creation.py`](../tests/test_device_creation.py) has seventeen host
cases that replace native device creation and the compile probe with recorders;
one checks that `slangimage.diff` explains a failed CUDA device creation.
`DeviceManager` accepts backend names as strings and rejects unknown names.
AUTO keeps a CUDA device that compiles; AUTO closes a CUDA device whose one-line
test compile fails, warns, and continues on Vulkan; Metal/Vulkan devices are
never probed; an AUTO failure lists every attempt; an explicit CUDA device that
fails the probe raises with setup guidance; explicit CUDA/Vulkan/Metal creation
failures on Linux and macOS raise `RuntimeError` naming the backend, SlangPy's
reason without its native source location, and the platform-specific hint; the
CPU rejection is unchanged. Real-device cases run the probe kernel on the
selected backend and check that `from_numpy`, `from_pil` and `load` accept the
selected backend's name or `Backend` member for `device=` (returning images on
that backend's default device) and reject unknown names;
`typing_tests/public_api.py` pins the widened `device=` type.

Validation 2026-09-27 on the NVIDIA L40S (driver 580.126.16, SlangPy 0.43.1,
run from an isolated source copy; system NVRTC hidden with a private mount
namespace where noted):

| Setup | Previous behavior | Now |
|---|---|---|
| apt NVRTC 13.0 | AUTO → CUDA | AUTO → CUDA; explicit CUDA works |
| Driver only (NVRTC hidden) | AUTO → Vulkan; explicit CUDA: `Failed to create device!` | AUTO → Vulkan; explicit CUDA raises with setup guidance |
| CUDA 12.9 pip NVRTC via `CUDA_PATH` only (NVRTC hidden) | AUTO → CUDA, first operation fails with `E52002 pass-through compiler not found` | AUTO warns, closes CUDA, quickstart runs on Vulkan; explicit CUDA raises with the `E52002` reason and setup guidance |
| CUDA 13.0 pip NVRTC via `CUDA_PATH` only (NVRTC hidden) | AUTO → Vulkan (CUDA device creation fails) | Unchanged |
| Full pip recipe, CUDA 13.0 or 12.9 (NVRTC hidden) | AUTO → CUDA | AUTO → CUDA |

The full Metal suite at this change passed **5337 tests** (5327 selected-
backend/host plus 10 AUTO lifecycle) on both SlangPy 0.42.0 and 0.43.1.

The first probe called a scalar-returning function; the benchmark re-run found
that this allocated SlangPy's first CUDA buffer, which reserves a 256 MiB heap
for the process (+268 MB in every CUDA VRAM cell). The probe now writes one
pixel of a 1×1 texture and reads it back. On the L40S with SlangPy 0.43.1 (five
fresh processes each, interleaved), NVML memory after device creation was 1114
MB with the probe versus 1112 MB without it, and 1129 MB after a first 1024²
operation either way — the published baseline. Device creation took
2.3–2.9 s versus 0.56–0.66 s, and the first operation 0.29–0.44 s versus
2.04–2.08 s, so the net cost is about 0.35 s per process. The installed
release-candidate wheel repeated this (five processes each): 1114 versus 1112 MB
after creation and 1129 MB after the first operation; creation 2.4–2.9 s versus
0.53–0.66 s and first operation 0.30–0.37 s versus 2.0–2.2 s, a net cost of
about 0.5 s. The half-configured
CUDA 12.9 case still warns and falls back to Vulkan (explicit CUDA raises), and
apt NVRTC, driver-only and fully configured pip setups behave as before.

### Helpers and effects

| Export | Evidence | Precision / prewarm |
|---|---|---|
| `prewarm` | [prewarm][prewarm]: `test_prewarm_covers_all_ops`, `test_prewarm_rejects_invalid_arguments_before_device_init` | Selected device, fp16/fp32, every exported op completes; no all-modes assertion |
| `BLEND_MODES` | [point][point]: `test_blend_mode_names_cover_shader_switch` | Host constant; no storage precision |
| `identity_lut_np` | [regressions][regressions]: `test_identity_lut_size_validation`; [lut][lut]: `test_identity_lut_builder_matches_grid_coordinates` at five sizes | Host float32 array; uploaded fp16/fp32 in tests and used by prewarm |
| `GaussianBlurEffect` | [smoke][smoke]: `test_effect_smoke`; [regressions][regressions]: `test_effect_parameter_contracts` (functional parity) | fp32; constituent ops prewarmed, no effect registry entry |
| `SoftGlowEffect` | [smoke][smoke]: smoke; [public][public]: `test_effect_accepts_image`; [regressions][regressions]: parameter contracts; [pool][pool]: repeatability/retained-output checks | fp32 smoke, fp16 API; constituent ops only |
| `KuwaharaPaintEffect` | [smoke][smoke]: smoke; [regressions][regressions]: constant-color invariant | fp32; constituent ops only |

## CUDA tensor operators (`si.diff`) and modules (`si.nn`)

These are separate implementations/storage paths from the texture API. Their
references **do not establish texture-kernel correctness**, even for matching
names. Independent mathematical references use fp32 CUDA tensors; explicit
bridge coverage also exercises fp16 and bfloat16 leaves. There is no tensor-
autograd prewarm registry and no Metal/Vulkan torch-autograd coverage.

The 2026-09-12 core extension in [test_diff_core_contracts.py][diff-core] adds
31 CUDA cases. Pure-torch fp32 references compare both complete forward tensors
and random-projection VJPs for exposure, sepia, hue rotation, full-border
Gaussian, unsharp, a generated-op composition and the equivalent four-module
`nn.Sequential`. Blend covers multiply/screen/overlay with both inputs requiring
grad and each mixed-grad direction, away from overlay's branch boundary. Six
generated paths run no-grad→grad→no-grad→grad to exercise the separate SlangPy
call-data caches. Every one of the nine public tensor ops runs backward twice on
the same retained graph and must accumulate exactly twice its first finite
gradient. These are deterministic bounded fp32 cases, not double-precision
`gradcheck` or arbitrary-value guarantees.

The next 40 cases cover all nine ops with fp16 and bfloat16 inputs. Outputs and
random-projection VJPs are compared with the same quantized values passed as
fp32, establishing the bridge's fp32 compute/output and low-precision leaf-
gradient contract rather than comparing different input values. Each op also
uses both a spatially strided slice and an HWC view over channel-first storage;
blend and LUT exercise both inputs. Bilateral's manual bridge is observed saving
its fp32 source plus the four-channel forward result, retaining them after user
references are dropped, unpacking them during backward and rejecting in-place
source mutation. Resize saves no activation tensors, and its geometry-only
adjoint is exactly unchanged by input mutation. Bfloat16 was exercised on the
L40S after confirming native support.

| Differentiable operator | Evidence in [test_diff.py][diff] | Earlier fp16 smoke in [test_nn.py][nn] |
|---|---|---|
| `gaussian_blur` | Existing NumPy/finite differences plus [core contracts][diff-core]: full-border torch forward/VJP, transitions, composition and repeated backward | `test_fp16_inputs_upcast_and_grad` |
| `bilateral` | `test_bilateral_forward_matches_numpy`, `test_bilateral_fp32_finite_differences`; core contracts: repeated backward | Upcast/grad; autocast pipeline |
| `unsharp_mask` | Existing NumPy/finite differences plus core contracts: full torch forward/VJP and repeated backward | No direct fp16-input test |
| `resize` | `test_resize_matches_torch` (bilinear/bicubic forward/backward), finite differences/aspect; core contracts: repeated backward | Upcast/grad; autocast matmul input (finite output) |
| `exposure` | Existing analytic forward/gradient plus core contracts: pure-torch VJP, transitions, composition and repeated backward | No direct fp16-input test |
| `sepia` | Existing NumPy/finite differences plus core contracts: pure-torch VJP, transitions, composition and repeated backward | Upcast/grad; autocast pipeline |
| `hue_rotate` | Existing luma/finite differences plus core contracts: full pure-torch forward/VJP, transitions, composition and repeated backward | No direct fp16-input test |
| `blend` | Existing three-mode forward/overlay finite differences plus core contracts: all modes/both mixed-grad directions, transitions and repeated backward | No direct fp16-input test |
| `lut` | Existing identity/both-input finite differences/learning plus core contracts: transitions and repeated backward | Half image + fp32 LUT upcast/grad; no half-LUT gradient test |

The older mixed-precision smoke only checked fp32 outputs and finite fp16 input
gradients. The core contract file now provides quantized-fp32 numerical forward/
VJP comparisons for fp16 and bfloat16 across every op. This does not compare a
low-precision input with a differently quantized original fp32 input.
`diff.available()` is a capability helper, not a tested image operator.

| `nn` wrapper | Wrapper-level evidence in [test_nn.py][nn] |
|---|---|
| `GaussianBlur` | `test_modules_match_functionals`; `test_sequential_pipeline_backward` |
| `Bilateral` | Functional parity, sequential backward, `test_module_fp32_finite_differences`, autocast pipeline |
| `UnsharpMask` | `test_modules_match_functionals` |
| `Resize` | `test_modules_match_functionals` (bicubic); mixed-precision resize tests call `diff` directly |
| `Exposure` | Functional parity; sequential backward |
| `Sepia` | Functional parity; sequential backward; autocast pipeline |
| `HueRotate` | `test_modules_match_functionals` |
| `Blend` | `test_modules_match_functionals` (overlay) |
| `Lut` | `test_lut_module_learns`, `test_lut_state_dict_roundtrip`, `test_lut_module_rejects_invalid_layout` |

Wrapper parity is not another independent oracle; most gradient evidence is in
the corresponding `diff` function tests. The core extension adds an independent
pure-torch forward/VJP comparison for a four-wrapper `nn.Sequential`.

Validation 2026-09-12: all **31 new core
cases** passed on the L40S, and the complete three-file CUDA tensor selection
passed **82 tests** under Python 3.12.3, SlangPy 0.42.0 and PyTorch
2.13.0+cu130. No production tensor/shader behavior changed. The full Metal suite
passed **5313 tests** (5303 selected-backend/host plus 10 AUTO lifecycle), with
all three CUDA-only files skipped at collection. Mixed precision/bfloat16,
layouts, non-default devices/streams, manual-backward saved-tensor lifetime and
higher-order behavior remain outside this core extension.

The subsequent dtype/layout/lifetime extension adds **40
cases** to the same file. Three deliberately narrow native-bridge cases then
confirm clean rejection of CPU tensors, invalid float3 channels and mismatched
blend shapes. All **74 contract cases** and the complete three-file CUDA tensor
selection passed **125 tests** on the same L40S/Python/SlangPy/
PyTorch stack. No production tensor or shader behavior changed. Non-default
devices/streams, a broad unsupported-input matrix, repeated training and higher-
order behavior remain outside this extension. Because this CUDA-only file is
still collected as one skip node on Metal, the last full Metal count remains
5314 passed and three expected collection skips. A full Metal rerun on the
package content of the 2026-09-12 support/artifact pass below also passed 5314
tests, with the three CUDA-autograd files excluded at collection as intended.

### SlangPy 0.43 generated-backward regression

The 2026-09-17 fresh-L40S investigation explains why the CUDA tensor selection
does not complete with SlangPy 0.43.0/0.43.1. It is not a broad PyTorch bridge
failure: base CUDA texture smoke passed, exposure/sepia/hue/blend/LUT generated
backward passed, and bilateral/resize manual backward passed. The affected
generated paths are Gaussian blur and unsharp mask, plus their `nn` wrappers and
any composed graph containing them. The full selection reaches
`test_gaussian_backward_closed_form` and then remains in native backward.

The standalone
[`runtime_loop_autodiff_hang` reproducer](../experiments/runtime_loop_autodiff_hang/)
uses SlangPy tensor APIs and direct `.bwds()` without importing PyTorch. Its
runtime-count differentiable load produced this bounded matrix:

| SlangPy | Bundled Slang | Result |
|---|---|---|
| 0.42.0 | 2026.5.2 | pass, correct gradient sum |
| 0.43.0 | 2026.12 | hang; external 12-second timeout |
| 0.43.1 | 2026.12 | hang; external 12-second timeout |
| 0.43.0 rebuilt against 2026.5.2 | 2026.5.2 | pass, correct gradient sum |

The 0.43.1 fixed-count control also passed. Further controls isolate the trigger
to runtime-dependent control flow around a differentiable tensor load, rather
than dynamic indexing or the older negative-loop-start defect. Rebuilding the
same 0.43 native/wrapper code against the older compiler makes the reproducer
pass, assigning the regression to the Slang 2026.12 compiler bundled by the
0.43 line. The unchanged repository's focused Gaussian/unsharp selection then
passed **22 tests** under the SlangPy 0.43.0/Slang 2026.5.2 hybrid. This
validates the existing dependency policy: use SlangPy 0.42.0 for the Torch
extra, while the wider base-install range remains supported for non-autograd
texture operations. No production algorithm was changed.

### Installed-artifact checks (outside pytest)

[validate_installed_artifact.py](../scripts/validate_installed_artifact.py)
with `--gpu-smoke` explicitly selects its requested backend and uses fp32 input.
It checks `invert` against a constant-color/alpha reference and shape/finite
output for `tile`, `cmyk_halftone`, and `clahe`. It adds neither fp16 numerical
coverage nor autograd checks. Its results are separate from pytest group counts.
There is no hosted CI workflow; artifact validation is run explicitly on a local
Metal machine and on a Linux NVIDIA host for CUDA/Vulkan.

The 2026-09-12 support/artifact pass built the wheel and sdist twice with the
same source epoch and obtained byte-identical pairs. Both passed Twine plus the
complete archive/member/RECORD/source-byte checker; the wheel contains 95
runtime files. Clean base installs of that exact wheel passed dependency checks, payload/
version/entry-point validation and the four-op Metal artifact smoke on Python
3.10.20, 3.11.6, 3.12.12 and 3.13.15. Each resolved SlangPy 0.43.1 without
Torch. A clean Python 3.12 sdist install passed the same Metal smoke. The
`dev,bench,playground` wheel extras passed dependency checks; the benchmark CLI
help and packaged Gradio 6.27 playground import both succeeded.

On Linux Python 3.12.3, the exact wheel passed CUDA and Vulkan artifact smokes
in a clean base environment with SlangPy 0.43.1. A clean `torch` extra install
(PyTorch 2.13 first, then `slangpy-torch` 0.7.0 built with
`--no-build-isolation`, then the wheel's extra) resolved PyTorch 2.13.0+cu130,
SlangPy 0.42.0 and slangpy-torch 0.7.0, then completed generated-plus-manual
backward. A single `pip install "slangimage[torch]"` cannot succeed:
`slangpy-torch` is source-only and imports torch at build time (re-verified
2026-09-27; see [INSTALL_TORCH.md](INSTALL_TORCH.md)). That environment ran all
125 CUDA tensor tests from outside the source tree, importing the installed
wheel rather than the source.

Final full suites also imported the installed reproducible wheel from outside
the source tree.
CUDA passed **5439 tests**: 5304 selected-backend texture/host, 10 AUTO lifecycle
and 125 CUDA autograd. Vulkan passed **5314 tests** plus the three expected CUDA-
only collection skips: 5304 selected-backend texture/host and 10 AUTO lifecycle.
This is complete-suite installed-wheel evidence for Linux Python 3.12.3, not a
Python 3.10–3.13 Linux/backend cross-product.

The performance pass then changed README/docstrings and the packaged
playground comment without changing runtime behavior. Its fixed-epoch rebuild
was again byte-identical across two builds, passed the 95-file archive checker,
current Twine 7, and a clean Python 3.12 Metal installed-wheel smoke. These
were pre-release builds, not published release artifacts. Windows and a Python-
version-by-Linux-backend cross-product were not tested.

### Performance, service and shipped examples (outside pytest)

Protocol v2 was rerun without changing its suite or methodology. The published
tables report that run under protocol v3, which removed one case. The tracked `benchmarks/results_{metal,cuda,vulkan}.json` files contain
the remaining 36 cells per backend: 15 synchronized steady-state iterations at 1024², 2048² and 3456² for
fp16/fp32, plus first-call timing. Metal used the M3 Max; CUDA/Vulkan used the
L40S; all three used SlangPy 0.42 and the strict reporter accepted the merged
matrix. Upload, readback, encoding, network work and pipelining remain explicitly
outside this GPU-resident low-level benchmark. The standard low-level op cases
include raw output allocation.

Across the refreshed standard matrix, fp16/fp32 steady-state ratios range from
1.02× to 2.16×, with per-backend medians of 1.52×–1.63×. The exact storage claim
is narrower: rgba16 textures use half the bytes of rgba32 textures; total process
or device memory also includes live work, the pool, Slang/driver caches and the
GPU context. CUDA is the fastest backend in 35 of 36 standard-suite cells in
this run, including every resize cell; Vulkan is narrowly faster for 2048² fp32
Kuwahara (2.826 vs 2.863 ms). Older roadmap wording that excluded resize
describes the dated July result.

The standalone augmentation and differentiable comparisons were also rerun on
PyTorch 2.13, Kornia 0.8.3 and TorchVision 0.28, and their tracked JSON now records
adapter, OS, Python and all library versions plus completion time. The script
headers state non-equivalent border/color/filter semantics and the RGBA-vs-RGB
workload distinction. Current representative outcomes include bilateral forward
45–230× versus Kornia, bilateral training 93–97× at 1–4 MP (79–104× on a
second L40S host, 2026-09-27) with a 194× lower Torch-allocator peak, and CUDA autocontrast 2.6–5.3× slower than TorchVision.
Tiny CUDA elementwise cells and the 2048² Kornia Gaussian training step are also
reported as losses rather than hidden by aggregate wording.

From an installed wheel, examples 01–05 completed on Metal and emitted their
expected images/data/pool plateau; the no-Torch environment produced example
06's actionable install message. On the validated CUDA Torch stack, example 06
completed the chained backward and all 80 LUT-training steps with final loss
`3.02e-06`. The installed benchmark command completed full run/report flows and
the packaged playground prewarmed, served HTTP 200, then was terminated by the
test harness. Existing real-device sustained-workload evidence above remains the
service allocation/residency check; broader repeated-training matrices were
not run for this release.

HTTP 200 did not prove the playground worked: on 2026-09-27 its first preview
failed with the thread-ownership `RuntimeError`, because the page was served
from Gradio worker threads while the default device belonged to the main
thread. The playground now runs all GPU work on one dedicated thread.
[`test_playground.py`](../tests/test_playground.py) (AUTO group, skipped
without Gradio) imports the playground in a subprocess and calls its preview,
add-to-pipeline and clear callbacks from separate threads; it fails on the
previous code. The fixed playground was also exercised in a browser on Metal:
first preview, operator switch, add to pipeline, fp32 toggle and clear, with
no server errors.

## Reporting backend coverage accurately

Pytest's terminal summary separates **Selected-backend texture + host tests**,
**AUTO-device API/lifecycle tests** (the `auto_device` marker), and **CUDA
autograd tests** (the three CUDA-only filenames). It reports outcomes,
not an operator count or device telemetry for each test. Fixture setup/teardown
errors remain errors; skipped collection nodes are not counted as test cases.

| Selection | Texture `ctx` fixture | CUDA-only tensor files |
|---|---|---|
| `metal` | Metal | Excluded before torch/CUDA probing |
| `vulkan` | Vulkan | Excluded before torch/CUDA probing |
| `cuda` | CUDA | CUDA, if torch and CUDA are available; otherwise skipped |
| unset / `auto` | Automatic platform selection | CUDA if available, otherwise skipped |

Important limits:

- The environment variable steers the **test fixture**, not the library's
  `get_default_device()` globally. [public API tests][public] use AUTO
  deliberately; the release-device lifecycle test also creates AUTO. These are
  marked `auto_device` and reported separately. Explicit multi-device checks
  create extra devices on the selected backend. A full Vulkan invocation can
  still include AUTO texture dispatches; use `-m 'not auto_device'` to exclude
  that group. Prewarm is now an explicit-device check, not part of the AUTO group.
- The selected-backend texture + host group contains pure-host artifact,
  metadata, validation, and reporting checks. Never describe its total as that many Vulkan/Metal/CUDA
  numerical tests. The summary prints the requested backend and actual `ctx`
  device (or “not initialized” for a host-only selection).
- A typical explicit Vulkan/Metal run reports **three skipped collection nodes**:
  the three entire CUDA files, not three skipped gradient test cases. A CUDA run
  with those modules skipped has **no validated autograd coverage**, even if
  pytest otherwise exits successfully. Check the group and skip reasons (`-ra`).
- Same-backend fp16/fp32 PSNR is not cross-backend parity. The capture harness
  now compares common Metal/CUDA/Vulkan inputs explicitly, but pytest alone does
  not orchestrate three machines or enforce a permanent golden-output archive.
- No GitHub Actions workflow is enabled: the generic hosted runners do not meet
  this project's GPU validation requirements. Metal runs locally; CUDA/Vulkan
  run on a Linux NVIDIA host (helper scripts in `scripts/cuda_box/`). A
  command/configuration is not evidence of a run.

For separate test selections:

```bash
# Selected texture fixture + host checks; exclude AUTO and autograd
SLANGIMAGE_BACKEND=vulkan pytest tests/ -m 'not auto_device' --ignore=tests/test_diff.py --ignore=tests/test_diff_core_contracts.py --ignore=tests/test_nn.py -q -ra
# Intentional automatic-device API/lifecycle checks only
pytest tests/ -m auto_device --ignore=tests/test_diff.py --ignore=tests/test_diff_core_contracts.py --ignore=tests/test_nn.py -q -ra
# Every public texture operator dispatched on the selected device, both precisions
SLANGIMAGE_BACKEND=vulkan pytest tests/test_prewarm.py -q -ra
# CUDA tensor/autograd evidence only (requires the project's CUDA stack). Without
# torch/CUDA it reports three skipped modules and exits 5 ("no tests collected"):
# that is no evidence, not a pass.
SLANGIMAGE_BACKEND=cuda pytest tests/test_diff.py tests/test_diff_core_contracts.py tests/test_nn.py -q -ra
```

Attach source revision/dirty state, installed artifact identity (if applicable),
Python/SlangPy/PyTorch versions, GPU/driver, requested and actual devices, exact
selection, and per-group passed/failed/skipped results to release evidence.
The validated CUDA autograd stack is PyTorch 2.13 with SlangPy 0.42.0 (the
`torch` extra pins it); this inventory does not certify other dependency
versions. See [UPSTREAM_ISSUES.md](UPSTREAM_ISSUES.md).

## Known gaps and next priorities

The named blend and same-input cross-backend priorities are complete. The four
involved operators and twelve simpler operators are also covered by the dated
records above. Full-suite and artifact results for the published release are
not yet recorded on this page.

1. Extend boundary policy to remaining low-level/effect entry points and input
   pixel data, and define service-facing resource budgets where needed. Public
   `si.ops` numeric-control validation is now covered above. The default EP
   defect is fixed, with no expected failures in its regressions.
2. Further color/local-histogram/geometry/neighborhood cases are optional
   extensions, not known defects. The new harness provides true same-input
   cross-backend comparisons while preserving per-operation precision policy;
   it is intentionally not a permanent artifact golden corpus.
3. Blend's texture path now covers all public modes with mixed precision,
   normalized alpha/opacity, HDR RGB and the tiny-alpha cutoff. Out-of-range
   alpha, arbitrary overflow/nonfinite pixels, tensor autograd and performance
   remain separate contracts.
4. Extend registry mode/variant coverage beyond the now-enforced exported-operator
   completeness and explicit fp16/fp32 device checks. Expand CUDA mixed-precision
   numerical and bfloat16 coverage separately, not under a Vulkan test count.

Maintain this map when adding/removing exports, tests, or registry cases. Update
evidence strength, not just the number of test functions.

[precision]: ../tests/test_precision.py
[public]: ../tests/test_public_api.py
[prewarm]: ../tests/test_prewarm.py
[regressions]: ../tests/test_regressions.py
[extra]: ../tests/test_colorspace_and_extra_ops.py
[adjust]: ../tests/test_color_adjust.py
[aug]: ../tests/test_color_aug.py
[wb]: ../tests/test_white_balance.py
[point]: ../tests/test_blend_and_point_ops.py
[neighborhood]: ../tests/test_neighborhood_ops.py
[geometry]: ../tests/test_geometry.py
[core]: ../tests/test_core_numeric.py
[complex]: ../tests/test_complex_numeric.py
[parameters]: ../tests/test_parameter_validation.py
[resampling]: ../tests/test_resampling_numeric.py
[lut]: ../tests/test_lut_numeric.py
[textures]: ../tests/test_textures.py
[blend-alpha]: ../tests/test_blend_alpha.py
[soft-light]: ../tests/test_blend_soft_light.py
[color-mode]: ../tests/test_blend_color.py
[equalize]: ../tests/test_equalize_numeric.py
[autocontrast]: ../tests/test_autocontrast_numeric.py
[clahe]: ../tests/test_clahe_numeric.py
[srgb]: ../tests/test_srgb_numeric.py
[bilateral]: ../tests/test_bilateral_numeric.py
[nlm]: ../tests/test_nl_means_numeric.py
[kuwahara]: ../tests/test_kuwahara_numeric.py
[color-controls]: ../tests/test_color_controls_numeric.py
[highlight-shadow]: ../tests/test_highlight_shadow_numeric.py
[hue]: ../tests/test_hue_numeric.py
[false-color]: ../tests/test_false_color_numeric.py
[simple-color]: ../tests/test_simple_color_numeric.py
[color-aug-numeric]: ../tests/test_color_aug_numeric.py
[point-numeric]: ../tests/test_point_ops_numeric.py
[white-balance-numeric]: ../tests/test_white_balance_numeric.py
[stylize-numeric]: ../tests/test_stylize_numeric.py
[noise-numeric]: ../tests/test_random_noise_numeric.py
[blend-numeric]: ../tests/test_blend_numeric.py
[cross-backend]: ../tests/test_cross_backend_numeric.py
[cross-backend-script]: ../scripts/cross_backend_numeric.py
[pool]: ../tests/test_pool.py
[pool-lifetime]: ../tests/test_texture_pool_lifetime.py
[pool-stress]: ../scripts/pool_memory_stress.py
[device-teardown]: ../tests/test_device_teardown.py
[device-teardown-stress]: ../scripts/device_teardown_stress.py
[threading-contracts]: ../tests/test_threading_contracts.py
[threading-stress]: ../scripts/threading_stress.py
[smoke]: ../tests/test_smoke.py
[diff]: ../tests/test_diff.py
[diff-core]: ../tests/test_diff_core_contracts.py
[nn]: ../tests/test_nn.py
