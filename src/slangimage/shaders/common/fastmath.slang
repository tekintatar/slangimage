// shaders/common/fastmath.slang
// Approximation-tolerant transcendentals for hot-loop WEIGHTING math
// (never for accumulation math). Semantics: "this call site tolerates
// ~1e-5 relative error — compile it as fast as the target allows."
//
// Target reality (measured, experiments/nvrtc_fastmath + cuda_hotloop):
// Vulkan and Metal drivers compile exp/pow to fast hardware approximations
// BY DEFAULT. NVRTC defaults to precise software routines (pow 4.4x
// slower) — so runtime/device.py sets floating_point_mode=fast, which
// maps them to the hardware ops and makes CUDA behave like the other
// backends. The restructuring wins (hoisted invariants, algebraic
// consolidation, integer classification instead of atan2) still matter:
// they reduce operation COUNT on every backend.

static const float FASTMATH_LOG2E = 1.44269504;

// e^x via the exp2 hardware instruction
float fast_exp(float x)
{
    return exp2(x * FASTMATH_LOG2E);
}

// x^y for x > 0 via log2/exp2 hardware instructions
float fast_pow(float x, float y)
{
    return exp2(y * log2(x));
}
