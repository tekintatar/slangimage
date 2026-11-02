# Minimal repro: a [Differentiable] function that BOTH contains a loop
# AND returns a vector (float3) produces silently WRONG input gradients
# on the torch autograd path. The IDENTICAL loop body with a scalar
# return gives exact gradients.
#
#   python repro_vector_return_grads.py
#
# The two kernels below share the same body; the losses built on them
# are the same mathematical function of the input (the script prints
# both forward losses to show they agree). Ground truth is central
# finite differences THROUGH THE SAME slangpy forward — no external
# reference involved. Forward is correct in both variants; only the
# vector-return backward is wrong, and no error or warning is raised.
#
# Requires torch + CUDA. slangpy 0.42.0.

import torch
import slangpy as spy

SHADER = """
import slangpy;

// Toy edge-aware filter: 1D neighborhood, data-dependent weights.
// Two functions with IDENTICAL bodies; only the return differs.

[Differentiable]
float3 filter_vec(no_diff int2 pix, IDiffTensor<float, 3> src,
                  no_diff int r, no_diff float sr, no_diff int width)
{
    float3 c = float3(src[uint3(0, pix.x, pix.y)],
                      src[uint3(1, pix.x, pix.y)],
                      src[uint3(2, pix.x, pix.y)]);
    float a0 = 0.0; float a1 = 0.0; float a2 = 0.0; float wsum = 0.0;
    [MaxIters(5)]
    for (int i = 0; i < 2 * r + 1; ++i)
    {
        int q = clamp(pix.x + i - r, 0, width - 1);
        float3 s = float3(src[uint3(0, q, pix.y)],
                          src[uint3(1, q, pix.y)],
                          src[uint3(2, q, pix.y)]);
        float3 d = s - c;
        float w = exp(-dot(d, d) / (2.0 * sr * sr));
        a0 += w * s.x; a1 += w * s.y; a2 += w * s.z; wsum += w;
    }
    return float3(a0, a1, a2) / wsum;          // vector return -> WRONG grads
}

[Differentiable]
float filter_scalar(no_diff int2 pix, IDiffTensor<float, 3> src,
                    no_diff int r, no_diff float sr, no_diff int width)
{
    float3 c = float3(src[uint3(0, pix.x, pix.y)],
                      src[uint3(1, pix.x, pix.y)],
                      src[uint3(2, pix.x, pix.y)]);
    float a0 = 0.0; float a1 = 0.0; float a2 = 0.0; float wsum = 0.0;
    [MaxIters(5)]
    for (int i = 0; i < 2 * r + 1; ++i)
    {
        int q = clamp(pix.x + i - r, 0, width - 1);
        float3 s = float3(src[uint3(0, q, pix.y)],
                          src[uint3(1, q, pix.y)],
                          src[uint3(2, q, pix.y)]);
        float3 d = s - c;
        float w = exp(-dot(d, d) / (2.0 * sr * sr));
        a0 += w * s.x; a1 += w * s.y; a2 += w * s.z; wsum += w;
    }
    return (a0 + 2.0 * a1 + 3.0 * a2) / wsum;  // scalar return -> exact grads
}
"""

H, W, R, SR = 6, 7, 2, 0.5
PROBES = [(0, 0, 0), (2, 3, 1), (5, 6, 2), (3, 0, 0), (1, 1, 2)]
EPS = 1e-2


def main() -> None:
    device = spy.create_torch_device(type=spy.DeviceType.cuda)
    module = spy.Module.load_from_source(device, "vec_return_repro", SHADER)

    torch.manual_seed(1)
    base = torch.rand(H, W, 3, device="cuda")
    # channel weights (1, 2, 3): loss_vec == loss_scalar as functions
    cw = torch.tensor([1.0, 2.0, 3.0], device="cuda")

    def loss_vec(src):
        out = module.filter_vec(spy.grid((H, W)), src, R, SR, W)
        return (out * cw).sum()

    def loss_scalar(src):
        return module.filter_scalar(spy.grid((H, W)), src, R, SR, W).sum()

    # analytic gradients via torch autograd (grad calls FIRST — avoids
    # interaction with issue #1052)
    grads = {}
    for name, fn in [("vec", loss_vec), ("scalar", loss_scalar)]:
        src = base.clone().requires_grad_(True)
        loss = fn(src)
        loss.backward()
        grads[name] = src.grad.clone()
        print(f"forward loss ({name:6s}): {float(loss):.6f}")

    # ground truth: central finite differences through the scalar forward
    print(f"\n{'element':>10s} {'FD':>10s} {'scalar':>10s} {'vec':>10s}")
    errs = {"vec": [], "scalar": []}
    for (i, j, k) in PROBES:
        p = base.clone()
        p[i, j, k] += EPS
        m = base.clone()
        m[i, j, k] -= EPS
        fd = float(loss_scalar(p) - loss_scalar(m)) / (2 * EPS)
        for name in ("vec", "scalar"):
            errs[name].append(abs(fd - float(grads[name][i, j, k])))
        print(f"{(i, j, k)!s:>10s} {fd:10.4f} "
              f"{float(grads['scalar'][i, j, k]):10.4f} "
              f"{float(grads['vec'][i, j, k]):10.4f}")

    print(f"\nscalar-return max grad err: {max(errs['scalar']):.2e}  "
          f"{'OK' if max(errs['scalar']) < 0.05 else 'WRONG'}")
    print(f"vector-return max grad err: {max(errs['vec']):.2e}  "
          f"{'OK' if max(errs['vec']) < 0.05 else 'WRONG'}")


if __name__ == "__main__":
    main()
