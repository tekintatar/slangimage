"""Host-side numeric contracts for texture operators (not workload budgets)."""

import math
import struct


FLOAT32_MAX = float.fromhex("0x1.fffffep+127")
INT32_MAX = 2**31 - 1
# Keeps squared offsets and (2*r+1)**2 inside signed shader integer arithmetic.
MAX_RADIUS = 16383
# Keep squared Gaussian sigmas and their reciprocals normal/finite in fp32,
# including fast-math backends which may flush subnormals to zero.
MIN_SIGMA = 2.0**-60
MAX_SIGMA = 2.0**60


def finite_float(name, value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite fp32 scalar") from error
    if (not math.isfinite(result) or abs(result) > FLOAT32_MAX
            or (result != 0 and fp32(result) == 0)):
        raise ValueError(f"{name} must be a finite fp32 scalar")
    return result


def fp32(value):
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def integer(name, value, minimum=0, maximum=INT32_MAX):
    result = finite_float(name, value)
    if not result.is_integer() or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return int(result)


def radius(name, value, minimum=1):
    return integer(name, value, minimum, MAX_RADIUS)


def boolean(name, value):
    return bool(integer(name, value, 0, 1))


def sigma(name, value):
    result = finite_float(name, value)
    if not MIN_SIGMA <= result <= MAX_SIGMA:
        raise ValueError(f"{name} must be > 0 and in [{MIN_SIGMA:g}, {MAX_SIGMA:g}] for fp32 weights")
    return result


def vector(name, value, length=3):
    try:
        values = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must have {length} components") from error
    if len(values) != length:
        raise ValueError(f"{name} must have {length} components")
    return tuple(finite_float(f"{name}[{i}]", v) for i, v in enumerate(values))


def dimensions(width, height, device):
    # This is the first device access for generators, prewarm and some direct
    # geometry-kernel calls, so enforce affinity before reading device limits.
    from .runtime.device import _claim_device_thread

    _claim_device_thread(device)
    width, height = integer("width", width, 1), integer("height", height, 1)
    limit = device.info.limits.max_texture_dimension_2d
    if limit > 0 and max(width, height) > limit:
        raise ValueError(f"width and height must not exceed device texture limit {limit}")
    if width * height > INT32_MAX // 4:
        raise ValueError("width * height exceeds signed RGBA indexing capacity")
    return width, height
