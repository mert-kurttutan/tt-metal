# `ttnn.pow` Trace

This note traces `ttnn.pow` from the Python API surface down to the C++ implementation.

## High-level path

`ttnn.pow(...)`

-> Python symbol imported from `_ttnn`

-> binary nanobind registration

-> nanobind overload selection

-> C++ composite `ttnn::pow(...)`

-> one of two execution paths:

- scalar exponent:
  unary-style power helpers
- tensor exponent:
  binary-ng / legacy binary dispatch

## Python surface

There is no Python `def pow(...)` wrapper implementing the op itself. The callable comes from the compiled extension and is auto-registered onto the `ttnn` module. The C++ auto-registration happens from [ttnn/ttnn/__init__.py](/root/tt-metal-gather/ttnn/ttnn/__init__.py#L358).

Python does attach the golden/reference function for validation:

- [ttnn/ttnn/operations/unary.py](/root/tt-metal-gather/ttnn/ttnn/operations/unary.py#L261)

So the split is:

- C++ owns the real op behavior
- Python owns golden/reference behavior for `torch.pow`

## Benchmark in this folder

The local repro harness is:

- [bench_pow.py](/root/tt-metal-gather/experiments/pow/bench_pow.py#L1)

This benchmark currently exercises the scalar-exponent path:

- exponent is a host float `2.5`: [bench_pow.py](/root/tt-metal-gather/experiments/pow/bench_pow.py#L13)
- input is built in row-major and passed to `ttnn.pow`: [bench_pow.py](/root/tt-metal-gather/experiments/pow/bench_pow.py#L19)
- the real op call is: [bench_pow.py](/root/tt-metal-gather/experiments/pow/bench_pow.py#L26)

## Nanobind registration

`ttnn.pow` is bound in the binary nanobind module:

- [ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp#L1598)

The overload set is important:

- tensor + `int32_t` exponent: [ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp#L1603)
- tensor + `float` exponent: [ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp#L1614)
- tensor + tensor exponent: [ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp#L1622)

This is the first major design detail for `pow`: different argument shapes enter different C++ paths.

## Public C++ API

The public overloads are declared here:

- [ttnn/cpp/ttnn/operations/eltwise/binary/binary_composite.hpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/binary_composite.hpp#L31)

There are four user-visible forms:

- `pow(Tensor, int32_t)`
- `pow(Tensor, float)`
- `pow(Tensor, Tensor)`
- `pow(float, Tensor)`

## Scalar exponent path

### Float exponent

The float-exponent overload starts here:

- [ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp#L823)

Behavior:

- if the float exponent is actually an integer, it converts to the integer overload: [ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp#L828)
- otherwise it calls `ttnn::power(...)`: [ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp#L833)

### Integer exponent

The integer-exponent overload starts here:

- [ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp#L837)

Behavior:

- exponents `0`, `1`, `2`, `3` use `power_iterative(...)`: [ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp#L842)
- larger integer exponents route to `ttnn::power(...)`: [ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp#L847)

The unary power op is declared in the unary API as a scalar-variant op:

- [ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp#L171)

This is the second major design detail for `pow`: scalar exponent `pow` is not treated like a binary tensor-tensor op. It delegates to unary power helpers when possible.

## Tensor exponent path

The tensor-exponent overload starts here:

- [ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp#L851)

This path forwards directly into:

- `ttnn::detail::invoke_binary_ng(..., BinaryOpType::POWER, ...)`
- see [ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp#L861)

The shared binary dispatch logic is in:

- [ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp#L464)

That dispatch may:

- use legacy `prim::binary(...)` under legacy-only conditions: [ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp#L477)
- otherwise route to `prim::binary_ng(...)`: [ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp#L540)

The scalar-input / tensor-exponent overload is a thin helper that materializes the scalar as a tensor with `full_like(...)` and then reuses the tensor-exponent path:

- [ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp#L875)

## Where the kernel actually is

There is not one single `ttnn.pow` kernel file. The low-level kernel entrypoint depends on whether the exponent is scalar or tensor.

### Scalar exponent kernel path

For `ttnn.pow(x, c)`:

- small integer exponents may use `power_iterative(...)`: [ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp#L394)
- unary code generation emits `power_tile(...)` or `power_iterative_tile(...)`: [ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp#L169)
- the public compute-kernel API wrappers are:
  - [tt_metal/hw/inc/api/compute/compute_kernel_api.h](/root/tt-metal-gather/tt_metal/hw/inc/api/compute/compute_kernel_api.h#L337)
  - [tt_metal/hw/inc/api/compute/compute_kernel_api.h](/root/tt-metal-gather/tt_metal/hw/inc/api/compute/compute_kernel_api.h#L364)

The actual numerical implementation for unary/scalar power is in:

- [tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_power.h](/root/tt-metal-gather/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_power.h#L19)

The LLK wrapper that connects the tile API to that implementation is:

- [tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_power.h](/root/tt-metal-gather/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_power.h#L13)

### Tensor exponent kernel path

For `ttnn.pow(x, y)`:

- `BinaryOpType::POWER` is routed as an SFPU binary op: [ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp#L281)
- binary-ng emits the compute op name `power_binary_tile`: [ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp#L409)
- the public compute-kernel API wrapper is:
  - [tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h](/root/tt-metal-gather/tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h#L55)

The actual numerical implementation for tensor-tensor power is in:

- [tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h](/root/tt-metal-gather/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h#L19)

The LLK wrapper that connects the tile API to that implementation is:

- [tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binary_pow.h](/root/tt-metal-gather/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binary_pow.h#L13)

## Numerical algorithm

The numerical algorithm is implemented in the SFPU ckernel headers, not in the high-level TTNN wrapper.

For both unary/scalar and binary/tensor power, the core method is:

1. compute `log2(base)`
2. compute `base^pow` as `2^(pow * log2(base))`

The unary/scalar implementation explicitly documents this in:

- [tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_power.h](/root/tt-metal-gather/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_power.h#L20)

That file also notes:

- polynomial approximation based on Moroz et al. 2022
- handling for `base = 0, pow < 0`
- handling for negative base with integer vs non-integer exponent
- explicit bf16 rounding behavior

The tensor-tensor implementation documents the same structure in:

- [tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h](/root/tt-metal-gather/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_pow.h#L19)

There is also a distinct fast path for small positive integer exponents:

- `power_iterative_tile(...)` is documented as an iterative multiplication loop and faster for exponents like `1`, `2`, `3`: [tt_metal/hw/inc/api/compute/compute_kernel_api.h](/root/tt-metal-gather/tt_metal/hw/inc/api/compute/compute_kernel_api.h#L354)

## Practical interpretation

`ttnn.pow` is really two families of ops behind one Python name:

1. Scalar exponent:
   This is mostly a unary-style power operation, with special handling for small integer exponents.
2. Tensor exponent:
   This is a real binary elementwise op using binary-ng or legacy binary infrastructure.

That means debugging depends on the call shape:

- wrong scalar-exponent results:
  start in [binary_composite_op.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp#L823) and the unary power declarations in [unary.hpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp#L171)
- wrong tensor-exponent results:
  start in [binary_composite_op.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp#L851) and then follow [binary.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp#L464)
- golden/reference mismatch only:
  check [unary.py](/root/tt-metal-gather/ttnn/ttnn/operations/unary.py#L261)

## Bottom line

`ttnn.pow` looks simple from Python, but it is not a single backend path:

- nanobind exposes multiple overloads
- Python adds only the golden function
- scalar exponent uses unary/composite power helpers
- tensor exponent uses binary-ng / legacy binary dispatch

That distinction is the main thing to preserve when changing or benchmarking `ttnn.pow`.
