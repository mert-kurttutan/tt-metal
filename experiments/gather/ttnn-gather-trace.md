# `ttnn.gather` Trace

This note traces `ttnn.gather` from the Python API surface down to the C++ device implementation.

## High-level path

`ttnn.gather(...)`

-> Python symbol imported from `_ttnn`

-> data-movement nanobind module registration

-> nanobind function binding for `"gather"`

-> high-level C++ wrapper `ttnn::gather(...)`

-> primitive/device op entry `ttnn::prim::gather(...)`

-> generic `ttnn::device_operation::launch<GatherDeviceOperation>(...)`

-> `GatherDeviceOperation` validation / output allocation / factory selection

-> gather program factory builds kernels and runtime args

-> gather reader/writer kernels execute on device

## Python surface

There is no Python `def gather(...)` wrapper in the package. The symbol comes from the compiled extension module.

- Python package import surface:
  [ttnn/ttnn/__init__.py](/root/tt-metal-gather/ttnn/ttnn/__init__.py#L500)

## Nanobind registration

The data-movement extension module registers the gather binding here:

- Module registration:
  [ttnn/cpp/ttnn/operations/data_movement/data_movement_nanobind.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/data_movement_nanobind.cpp#L99)

The actual Python binding is created here. The Python name is `"gather"`, and it binds directly to `&ttnn::gather`:

- Binding implementation:
  [ttnn/cpp/ttnn/operations/data_movement/gather/gather_nanobind.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/gather_nanobind.cpp#L18)
- Function binding call:
  [ttnn/cpp/ttnn/operations/data_movement/gather/gather_nanobind.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/gather_nanobind.cpp#L70)

Bound arguments:

- `input`
- `dim`
- `index`
- keyword-only:
  `sparse_grad=False`, `memory_config=None`, `out=None`, `sub_core_grids=None`

## High-level C++ wrapper

The first real implementation layer is `ttnn::gather(...)`:

- Declaration:
  [ttnn/cpp/ttnn/operations/data_movement/gather/gather.hpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/gather.hpp)
- Implementation:
  [ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp#L154)

This wrapper does the shape/layout normalization before the device op:

- Early-exits for empty tensors.
- Treats gather as a last-dimension operation internally.
- Transposes if `dim` is not already the last dimension.
- Converts tensors to a 4D form when needed.
- Pads index tiles with zeros.
- Slices and pads the input tensor to match the index tensor’s non-gather dimensions.
- Calls `ttnn::prim::gather(...)`.
- Postprocesses the output by squeezing/transposing/reshaping back to the original logical shape.

Key helper sections:

- Preprocessing helper:
  [ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp#L43)
- Postprocessing helper:
  [ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp#L109)
- Primitive call:
  [ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp#L204)

Important design point:

The kernel implementation is effectively built around gather on the last dimension. The wrapper makes arbitrary `dim` work by reordering tensors before the device launch.

## Primitive/device-op entry

The primitive entrypoint is:

- Device-op header:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.hpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.hpp)
- Device-op implementation:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp#L135)

This function forwards into the generic launcher:

- Launch call:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp#L143)

## Generic launch framework

The generic TT-NN device-op launcher lives here:

- Launcher:
  [ttnn/api/ttnn/device_operation.hpp](/root/tt-metal-gather/ttnn/api/ttnn/device_operation.hpp#L431)

Relevant launch behavior:

- Tracks the op in graph tracing.
- Allocates output tensors through `create_output_tensors(...)`.
- Resolves the mesh/device context.
- Goes through validation, program-cache lookup, program factory selection, program creation, and enqueue.

Important lines:

- Launch entry:
  [ttnn/api/ttnn/device_operation.hpp](/root/tt-metal-gather/ttnn/api/ttnn/device_operation.hpp#L431)
- Output tensor creation:
  [ttnn/api/ttnn/device_operation.hpp](/root/tt-metal-gather/ttnn/api/ttnn/device_operation.hpp#L448)
- Validation on cache miss:
  [ttnn/api/ttnn/device_operation.hpp](/root/tt-metal-gather/ttnn/api/ttnn/device_operation.hpp#L288)
- Program factory selection:
  [ttnn/api/ttnn/device_operation.hpp](/root/tt-metal-gather/ttnn/api/ttnn/device_operation.hpp#L290)

## `GatherDeviceOperation`

`GatherDeviceOperation` defines validation, output shape/spec computation, output allocation, performance model, and program-factory selection.

- Types:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation_types.hpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation_types.hpp)
- Validation and spec logic:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp#L32)

Validation performed:

- input/index rank must match
- optional output shape must match index shape
- index dtype must be `UINT16` or `UINT32`
- for all non-gather dimensions, `index.shape[d] <= input.shape[d]`
- sharded output is not supported
- both tensors must be `TILE` layout
- both tensors must already be device tensors with backing buffers
- `sparse_grad` must be `false`

Output behavior:

- output logical shape = index tensor logical shape
- output dtype = input tensor dtype
- output layout = tile layout
- output memory config = requested config or input tensor config

## Program factory selection

Factory selection is based on width in tiles:

- Selection logic:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp#L16)

Rule:

- If `Wt_input > 60` or `Wt_index > 60`, use the multi-core-width path.
- Otherwise use the single-row/single-core-row path.

## Program factories

Factory definitions:

- Header:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.hpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.hpp)
- Implementation:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp#L13)

What the factories do:

- create circular buffers for input tensor, index tensor, and output tensor
- select a core grid, optionally overridden by `sub_core_grids`
- split work across cores
- create reader and writer kernels
- attach runtime arguments per core

### Single-row / single-core-row path

This path parallelizes over `Ht` rows.

- Factory body:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp#L13)
- Work split:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp#L66)
- Reader kernel path:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp#L107)
- Writer kernel path:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp#L129)

### Multi-core-width path

This path parallelizes over `Wt_index`.

- Factory body:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp#L182)
- Work split:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp#L235)
- Reader kernel path:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp#L277)
- Writer kernel path:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp#L298)

## Device kernels

Kernel files:

- Common helpers:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_common.hpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_common.hpp)
- Single-core reader:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_single_core.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_single_core.cpp)
- Single-core writer:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_single_core.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_single_core.cpp)
- Multi-core reader:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_multi_core.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_multi_core.cpp)
- Multi-core writer:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_multi_core.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_multi_core.cpp)

### Single-core reader

The reader kernel:

- reads one index tile
- waits until the full input row is available in L1
- maps each global gather index to:
  tile index + local offset within the tile
- writes gathered values into the output tile buffer

Key section:

- Kernel main and index mapping:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_single_core.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_single_core.cpp#L108)
- Mapping math:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_single_core.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_single_core.cpp#L175)

### Single-core writer

The writer kernel:

- streams the full input row into L1
- writes completed output tiles back to DRAM/L1 output storage

- Writer logic:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_single_core.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_single_core.cpp#L24)

### Multi-core reader

The multi-core reader:

- parallelizes over output/index tiles along `Wt_index`
- for each assigned output tile, scans input tiles and only fills entries whose gather index belongs to the current input tile

- Kernel logic:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_multi_core.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_multi_core.cpp#L46)
- Tile membership check:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_multi_core.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_multi_core.cpp#L120)

### Multi-core writer

The multi-core writer:

- reads input tiles into L1
- writes the assigned output tile for that core

- Writer logic:
  [ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_multi_core.cpp](/root/tt-metal-gather/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_multi_core.cpp#L16)

## Summary

`ttnn.gather` is a direct Python binding to a C++ implementation. The high-level wrapper normalizes arbitrary-rank and arbitrary-dimension gathers into a last-dimension tiled gather, then launches a device op through the generic TT-NN device-operation framework. The actual execution is handled by reader/writer dataflow kernels chosen through a simple width-based factory selection rule.
