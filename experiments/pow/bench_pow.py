# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


DEVICE_ID = 0
L1_SMALL_SIZE = 16384

LENGTHS = [1_000_000]
EXPONENT = 2.5


def run_case(device, length):
    torch_input = torch.linspace(0.5, 2.0, steps=length, dtype=torch.float32).reshape(1, length)
    torch_output = torch.pow(torch_input, EXPONENT)

    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = ttnn.pow(tt_input, EXPONENT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # below does not give error but above does
    # tt_output = ttnn.mul(tt_input, EXPONENT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)

    max_abs_diff = torch.max(torch.abs(torch_output - tt_output_torch)).item()
    mean_abs_diff = torch.mean(torch.abs(torch_output - tt_output_torch)).item()

    print(
        f"L={length:4d}  shape={tuple(tt_output_torch.shape)}  "
        f"max_abs_diff={max_abs_diff:.6f}  mean_abs_diff={mean_abs_diff:.6f}"
    )

    ttnn.deallocate(tt_output)
    ttnn.deallocate(tt_input)


def main():
    torch.manual_seed(0)
    device = ttnn.CreateDevice(device_id=DEVICE_ID, l1_small_size=L1_SMALL_SIZE)

    print(f"Testing ttnn.pow with exponent={EXPONENT}")
    for length in LENGTHS:
        run_case(device, length)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
