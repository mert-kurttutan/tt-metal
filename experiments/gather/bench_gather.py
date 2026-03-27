import time

import torch
import ttnn


torch.manual_seed(0)


DEVICE_ID = 0
WARMUP_RUNS = 2
MEASURE_RUNS = 10

INPUT_DTYPE = ttnn.bfloat16
INDEX_DTYPE = ttnn.uint32
LAYOUT = ttnn.TILE_LAYOUT


CASES = [
    {
        "name": "2d_dim1_small",
        "input_shape": (32, 128),
        "index_shape": (32, 64),
        "dim": 1,
    },
    {
        "name": "2d_dim0_small",
        "input_shape": (128, 32),
        "index_shape": (64, 32),
        "dim": 0,
    },
    {
        "name": "4d_last_dim_medium",
        "input_shape": (1, 4, 64, 512),
        "index_shape": (1, 4, 64, 256),
        "dim": 3,
    },
    {
        "name": "4d_dim2_medium",
        "input_shape": (1, 4, 256, 128),
        "index_shape": (1, 4, 64, 128),
        "dim": 2,
    },
    {
        "name": "4d_last_dim_wide",
        "input_shape": (1, 2, 32, 4096),
        "index_shape": (1, 2, 32, 2048),
        "dim": 3,
    },
    {
        "name": "4d_dim1_batched",
        "input_shape": (2, 8, 64, 128),
        "index_shape": (2, 4, 64, 128),
        "dim": 1,
    },
]


def make_inputs(case):
    input_shape = case["input_shape"]
    index_shape = case["index_shape"]
    dim = case["dim"]

    torch_input = torch.randn(input_shape, dtype=torch.float32)
    torch_index = torch.randint(0, input_shape[dim], index_shape, dtype=torch.int64)
    torch_index_ttnn = torch_index.to(torch.int32)

    return torch_input, torch_index, torch_index_ttnn


def benchmark_torch(torch_input, torch_index, dim):
    for _ in range(WARMUP_RUNS):
        torch.gather(torch_input, dim, torch_index)

    start = time.time()
    for _ in range(MEASURE_RUNS):
        torch_output = torch.gather(torch_input, dim, torch_index)
    end = time.time()

    avg_ms = (end - start) * 1000.0 / MEASURE_RUNS
    return torch_output, avg_ms


def benchmark_ttnn(device, torch_input, torch_index_ttnn, dim):
    tt_input = ttnn.from_torch(torch_input, dtype=INPUT_DTYPE, layout=LAYOUT, device=device)
    tt_index = ttnn.from_torch(torch_index_ttnn, dtype=INDEX_DTYPE, layout=LAYOUT, device=device)

    ttnn.synchronize_device(device)
    compile_start = time.time()
    tt_output = ttnn.gather(tt_input, dim, tt_index)
    ttnn.synchronize_device(device)
    compile_end = time.time()

    for _ in range(WARMUP_RUNS):
        tt_output = ttnn.gather(tt_input, dim, tt_index)
        ttnn.synchronize_device(device)

    ttnn.synchronize_device(device)
    start = time.time()
    for _ in range(MEASURE_RUNS):
        tt_output = ttnn.gather(tt_input, dim, tt_index)
        ttnn.synchronize_device(device)
    end = time.time()

    avg_ms = (end - start) * 1000.0 / MEASURE_RUNS
    compile_ms = (compile_end - compile_start) * 1000.0
    torch_output = ttnn.to_torch(tt_output)
    return torch_output, compile_ms, avg_ms


def main():
    device = ttnn.open_device(device_id=DEVICE_ID)

    print(f"device_id={DEVICE_ID}")
    print(f"warmup_runs={WARMUP_RUNS}")
    print(f"measure_runs={MEASURE_RUNS}")
    print()

    header = (
        f"{'case':<24}"
        f"{'dim':>4}"
        f"{'input_shape':>24}"
        f"{'index_shape':>24}"
        f"{'torch_ms':>12}"
        f"{'ttnn_first_ms':>16}"
        f"{'ttnn_avg_ms':>14}"
        f"{'speedup':>10}"
        f"{'max_abs_diff':>16}"
    )
    print(header)
    print("-" * len(header))

    for case in CASES:
        torch_input, torch_index, torch_index_ttnn = make_inputs(case)
        torch_output, torch_ms = benchmark_torch(torch_input, torch_index, case["dim"])
        ttnn_output, ttnn_first_ms, ttnn_avg_ms = benchmark_ttnn(device, torch_input, torch_index_ttnn, case["dim"])

        max_abs_diff = (torch_output - ttnn_output.to(torch.float32)).abs().max().item()
        speedup = torch_ms / ttnn_avg_ms

        print(
            f"{case['name']:<24}"
            f"{case['dim']:>4}"
            f"{str(case['input_shape']):>24}"
            f"{str(case['index_shape']):>24}"
            f"{torch_ms:>12.3f}"
            f"{ttnn_first_ms:>16.3f}"
            f"{ttnn_avg_ms:>14.3f}"
            f"{speedup:>10.2f}"
            f"{max_abs_diff:>16.6f}"
        )

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
