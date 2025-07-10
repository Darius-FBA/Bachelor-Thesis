#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

def decode_time_words(f0, f1, f2):
    a = np.frombuffer(np.float32(f0).tobytes(), dtype=np.uint32)[0]
    b = np.frombuffer(np.float32(f1).tobytes(), dtype=np.uint32)[0]
    c = np.frombuffer(np.float32(f2).tobytes(), dtype=np.uint32)[0]
    lo0 = a & 0xFFFF
    hi0 = (a >> 16) & 0xFFFF
    lo1 = b & 0xFFFF
    hi1 = (b >> 16) & 0xFFFF
    lo2 = c & 0xFFFF
    hi2 = (c >> 16) & 0xFFFF
    start = lo0 + (hi0 << 16) + (lo1 << 32)
    end = hi1 + (lo2 << 16) + (hi2 << 32)
    return start, end

parser = argparse.ArgumentParser()
parser.add_argument('--name', help="compiled output folder")
parser.add_argument('--cmaddr', help="IP:port of the WSE")
args = parser.parse_args()

with open(f"{args.name}/out.json", encoding='utf-8') as f:
    compile_data = json.load(f)

N_PER_PE = int(compile_data['params']['N'])
width = int(compile_data['params']['width'])
N = N_PER_PE * width
alpha = 2.0

x = np.arange(N, dtype=np.float32)
scal_expected = alpha * x

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
x_symbol = runner.get_id("x")
time_symbol = runner.get_id("time_memcpy")
ref_symbol = runner.get_id("time_ref")

runner.load()
runner.run()

runner.memcpy_h2d(x_symbol, x, 0, 0, width, 1, N_PER_PE,
    streaming=False, order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.call("f_sync", [], nonblock=False)
runner.call("f_tic", [], nonblock=False)
runner.call("f_scal", [], nonblock=False)
runner.call("f_toc", [], nonblock=False)
runner.call("f_memcpy_timestamps", [], nonblock=False)
runner.call("f_reference_timestamps", [], nonblock=False)

x_result = np.zeros(N, dtype=np.float32)
runner.memcpy_d2h(x_result, x_symbol, 0, 0, width, 1, N_PER_PE,
    streaming=False, order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

time_data = np.zeros(width * 3, dtype=np.float32)
ref_data = np.zeros(width * 2, dtype=np.float32)
runner.memcpy_d2h(time_data, time_symbol, 0, 0, width, 1, 3,
    streaming=False, order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.memcpy_d2h(ref_data, ref_symbol, 0, 0, width, 1, 2,
    streaming=False, order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.stop()

np.testing.assert_allclose(dot_result, dot_expected, rtol=1e-5, atol=1e-3)
print("SCAL PARALLEL SUCCESS!")


starts, ends = [], []
for i in range(width):
    f0, f1, f2 = time_data[3 * i], time_data[3 * i + 1], time_data[3 * i + 2]
    r0, r1 = ref_data[2 * i], ref_data[2 * i + 1]
    start, end = decode_time_words(f0, f1, f2)
    ref = int(np.frombuffer(np.float32(r0).tobytes(), dtype=np.uint32)[0])
    starts.append(start - ref - i)
    ends.append(end - ref - i)

cycles = max(ends) - min(starts)
print(cycles)
