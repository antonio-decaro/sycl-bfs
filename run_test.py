#!/usr/bin/env python3

import sys, subprocess
from tqdm import tqdm

if len(sys.argv) < 2:
    print("Usage: run_test.py <iters> -- executable [args]")
    exit(1)

iters = int(sys.argv[1])

executable = sys.argv[2]
args = sys.argv[3:] if (len(sys.argv) > 3) else []

kernel_times = []
runtime_times = []

print("[*] Running benchmark...")
for iter in tqdm(range(1, iters + 1)):
    out = subprocess.run([executable, *args],  capture_output=True)
    if not out.returncode:
        bench_out = out.stdout.decode('utf-8')
        for line in bench_out.splitlines():
            if "Kernels" in line:
                kernel_times.append(int(line.split(' ')[-2]))
            elif "Total" in line:
                runtime_times.append(int(line.split(' ')[-2]))

print("[*] Benchmark finished (all results are expressed in us)")

print(f"Kernel times: {kernel_times}")
print(f"Runtime times: {runtime_times}")

# print Mean
print(f"- Mean kernel time: {sum(kernel_times)/len(kernel_times)}")
print(f"- Mean runtime time: {sum(runtime_times)/len(runtime_times)}")

# print harmonic mean
print(f"- Harmonic mean kernel time: {len(kernel_times)/sum(1/x for x in kernel_times)}")
print(f"- Harmonic mean runtime time: {len(runtime_times)/sum(1/x for x in runtime_times)}")

# print median
kernel_times.sort()
runtime_times.sort()
print(f"- Median kernel time: {kernel_times[len(kernel_times)//2]}")
print(f"- Median runtime time: {runtime_times[len(runtime_times)//2]}")

# print min
print(f"- Min kernel time: {min(kernel_times)}")
print(f"- Min runtime time: {min(runtime_times)}")

# print max
print(f"- Max kernel time: {max(kernel_times)}")
print(f"- Max runtime time: {max(runtime_times)}")

# print variance
print(f"- Variance kernel time: {sum((x - sum(kernel_times)/len(kernel_times))**2 for x in kernel_times)/len(kernel_times)}")
print(f"- Variance runtime time: {sum((x - sum(runtime_times)/len(runtime_times))**2 for x in runtime_times)/len(runtime_times)}")

# print standard deviation
print(f"- Standard deviation kernel time: {sum((x - sum(kernel_times)/len(kernel_times))**2 for x in kernel_times)/len(kernel_times)**0.5}")
print(f"- Standard deviation runtime time: {sum((x - sum(runtime_times)/len(runtime_times))**2 for x in runtime_times)/len(runtime_times)**0.5}")

# print standard error
print(f"- Standard error kernel time: {sum((x - sum(kernel_times)/len(kernel_times))**2 for x in kernel_times)/len(kernel_times)**0.5/len(kernel_times)**0.5}")
print(f"- Standard error runtime time: {sum((x - sum(runtime_times)/len(runtime_times))**2 for x in runtime_times)/len(runtime_times)**0.5/len(runtime_times)**0.5}")

# print confidence interval
print(f"- Confidence interval kernel time: {sum((x - sum(kernel_times)/len(kernel_times))**2 for x in kernel_times)/len(kernel_times)**0.5/len(kernel_times)**0.5*1.96}")
print(f"- Confidence interval runtime time: {sum((x - sum(runtime_times)/len(runtime_times))**2 for x in runtime_times)/len(runtime_times)**0.5/len(runtime_times)**0.5*1.96}")

