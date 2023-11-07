import sys

if len(sys.argv) < 2:
  print("Usage: python collect_metrics.py <path_to_dataset>")
  sys.exit(1)

dataset_path = sys.argv[1]

with open(dataset_path, 'r') as f:
  lines = f.readlines()

sg_8_kernel = []
sg_8_total = []
sg_16_kernel = []
sg_16_total = []
sg_32_kernel = []
sg_32_total = []

for i in range(len(lines)):
  if "SubGroup size  8" in lines[i]:
    sg_8_kernel.append(float(lines[i+1].split()[3]))
    sg_8_total.append(float(lines[i+2].split()[3]))
  elif "SubGroup size 16" in lines[i]:
    sg_16_kernel.append(float(lines[i+1].split()[3]))
    sg_16_total.append(float(lines[i+2].split()[3]))
  elif "SubGroup size 32" in lines[i]:
    sg_32_kernel.append(float(lines[i+1].split()[3]))
    sg_32_total.append(float(lines[i+2].split()[3]))

# Print avarage
print("[Avarage]")
print("SubGroup size  8: kernel: ", sum(sg_8_kernel)/len(sg_8_kernel), " total: ", sum(sg_8_total)/len(sg_8_total))
print("SubGroup size 16: kernel: ", sum(sg_16_kernel)/len(sg_16_kernel), " total: ", sum(sg_16_total)/len(sg_16_total))
print("SubGroup size 32: kernel: ", sum(sg_32_kernel)/len(sg_32_kernel), " total: ", sum(sg_32_total)/len(sg_32_total))
print()

# Print min and max
print("[Min]")
print("SubGroup size  8: kernel: ", min(sg_8_kernel), " total: ", min(sg_8_total))
print("SubGroup size 16: kernel: ", min(sg_16_kernel), " total: ", min(sg_16_total))
print("SubGroup size 32: kernel: ", min(sg_32_kernel), " total: ", min(sg_32_total))
print("[Max]")
print("SubGroup size  8: kernel: ", max(sg_8_kernel), " total: ", max(sg_8_total))
print("SubGroup size 16: kernel: ", max(sg_16_kernel), " total: ", max(sg_16_total))
print("SubGroup size 32: kernel: ", max(sg_32_kernel), " total: ", max(sg_32_total))
print()

# Print standard deviation
print("[Standard deviation]")
import statistics
print("SubGroup size  8: kernel: ", statistics.stdev(sg_8_kernel), " total: ", statistics.stdev(sg_8_total))
print("SubGroup size 16: kernel: ", statistics.stdev(sg_16_kernel), " total: ", statistics.stdev(sg_16_total))
print("SubGroup size 32: kernel: ", statistics.stdev(sg_32_kernel), " total: ", statistics.stdev(sg_32_total))
print()

# Print harmonic mean
print("[Harmonic mean]")
import scipy.stats
print("SubGroup size  8: kernel: ", scipy.stats.hmean(sg_8_kernel), " total: ", scipy.stats.hmean(sg_8_total))
print("SubGroup size 16: kernel: ", scipy.stats.hmean(sg_16_kernel), " total: ", scipy.stats.hmean(sg_16_total))
print("SubGroup size 32: kernel: ", scipy.stats.hmean(sg_32_kernel), " total: ", scipy.stats.hmean(sg_32_total))
print()
