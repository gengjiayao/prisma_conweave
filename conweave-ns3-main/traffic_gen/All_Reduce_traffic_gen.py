# Ring All-Reduce traffic generator
# Output lines format: "src dst 3 size start_time(seconds)"

import sys
import random
import math
import heapq
from optparse import OptionParser
from custom_rand import CustomRand

class Flow:
	def __init__(self, src, dst, size, t):
		self.src, self.dst, self.size, self.t = src, dst, size, t
	def __str__(self):
		return "%d %d 3 %d %.9f"%(self.src, self.dst, self.size, self.t)

def translate_bandwidth(b):
    if b is None or type(b) != str: return None
    if b.endswith('G'): return float(b[:-1]) * 1e9
    if b.endswith('M'): return float(b[:-1]) * 1e6
    if b.endswith('K'): return float(b[:-1]) * 1e3
    return float(b)

#We kept this for compatibility with the original code.but we don't use it in our code.
def poisson(lam):
    return -math.log(1 - random.random()) * lam

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-c", "--cdf", dest = "cdf_file", help = "the file of the traffic size cdf", default = "uniform_distribution.txt")
    parser.add_option("-n", "--nhost", dest = "nhost", help = "number of hosts")
    parser.add_option("-l", "--load", dest = "load", help = "the percentage of the traffic load to the network capacity, by default 0.3", default = "0.3")
    parser.add_option("-b", "--bandwidth", dest = "bandwidth", help = "the bandwidth of host link (G/M/K), by default 10G", default = "10G")
    parser.add_option("-t", "--time", dest = "time", help = "the total run time (s), by default 10", default = "10")
    parser.add_option("-o", "--output", dest = "output", help = "the output file", default = "tmp_traffic.txt")
    
    parser.add_option("--rounds", dest = "rounds", help = "number of communication rounds;default 2*(nhost-1)", default = None)
    parser.add_option("--step_us", dest = "step_us", help = "inter-round interval (microseconds)", default = 340)
    parser.add_option("--jitter_us", dest = "jitter_us", help = "per-flow random jitter amplitude (microseconds)", default = 10) 
    parser.add_option("--chunk_bytes", dest = "chunk_bytes", help = "per-round message size in bytes if omitted, sample from cdf", default = 4194304)
    parser.add_option("--rotate_ring", dest="rotate_ring", action="store_true", default=False,
                      help="if set, dst=(i+1+r)%N instead of (i+1)%N")
    options,args = parser.parse_args()

    if not options.nhost:
        print("please use -n to enter the number of hosts")
        sys.exit(0)

    nhost = int(options.nhost)
    bandwidth = translate_bandwidth(options.bandwidth) # parsed but not required
    total_time_s =  float(options.time)
    out_path = options.output
    simul_time_s = float(options.time)

    # AllReduce 流量不需要cdf逻辑
    # cdf = []
    # with open(options.cdf_file, 'r') as fin:
    #     for line in fin:
    #         x, y = line.strip().split()
    #         cdf.append([float(x),float(y)])

    # cr = CustomRand()
    # if not cr.setCdf(cdf):
    #     print("Error: Not valid cdf")
    #     sys.exit(0)

    #rounds = int(options.rounds) if options.rounds is not None else 2*(nhost-1)
    rounds = int(options.rounds) if options.rounds is not None else 1000000
    step_us = float(options.step_us)
    jitter_us = float(options.jitter_us)
    chunk_bytes = int(options.chunk_bytes) if options.chunk_bytes is not None else 4194304

    base_t_ns = 2_000_000_000 # 2 seconds ,consistent with the original code
    step_ns = int(step_us * 1000)
    jitter_ns = int(jitter_us * 1000)

    max_end_time_ns = base_t_ns + int(simul_time_s * 1e9)

    flows = []
    for r in range(rounds):
        round_t_ns = base_t_ns + r * step_ns

        if round_t_ns > max_end_time_ns:
            break

        for i in range(nhost):
            if options.rotate_ring:
                dst = (i + 1 + r) % nhost
            else:
                dst = (i + 1) % nhost

            size = chunk_bytes
            #强制使用固定的大包制造高压
            # if chunk_bytes is not None:
            #     size = max(1, int(chunk_bytes))
            # else:
            #     size = max(1, int(cr.rand()))

            jit = random.randint(0, jitter_ns) if jitter_ns > 0 else 0
            t_ns = round_t_ns + jit
            
            if t_ns <= max_end_time_ns:
                flows.append((t_ns, i, dst, size))

    flows.sort(key=lambda x: x[0])

    with open(out_path, "w") as fo:
        fo.write(f"{len(flows)}\n")
        for t_ns, src, dst, size in flows:
            t_s = t_ns * 1e-9
            fo.write(f"{src} {dst} 3 {size} {t_s:.9f}\n")

    print(f"Generated {len(flows)} flows. Stop time limit: {simul_time_s}s")