import sys
import random
from optparse import OptionParser

BASE_TIME_NS = 2000000000  # 2.0 seconds, keep consistent with original traffic_gen.py


def translate_bandwidth(b):
    """Convert bandwidth string (e.g. '100G') to bps."""
    if b is None:
        return None
    if not isinstance(b, str):
        return None

    b = b.strip().upper()
    if b.endswith('G'):
        return float(b[:-1]) * 1e9
    if b.endswith('M'):
        return float(b[:-1]) * 1e6
    if b.endswith('K'):
        return float(b[:-1]) * 1e3
    # plain number: assume bps
    return float(b)


if __name__ == "__main__":
    parser = OptionParser()

    # ===== 1. 基础参数（与原 traffic_gen 风格对齐） =====
    parser.add_option("-n", "--nhost", dest="nhost",
                      help="number of hosts")
    parser.add_option("-b", "--bandwidth", dest="bandwidth", default="100G",
                      help="NIC bandwidth (G/M/K), default: 100G")
    parser.add_option("-t", "--time", dest="time", default="0.05",
                      help="STRICT traffic generation window in seconds (default: 0.05)")
    parser.add_option("-o", "--output", dest="output", default="traffic.txt",
                      help="output traffic file path")

    # 兼容原 traffic_gen 的占位参数，防止 run.py 误传时报错
    parser.add_option("-c", "--cdf", dest="cdf_file",
                      help="ignored in all-to-all mode (for compatibility)")
    parser.add_option("-l", "--load", dest="load",
                      help="ignored in all-to-all mode (for compatibility)")

    # ===== 2. All-to-All 专用参数 =====
    parser.add_option("--rounds", dest="rounds", type="int", default=1,
                      help="number of All-to-All epochs (default: 1)")
    parser.add_option("--chunk_bytes", dest="chunk_bytes", type="int", default=4194304,
                      help="message size per (src,dst) flow in bytes, default: 4MB")
    parser.add_option("--burst_us", dest="burst_us", type="float", default=5000.0,
                      help="per-destination incast burst window in microseconds (default: 5000us)")
    parser.add_option("--seed", dest="seed", type="int", default=None,
                      help="random seed for reproducibility (default: None)")

    options, args = parser.parse_args()

    # ===== 3. 参数解析与基本校验 =====
    if not options.nhost:
        print("Error: please use -n to specify number of hosts")
        sys.exit(1)

    nhost = int(options.nhost)
    bw_bps = translate_bandwidth(options.bandwidth)
    if bw_bps is None or bw_bps <= 0:
        print("Error: invalid bandwidth:", options.bandwidth)
        sys.exit(1)

    simul_time_s = float(options.time)   # strict window for flow START times
    output_file = options.output

    rounds = int(options.rounds)
    if rounds <= 0:
        print("Error: --rounds must be >= 1")
        sys.exit(1)

    chunk_bytes = int(options.chunk_bytes)
    if chunk_bytes <= 0:
        print("Error: --chunk_bytes must be > 0")
        sys.exit(1)

    burst_us = float(options.burst_us)
    if burst_us <= 0:
        print("Error: --burst_us must be > 0")
        sys.exit(1)
    burst_s = burst_us / 1e6

    if simul_time_s <= 0:
        print("Error: --time must be > 0")
        sys.exit(1)

    # clamp burst window if longer than simul_time
    if burst_s > simul_time_s:
        print(f"[WARN] burst_us ({burst_us:.1f} us) > simul_time ({simul_time_s * 1e6:.1f} us).")
        print("       Clamping burst_us to simul_time.")
        burst_s = simul_time_s
        burst_us = simul_time_s * 1e6

    if options.seed is not None:
        random.seed(options.seed)

    # ===== 4. 计算主机平均负载（避免重蹈 168% 覆辙） =====
    # 每个 host 在每一轮需要向 (N-1) 个目标各发一个 chunk
    flows_per_host = (nhost - 1) * rounds
    bits_per_host = flows_per_host * chunk_bytes * 8.0

    # 这里的平均负载是以整个 simul_time 为窗口的平均注入速率
    offered_load = bits_per_host / (bw_bps * simul_time_s)

    print("--- All-to-All Configuration ---")
    print(f"  Hosts            : {nhost}")
    print(f"  Rounds           : {rounds}")
    print(f"  Chunk Size       : {chunk_bytes} bytes ({chunk_bytes / 1024.0 / 1024.0:.2f} MB)")
    print(f"  NIC Bandwidth    : {options.bandwidth} ({bw_bps / 1e9:.1f} Gbps)")
    print(f"  Time Window      : {simul_time_s:.6f} s (strict for flow START times)")
    print(f"  Burst per-dest   : {burst_us:.1f} us")
    print(f"  Flows per Host   : {flows_per_host}")
    print(f"  Avg Host Load    : {offered_load * 100.0:.1f}% of line rate")

    if offered_load > 1.0:
        print("  [WARN] Average host load > 100% over the specified time window.")
        print("         This will create persistent queue build-up (extreme stress).")
    elif offered_load > 0.9:
        print("  [INFO] Average host load in 90%-100% range (saturation edge).")
    else:
        print("  [INFO] Average host load < 90% (moderate load).")

    # ===== 5. 生成 All-to-All + Incast 型流量 =====
    #
    # 设计思想：
    #   - 对于每个目的节点 d：
    #       1) 抽一个基准时间 t0_d ~ U(0, simul_time_s - burst_s)
    #       2) 所有 src!=d 发往 d 的 flow 的 start_time ~ t0_d + U(0, burst_s)
    #     -> 这样对于每个 d，在 [t0_d, t0_d+burst_s] 内出现 N-1 个并发到达，形成 ToR 级别 Incast。
    #   - 对于每个 src，它向不同 dst 发送的时间由各个 t0_d 叠加而成，整体上比较均匀分布在 [0, simul_time_s] 内。
    #
    flows = []

    for r in range(rounds):
        # 每一轮单独为每个目的地选择一个 t0_d（增加一些随机性）
        for dst in range(nhost):
            # t0_d ∈ [0, simul_time_s - burst_s]
            t0_d = random.uniform(0.0, simul_time_s - burst_s)

            for src in range(nhost):
                if src == dst:
                    continue

                # 在 [t0_d, t0_d + burst_s] 内加一点随机扰动
                local_offset = random.uniform(0.0, burst_s)
                start_time_s = t0_d + local_offset

                # 严格保证不超过 simul_time_s（考虑浮点误差，min 一下）
                if start_time_s > simul_time_s:
                    start_time_s = simul_time_s

                start_ns = BASE_TIME_NS + int(start_time_s * 1e9)
                flows.append((start_ns, src, dst, chunk_bytes))

    # 按时间排序
    flows.sort(key=lambda x: x[0])

    # ===== 6. 写入文件（完全兼容原 traffic 格式） =====
    # 第一行：flow 总数
    # 后续： src dst 3 size start_time(s)
    try:
        with open(output_file, "w") as fo:
            fo.write(f"{len(flows)}\n")
            for t_ns, src, dst, size in flows:
                t_s = t_ns * 1e-9
                fo.write(f"{src} {dst} 3 {size} {t_s:.9f}\n")

        first_t = (flows[0][0] - BASE_TIME_NS) * 1e-9 if flows else 0.0
        last_t = (flows[-1][0] - BASE_TIME_NS) * 1e-9 if flows else 0.0
        print(f"  Generated flows  : {len(flows)} -> {output_file}")
        print(f"  First start      : {first_t:.9f} s after BASE_TIME")
        print(f"  Last start       : {last_t:.9f} s after BASE_TIME "
              f"(<= {simul_time_s:.9f} s)")
    except IOError as e:
        print(f"Error writing to file: {e}")
        sys.exit(1)
