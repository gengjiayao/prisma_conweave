#!/usr/bin/python3
from genericpath import exists
import subprocess
import os
import time
from xmlrpc.client import boolean
import numpy as np
import copy
import shutil
import random
from datetime import datetime
import sys
import os
import argparse
from datetime import date

# randomID
random.seed(datetime.now())
MAX_RAND_RANGE = 1000000000

#####################################################
#配置文件生成
#利用Python字符串模板自动生成供C++模拟器读取的配置文件
#####################################################

# config template
config_template = """TOPOLOGY_FILE config/{topo}.txt
FLOW_FILE config/{flow}.txt

FLOW_INPUT_FILE mix/output/{id}/{id}_in.txt
CNP_OUTPUT_FILE mix/output/{id}/{id}_out_cnp.txt
FCT_OUTPUT_FILE mix/output/{id}/{id}_out_fct.txt
PFC_OUTPUT_FILE mix/output/{id}/{id}_out_pfc.txt
QLEN_MON_FILE mix/output/{id}/{id}_out_qlen.txt
VOQ_MON_FILE mix/output/{id}/{id}_out_voq.txt
VOQ_MON_DETAIL_FILE mix/output/{id}/{id}_out_voq_per_dst.txt
UPLINK_MON_FILE mix/output/{id}/{id}_out_uplink.txt
CONN_MON_FILE mix/output/{id}/{id}_out_conn.txt
EST_ERROR_MON_FILE mix/output/{id}/{id}_out_est_error.txt

QLEN_MON_START {qlen_mon_start}
QLEN_MON_END {qlen_mon_end}
SW_MONITORING_INTERVAL {sw_monitoring_interval}

FLOWGEN_START_TIME {flowgen_start_time}
FLOWGEN_STOP_TIME {flowgen_stop_time}
BUFFER_SIZE {buffer_size}

CC_MODE {cc_mode}
LB_MODE {lb_mode}
ENABLE_PFC {enabled_pfc}
ENABLE_IRN {enabled_irn}

CONWEAVE_TX_EXPIRY_TIME {cwh_tx_expiry_time}
CONWEAVE_REPLY_TIMEOUT_EXTRA {cwh_extra_reply_deadline}
CONWEAVE_PATH_PAUSE_TIME {cwh_path_pause_time}
CONWEAVE_EXTRA_VOQ_FLUSH_TIME {cwh_extra_voq_flush_time}
CONWEAVE_DEFAULT_VOQ_WAITING_TIME {cwh_default_voq_waiting_time}

ALPHA_RESUME_INTERVAL 1
RATE_DECREASE_INTERVAL 4
CLAMP_TARGET_RATE 0
RP_TIMER 300 
FAST_RECOVERY_TIMES 1
EWMA_GAIN {ewma_gain}
RATE_AI {ai}Mb/s
RATE_HAI {hai}Mb/s
MIN_RATE 100Mb/s
DCTCP_RATE_AI {dctcp_ai}Mb/s

ERROR_RATE_PER_LINK 0.0000
L2_CHUNK_SIZE 4000
L2_ACK_INTERVAL 8
L2_BACK_TO_ZERO 0

RATE_BOUND 1
HAS_WIN {has_win}
VAR_WIN {var_win}
FAST_REACT {fast_react}
MI_THRESH {mi}
INT_MULTI {int_multi}
GLOBAL_T 1
U_TARGET 0.95
MULTI_RATE 0
SAMPLE_FEEDBACK 0

ENABLE_QCN 1
USE_DYNAMIC_PFC_THRESHOLD 1
PACKET_PAYLOAD_SIZE 1000
BASE_PORT {basePort} 
OVERLAY_MAT_FILE {overlay_mat_file_name}
INDEX_TO_SWITCH_ID_MAP_FILE {index_to_switch_id_map_file}

LINK_DOWN 0 0 0
KMAX_MAP {kmax_map}
KMIN_MAP {kmin_map}
PMAX_MAP {pmax_map}
LOAD {load}
RANDOM_SEED 1
"""
#BASE_PORT OVERLAY_FILE_NAME新增占位符

# LB/CC mode matching
cc_modes = {
    "dcqcn": 1,
    "hpcc": 3,
    "timely": 7,
    "dctcp": 8,
}

lb_modes = {
    "fecmp": 0,
    "drill": 2,
    "conga": 3,
    "letflow": 6,
    "conweave": 9,
    "rl": 7,
}

topo2bdp = {
    "leaf_spine_128_100G_OS2": 5000,  # 2-tier -> all 1Gbps
    "fat_k8_100G_OS2": 156000,  # 3-tier -> all 100Gbps
}

FLOWGEN_DEFAULT_TIME = 2.0  # see /traffic_gen/traffic_gen.py::base_t


def main():
    # make directory if not exists
    #isExist = os.path.exists(os.getcwd() + "/mix/output/")
    out_root = os.path.join(os.getcwd(), "mix", "output")
    if not os.path.exists(out_root):
        os.makedirs(out_root)
        print("The new directory is created - {}".format(out_root))

    ################################################################
    #命令行参数解析
    # cc：选择拥塞控制算法，默认dcqcn
    # lb：选择负载均衡算法，默认fecmp
    # pfc：是否启用pfc，即基于优先级的流量控制
    # irn：功能作用不知
    # topo：拓扑文件名
    # simul_time：模拟时间
    # buffer：交换机缓冲区大小
    # netload：网络负载单位为百分比
    # bw：NIC网络接口卡的带宽
    # cdf：累计分布函数CDF文件名
    # enforce_win:强制使用窗口
    # sw_monitoring_interval:指定交换机采样统计队列状态的间隔
    # basePort:建立zmq连接的端口号,后续在此基础上增加交换机的序号得到对应端口号
    ################################################################

    parser = argparse.ArgumentParser(description='run simulation') 
    parser.add_argument('--cc', dest='cc', action='store',
                        default='dcqcn', help="hpcc/dcqcn/timely/dctcp (default: dcqcn)")
    parser.add_argument('--lb', dest='lb', action='store',
                        default='rl', help="fecmp/drill/conga/letflow/conweave/rl (default: rl)")
    parser.add_argument('--pfc', dest='pfc', action='store',
                        type=int, default=1, help="enable PFC (default: 1)")
    parser.add_argument('--irn', dest='irn', action='store',
                        type=int, default=0, help="enable IRN (default: 0)")
    parser.add_argument('--simul_time', dest='simul_time', action='store',
                        default='0.02', help="traffic time to simulate (up to 3 seconds) (default: 0.02)")
    parser.add_argument('--buffer', dest="buffer", action='store',
                        default='9', help="the switch buffer size (MB) (default: 9)")
    parser.add_argument('--netload', dest='netload', action='store', type=int,
                        default=6, help="Network load at NIC to generate traffic (default: 6)")
    parser.add_argument('--bw', dest="bw", action='store',
                        default='100', help="the NIC bandwidth (Gbps) (default: 100)")
    parser.add_argument('--topo', dest='topo', action='store',
                        default='leaf_spine_128_100G', help="the name of the topology file (default: leaf_spine_128_100G_OS2)")
    parser.add_argument('--cdf', dest='cdf', action='store',
                        default='AliStorage2019', help="the name of the cdf file (default: AliStorage2019)")
    parser.add_argument('--enforce_win', dest='enforce_win', action='store',
                        type=int, default=0, help="enforce to use window scheme (default: 0)")
    parser.add_argument('--sw_monitoring_interval', dest='sw_monitoring_interval', action='store',
                        type=int, default=10000, help="interval of sampling statistics for queue status (default: 10000ns)")
    parser.add_argument('--basePort',dest='basePort', type=int, default=6555, help="zmq base port(default:6555)")
    parser.add_argument('--overlay_mat_file_name',dest='overlay_mat_file_name',default="../prisma/examples/abilene/topology_files/overlay_adjacency_matrix.txt",help="path to the overlay adjacency matrix file")
    parser.add_argument('--index_to_switch_id_map_file', dest='index_to_switch_id_map_file', default="", help="path to the file mapping RL agent index to switch ID")
    print(f"来自PRISMA的命令已经传递到了conweave！！！")
    
    #新流量生成文件参数
    parser.add_argument('--traffic_mode', dest='traffic_mode', default='cdf',
                        choices=['cdf', 'allreduce', 'alltoall'],
                        help="traffic generate pattern:cdf or allreduce")

    parser.add_argument('--ar_rounds', dest='ar_rounds', type=int, default=None, help="#rounds default=2*(n_host-1)")
    parser.add_argument('--ar_step_us', dest='ar_step_us', type=float, default=340.0, help="inter-round interval in microseconds")
    parser.add_argument('--ar_jitter_us', dest='ar_jitter_us', type=float, default=10.0, help="per-flow random jitter in microseconds")
    parser.add_argument('--ar_chunk_bytes', dest='ar_chunk_bytes', type=int, default=4194304, help="per-round message size in bytes;if omitted,sample from CDF")
    parser.add_argument('--ar_rotate_ring', dest='ar_rotate_ring', type=int, default=0, help="if 1 ,dst=(i+1+r)%N;if 0,dst=(i+1)%N")
    
    parser.add_argument('--aa_rounds', dest='aa_rounds', type=int, default=1)
    parser.add_argument('--aa_chunk_bytes', dest='aa_chunk_bytes', type=int, default=4194304)
    parser.add_argument('--aa_burst_us', dest='aa_burst_us', type=float, default=5000.0)
    # #### CONWEAVE PARAMETERS ####
    # parser.add_argument('--cwh_extra_reply_deadline', dest='cwh_extra_reply_deadline', action='store',
    #                     type=int, default=4, help="extra-timeout, where reply_deadline = base-RTT + extra-timeout (default: 4us)")
    # parser.add_argument('--cwh_path_pause_time', dest='cwh_path_pause_time', action='store',
    #                     type=int, default=16, help="Time to pause the path with ECN feedback (default: 8us")
    # parser.add_argument('--cwh_extra_voq_flush_time', dest='cwh_extra_voq_flush_time', action='store',
    #                     type=int, default=16, help="Extra VOQ Flush Time (default: 8us for IRN)")
    # parser.add_argument('--cwh_default_voq_waiting_time', dest='cwh_default_voq_waiting_time', action='store',
    #                     type=int, default=400, help="Default VOQ Waiting Time (default: 400us)")
    # parser.add_argument('--cwh_tx_expiry_time', dest='cwh_tx_expiry_time', action='store',
    #                     type=int, default=1000, help="timeout value of ConWeave Tx for CLEAR signal (default: 1000us)")

    args = parser.parse_args()
    print("Base port to be written into config.txt:", args.basePort)

    # make running ID of this config
    # need to check directory exists or not
    isExist = True
    config_ID = 0
    #out_root = os.path.join(os.getcwd(), "mix", "output")
    while (isExist):
        config_ID = str(random.randrange(MAX_RAND_RANGE))
        run_dir = os.path.join(out_root, config_ID)
        isExist = os.path.exists(run_dir)

    # input parameters
    cc_mode = cc_modes[args.cc]
    lb_mode = lb_modes[args.lb]
    enabled_pfc = int(args.pfc)
    enabled_irn = int(args.irn)
    bw = int(args.bw)
    buffer = args.buffer
    topo = args.topo
    enforce_win = args.enforce_win
    cdf = args.cdf
    # sniff number of servers (needed by flow duration correction and file naming)
    with open("config/{topo}.txt".format(topo=args.topo), 'r') as f_topo:
        line = f_topo.readline().split(" ")
        n_host = int(line[0]) - int(line[1])

    flowgen_start_time = FLOWGEN_DEFAULT_TIME  # default: 2.0
    flowgen_stop_time = flowgen_start_time + float(args.simul_time)
    # if args.traffic_mode == 'allreduce':
    #     ar_rounds = args.ar_rounds if args.ar_rounds is not None else 2 * (n_host - 1)
    #     implied_duration_s = (ar_rounds * args.ar_step_us) / 1e6
    #     min_stop = FLOWGEN_DEFAULT_TIME + implied_duration_s + 0.01
    #     if flowgen_stop_time < min_stop:
    #         flowgen_stop_time = min_stop
    #         print("Auto-extend flowgen_stop_time to cover all rounds: stop=%.6fs" % flowgen_stop_time)
    if args.traffic_mode == 'allreduce':
        # 不根据轮数延长仿真时间
        print(f"流量生成会在{args.simul_time}s结束")
        # 保留ar_rounds参数，但不影响仿真时间
        ar_rounds = args.ar_rounds if args.ar_rounds is not None else 2*(n_host - 1)
    sw_monitoring_interval = int(args.sw_monitoring_interval)

    # get over-subscription ratio from topoogy name

    netload = args.netload
    oversub = int(topo.replace("\n", "").split("OS")[-1].replace(".txt", ""))
    assert (int(args.netload) % oversub == 0)
    hostload = int(args.netload) / oversub
    assert (hostload > 0)

    # Sanity checks
    if (args.cc == "timely" or args.cc == "hpcc") and args.lb == "conweave":
        raise Exception(
            "CONFIG ERROR : ConWeave currently does not support RTT-based protocols. Plz modify its logic accordingly.")
    if enabled_irn == 1 and enabled_pfc == 1:
        raise Exception(
            "CONFIG ERROR : If IRN is turn-on, then you should turn off PFC (for better perforamnce).")
    if enabled_irn == 0 and enabled_pfc == 0:
        raise Exception(
            "CONFIG ERROR : Either IRN or PFC should be true (at least one).")
    # if float(args.simul_time) < 0.005:
    #     raise Exception("CONFIG ERROR : Runtime must be larger than 5ms (= warmup interval).")

    assert (hostload >= 0 and hostload < 100)
    if args.traffic_mode == 'cdf':
        flow = "L_{load:.2f}_CDF_{cdf}_N_{n_host}_T_{time}ms_B_{bw}_flow".format(
            load=hostload, cdf=args.cdf, n_host=n_host, time=int(float(args.simul_time) * 1000), bw=bw)
    elif args.traffic_mode == 'allreduce':
        ar_rounds = args.ar_rounds if args.ar_rounds is not None else 2 * (n_host - 1)
        size_tag = ("CH{}".format(args.ar_chunk_bytes)
                    if args.ar_chunk_bytes is not None
                    else "CDF_{}".format(args.cdf))
        flow = "AR_R{R}_STEP_{S}us_J{J}us_{size}_N_{N}_B_{bw}".format(
            R=ar_rounds,
            S=int(args.ar_step_us),
            J=int(args.ar_jitter_us),
            size=size_tag,
            N=n_host,
            bw=bw,
        )
    elif args.traffic_mode == 'alltoall':
        aa_rounds = args.aa_rounds
        size_tag = "CH{}".format(args.aa_chunk_bytes)
        flow = "AA_R{R}_BURST_{B}us_{size}_N_{N}_B_{bw}".format(
              R=aa_rounds,
              B=int(args.aa_burst_us),
              size=size_tag,
              N=n_host,
              bw=bw,
        )
    else:
        raise Exception("Unknown traffic_mode: {}".format)
    # check the file exists
    out_path = os.getcwd() + "/config/" + flow + ".txt"
    if exists(out_path):
        print("Input traffic file already exists:", out_path)
    else:
        print("Generate a input traffic file... ->", out_path)
        if args.traffic_mode == 'cdf':
            cmd = ("python ./traffic_gen/traffic_gen.py "
                   "-c {cdf} -n {n_host} -l {load} -b {bw} -t {time} -o {output}").format(
                cdf=os.getcwd() + "/traffic_gen/" + args.cdf + ".txt",
                n_host=n_host, load=hostload / 100.0,
                bw=args.bw + "G", time=args.simul_time, output=out_path)
        elif args.traffic_mode == 'allreduce':
            chunk_arg = ("--chunk_bytes {cb}".format(cb=args.ar_chunk_bytes)) if args.ar_chunk_bytes else ""
            cdf_arg = ("-c " + os.getcwd() + "/traffic_gen/" + args.cdf + ".txt") if not args.ar_chunk_bytes else ""
            rotate_arg = "--rotate_ring {v}".format(v=int(args.ar_rotate_ring))

            # [修改] 显式传递 -t (时间) 和 -b (带宽)，并处理 rounds 可能为 None 的情况
            rounds_val = args.ar_rounds if args.ar_rounds is not None else 1000000 # 传个大数让生成器自己按时间切

            cmd = ("python ./traffic_gen/All_Reduce_traffic_gen.py "
                   "-n {n_host} -b {bw} -t {time} "
                   "{cdf_arg} --rounds {R} --step_us {S} --jitter_us {J} {chunk_arg} {rotate_arg} "
                   "-o {output}").format(
                n_host=n_host, bw=args.bw+"G", time=args.simul_time, 
                cdf_arg=cdf_arg,
                R=rounds_val,
                S=args.ar_step_us, J=args.ar_jitter_us, chunk_arg=chunk_arg,
                rotate_arg=rotate_arg, output=out_path)
        elif args.traffic_mode == 'alltoall':
            cmd = (
                "python ./traffic_gen/AllToAll_traffic_gen.py "
                "-n {n_host} -b {bw} -t {time} -o {output} "
                "--rounds {R} --chunk_bytes {CH} --burst_us {BU}"
            ).format(
                n_host=n_host,
                bw=args.bw + "G",
                time=args.simul_time,
                output=out_path,
                R=args.aa_rounds,
                CH=args.aa_chunk_bytes,
                BU=args.aa_burst_us,
            )
        else:
            raise Exception("Unknown traffic_mode: {}".format(args.traffic_mode))
        
        print(cmd)
        os.system(cmd)

    # sanity check - bandwidth
    with open("config/{topo}.txt".format(topo=args.topo), 'r') as f_topo:
        first_line = f_topo.readline().split(" ")
        n_host = int(first_line[0]) - int(first_line[1])
        n_link = int(first_line[2])
        i = 0
        for line in f_topo.readlines()[1:]:
            i += 1
            if (i > n_link):
                break
            parsed = line.split(" ")
            if len(parsed) > 2 and (int(parsed[0]) < n_host or int(parsed[1]) < n_host):
                assert (int(parsed[2].replace("Gbps", "")) == int(bw))
    print("All NIC bandwidth is {bw}Gbps".format(bw=bw))

    ##################################################################
    ##########              ConWeave parameters             ##########
    ##################################################################
    if (lb_mode == 9):
        cwh_extra_reply_deadline = 4  # 4us, NOTE: this is "extra" term to base RTT
        cwh_path_pause_time = 16  # 8us (K_min) or 16us

        if "leaf_spine" in topo:  # 2-tier
            cwh_extra_voq_flush_time = 16
            cwh_default_voq_waiting_time = 200
            cwh_tx_expiry_time = 300  # 300us
        elif "fat" in topo and enabled_pfc == 0 and enabled_irn == 1:  # 3-tier, IRN
            cwh_extra_voq_flush_time = 16
            cwh_default_voq_waiting_time = 300
            cwh_tx_expiry_time = 1000  # 1ms
        elif "fat" in topo and enabled_pfc == 1 and enabled_irn == 0:  # 3-tier, Lossless
            cwh_extra_voq_flush_time = 64
            cwh_default_voq_waiting_time = 600
            cwh_tx_expiry_time = 1000  # 1ms
        else:
            raise Exception(
                "Unsupported ConWeave Parameter Setup")
    else:
        #### CONWEAVE PARAMETERS (DUMMY) ####
        cwh_extra_reply_deadline = 4
        cwh_path_pause_time = 16
        cwh_extra_voq_flush_time = 64
        cwh_default_voq_waiting_time = 400
        cwh_tx_expiry_time = 1000

    ##################################################################

    # make directory if not exists
    # isExist = os.path.exists(os.getcwd() + "/mix/output/" + config_ID + "/")
    # assert (not isExist)
    # # if not isExist:
    # os.makedirs(os.getcwd() + "/mix/output/" + config_ID + "/")
    # print("The new directory is created  - {}".format(os.getcwd() +
    #       "/mix/output/" + config_ID + "/"))

    # config_name = os.getcwd() + "/mix/output/" + config_ID + "/config.txt"
    # print("Config filename:{}".format(config_name))
    os.makedirs(run_dir)
    print("The new directory is created  - {}".format(run_dir))

    os.environ["MIX_OUTPUT_DIR"] = run_dir
    print("MIX_OUTPUT_DIR set to:", os.environ["MIX_OUTPUT_DIR"])

    config_name = os.path.join(run_dir, "config.txt")
    print("Config filename:{}".format(config_name))

    # By default, DCQCN uses no window (rate-based).
    has_win = 0
    var_win = 0
    if (cc_mode == 3 or cc_mode == 8 or enforce_win == 1):  # HPCC or DCTCP or enforcement
        has_win = 1
        var_win = 1
        if enforce_win == 1:
            print("### INFO: Enforced to use window scheme! ###")

    # record to history
    simulday = datetime.now().strftime("%m/%d/%y")
    with open("./mix/.history", "a") as history:
        history.write("{simulday},{config_ID},{cc_mode},{lb_mode},{cwh_tx_expiry_time},{cwh_extra_reply_deadline},{cwh_path_pause_time},{cwh_extra_voq_flush_time},{cwh_default_voq_waiting_time},{pfc},{irn},{has_win},{var_win},{topo},{bw},{cdf},{load},{time}\n".format(
            simulday=simulday,
            config_ID=config_ID,
            cc_mode=cc_mode,
            lb_mode=lb_mode,
            cwh_tx_expiry_time=cwh_tx_expiry_time,
            cwh_extra_reply_deadline=cwh_extra_reply_deadline,
            cwh_path_pause_time=cwh_path_pause_time,
            cwh_extra_voq_flush_time=cwh_extra_voq_flush_time,
            cwh_default_voq_waiting_time=cwh_default_voq_waiting_time,
            pfc=enabled_pfc,
            irn=enabled_irn,
            has_win=has_win,
            var_win=var_win,
            topo=topo,
            bw=bw,
            cdf=cdf,
            load=netload,
            time=args.simul_time,
        ))

    # 1 BDP calculation
    if topo2bdp.get(topo) == None:
        print("ERROR - topology is not registered in run.py!!", flush=True)
        return
    bdp = int(topo2bdp[topo])
    print("1BDP = {}".format(bdp))

    # DCQCN parameters (NOTE: HPCC's 400KB/1600KB is too large, although used in Microsoft)
    kmax_map = "6 %d %d %d %d %d %d %d %d %d %d %d %d" % (
        bw*200000000, 400, bw*500000000, 400, bw*1000000000, 400, bw*2*1000000000, 400, bw*2500000000, 400, bw*4*1000000000, 400)
    kmin_map = "6 %d %d %d %d %d %d %d %d %d %d %d %d" % (
        bw*200000000, 100, bw*500000000, 100, bw*1000000000, 100, bw*2*1000000000, 100, bw*2500000000, 100, bw*4*1000000000, 100)
    pmax_map = "6 %d %d %d %d %d %.2f %d %.2f %d %.2f %d %.2f" % (
        bw*200000000, 0.2, bw*500000000, 0.2, bw*1000000000, 0.2, bw*2*1000000000, 0.2, bw*2500000000, 0.2, bw*4*1000000000, 0.2)

    # queue monitoring
    qlen_mon_start = flowgen_start_time
    qlen_mon_end = flowgen_stop_time

    if (cc_mode == 1):  # DCQCN
        ai = 10 * bw / 25
        hai = 25 * bw / 25
        dctcp_ai = 1000
        fast_react = 0
        mi = 0
        int_multi = 1
        ewma_gain = 0.00390625

        config = config_template.format(id=config_ID, topo=topo, flow=flow,
                                        qlen_mon_start=qlen_mon_start, qlen_mon_end=qlen_mon_end, flowgen_start_time=flowgen_start_time,
                                        flowgen_stop_time=flowgen_stop_time, sw_monitoring_interval=sw_monitoring_interval,
                                        load=netload, buffer_size=buffer, lb_mode=lb_mode, cwh_tx_expiry_time=cwh_tx_expiry_time,
                                        cwh_extra_reply_deadline=cwh_extra_reply_deadline, cwh_default_voq_waiting_time=cwh_default_voq_waiting_time,
                                        cwh_path_pause_time=cwh_path_pause_time, cwh_extra_voq_flush_time=cwh_extra_voq_flush_time,
                                        enabled_pfc=enabled_pfc, enabled_irn=enabled_irn,
                                        cc_mode=cc_mode,
                                        ai=ai, hai=hai, dctcp_ai=dctcp_ai,
                                        has_win=has_win, var_win=var_win,
                                        fast_react=fast_react, mi=mi, int_multi=int_multi, ewma_gain=ewma_gain,
                                        kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map, basePort = args.basePort,
                                        overlay_mat_file_name=args.overlay_mat_file_name,
                                        index_to_switch_id_map_file=args.index_to_switch_id_map_file,)#8.19新增overlay 9.2新增map
        print("Final config content preview:\n")
        #print(config)

    else:
        print("unknown cc:{}".format(args.cc))

    with open(config_name, "w") as file:
        file.write(config)

    # run program
    print("Running simulation...")
    output_log = config_name.replace(".txt", ".log")
    run_command = "python2 ./waf --run 'scratch/network-load-balance {config_name}' > {output_log} 2>&1".format(
        config_name=config_name, output_log=output_log)
    with open("./mix/.history", "a") as history:
        history.write(run_command + "\n")
        history.write(
            "python2 ./waf --run 'scratch/network-load-balance' --command-template='gdb --args %s {config_name}'\n".format(
                config_name=config_name)
        )
        history.write("\n")

    print(run_command)
    os.system("python2 ./waf --run 'scratch/network-load-balance {config_name}' > {output_log} 2>&1".format(
        config_name=config_name, output_log=output_log))

    ####################################################
    #                 Analyze the output FCT           #
    ####################################################
    # NOTE: collect data except warm-up and cold-finish period
    fct_analysis_time_limit_begin = int(
        flowgen_start_time * 1e9) # + int(0.005 * 1e9)  # warmup
    fct_analysistime_limit_end = int(
        flowgen_stop_time * 1e9) + int(0.05 * 1e9)  # extra term

    print("Analyzing output FCT...")
    print("python3 fctAnalysis.py -id {config_ID} -dir {dir} -bdp {bdp} -sT {fct_analysis_time_limit_begin} -fT {fct_analysistime_limit_end} > /dev/null 2>&1".format(
        config_ID=config_ID, dir=os.getcwd(), bdp=bdp, fct_analysis_time_limit_begin=fct_analysis_time_limit_begin, fct_analysistime_limit_end=fct_analysistime_limit_end))
    os.system("python3 fctAnalysis.py -id {config_ID} -dir {dir} -bdp {bdp} -sT {fct_analysis_time_limit_begin} -fT {fct_analysistime_limit_end} > /dev/null 2>&1".format(
        config_ID=config_ID, dir=os.getcwd(), bdp=bdp, fct_analysis_time_limit_begin=fct_analysis_time_limit_begin, fct_analysistime_limit_end=fct_analysistime_limit_end))

    if lb_mode == 9: # ConWeave Logging
        ################################################################
        #             Analyze hardware resource of ConWeave            #
        ################################################################
        # NOTE: collect data except warm-up and cold-finish period
        queue_analysis_time_limit_begin = int(
            flowgen_start_time * 1e9) + int(0.005 * 1e9)  # warmup
        queue_analysistime_limit_end = int(flowgen_stop_time * 1e9)
        print("Analyzing output Queue...")
        print("python3 queueAnalysis.py -id {config_ID} -dir {dir} -sT {queue_analysis_time_limit_begin} -fT {queue_analysistime_limit_end} > /dev/null 2>&1".format(
            config_ID=config_ID, dir=os.getcwd(), queue_analysis_time_limit_begin=queue_analysis_time_limit_begin, queue_analysistime_limit_end=queue_analysistime_limit_end))
        os.system("python3 queueAnalysis.py -id {config_ID} -dir {dir} -sT {queue_analysis_time_limit_begin} -fT {queue_analysistime_limit_end} > /dev/null 2>&1".format(
            config_ID=config_ID, dir=os.getcwd(), queue_analysis_time_limit_begin=queue_analysis_time_limit_begin, queue_analysistime_limit_end=queue_analysistime_limit_end,
            monitoringInterval=sw_monitoring_interval))  # TODO: parameterize

    print("\n\n============== Done ============== ")

    # Post-process：训练提速阶段暂时关闭曲线生成（避免运行时 I/O 与渲染开销）
    # 如需开启，请恢复此段。


if __name__ == "__main__":
    main()

