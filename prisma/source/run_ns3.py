"""
    Run the ns3 simulator
    Args: 
        params(dict): parameter dict
        configure(bool): if True, run ns3 configure
    Returns:
        subprocess.Popen: owned simulator launcher process
""" 


__author__ = "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__version__ = "0.1.0"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"

def run_ns3(params, configure=True):
    
    ## import libraries
    import os
    import shlex
    import subprocess
    
    ## check if ns3-gym is in the folder，首先检查main传来的ns3_sim_path下是否有waf文件是一个合法的ns3环境
    if "waf" not in os.listdir(params["ns3_sim_path"]):
        raise Exception(f'Unable to locate ns3-gym in the folder : {params["ns3_sim_path"]}')
        
    ## store current folder path
    current_folder_path = os.getcwd()

    ## Copy prisma into ns-3 folder 将本地项目的PRISMA项目的仿真脚本./ns3/复制到 NS3 仿真器中的 scratch/prisma 文件夹，同时更新了ipv4-interface.cc到NS3 的 src/internet/model 中
    #os.system(f'rsync -r ./ns3/* {params["ns3_sim_path"].rstrip("/")}/scratch/prisma')
    #os.system(f'rsync -r ./ns3_model/ipv4-interface.cc {params["ns3_sim_path"].rstrip("/")}/src/internet/model')

    ## go to ns3 dir
    os.chdir(params["ns3_sim_path"]) #ns3_sim_path被改为了conweave的main.py文件
    
    ## run ns3 configure
    #configure_command = './waf -d optimized configure'
    #if configure:
        #os.system('./waf configure')
    #上文的两部分进入NS3仿真目录，进行构建。若设定 configure=True，则调用 waf configure，用于（重新）配置项目构建状态

    '''
    ## run NS3 simulator
    ns3_params_format = ('prisma --simSeed={} --openGymPort={} --simTime={} --AvgPacketSize={} '
                        '--LinkDelay={} --LinkRate={} --MaxBufferLength={} --load_factor={} '
                        '--adj_mat_file_name={} --overlay_mat_file_name={} '
                        '--node_intensity_file_name={} --signaling={} --AgentType={} --signalingType={} '
                        '--syncStep={} --lossPenalty={} --activateOverlaySignaling={} '
                        '--train={} --movingAverageObsSize={} --activateUnderlayTraffic={} --opt_rejected_file_name={} '
                        '--map_overlay_file_name={} --pingAsObs={} --bigSignalingSize={} --pingPacketIntervalTime={}'.format( params["seed"],
                                             params["basePort"],
                                             str(params["simTime"]),
                                             params["packet_size"],
                                             str(params["link_delay"])+"ms",
                                             str(params["link_cap"]) + "bps",
                                             str(params["max_out_buffer_size"]) + "B",
                                             params["load_factor"],
                                             params["physical_adjacency_matrix_path"],
                                             params["overlay_adjacency_matrix_path"],
                                            #  params["node_coordinates_path"],
                                             params["traffic_matrix_path"],
                                             bool(params["signalingSim"]),
                                             params["agent_type"],
                                             params["signaling_type"],
                                             params["sync_step"],
                                             params["loss_penalty"],
                                             bool(params["activateOverlay"]),
                                             bool(params["train"]),
                                             params["movingAverageObsSize"],
                                             bool(params["activateUnderlayTraffic"]),
                                             params["opt_rejected_path"],
                                             params["map_overlay_path"],
                                             bool(params["pingAsObs"]),
                                             params["bigSignalingSize"],
                                             params["pingPacketIntervalTime"]
                                             ))
    run_ns3_command = shlex.split(f'./waf --run "{ns3_params_format}"')
    proc = subprocess.Popen(run_ns3_command)
    print(f"Running ns3 simulator with process id: {proc.pid}")
    #用params字典中传进来的参数生成NS3仿真命令的字符串，并通过subprocess.Popen 异步启动
    os.chdir(current_folder_path)
    return proc
    '''
    traffic_seed_arg = (
        ""
        if params.get("traffic_seed") is None
        else f'--traffic_seed {int(params["traffic_seed"])} '
    )
    conweave_timing_args = "".join(
        f'--{name} {int(params[name])} '
        for name in (
            "cwh_extra_reply_deadline",
            "cwh_path_pause_time",
            "cwh_extra_voq_flush_time",
            "cwh_default_voq_waiting_time",
            "cwh_tx_expiry_time",
        )
        if params.get(name) is not None
    )
    runpy_cmd = (
        f'python3 run.py '
        f'--cc {params["cc"]} '
        f'--lb {params["lb"]} '
        f'--pfc {params["pfc"]} '
        f'--irn {params["irn"]} '
        f'--simul_time {params["simul_time"]} '
        f'--buffer {params["buffer"]} '
        f'--netload {params["netload"]} '
        f'--bw {params["bw"]} '
        f'--topo {params["topo"]} '
        f'--cdf {params["cdf"]} '
        f'--enforce_win {params["enforce_win"]} '
        f'--sw_monitoring_interval {params["sw_monitoring_interval"]} '
        f'--seed {params["seed"]} '
        f'{traffic_seed_arg}'
        f'{conweave_timing_args}'
        f'--session_name {params["session_name"]} '
        f'--basePort {params["basePort"]} '
        f'--overlay_mat_file_name {params["overlay_adjacency_matrix_path"]} ' #8.19新增传输overlay矩阵路径到conweave的启动命令 
        f'--index_to_switch_id_map_file {params["index_to_switch_id_map_path"]}' #9.2新增传递映射文件路径
        ##todo：扩展用
        #f'--conweave_use_prisma {params["conweave_use_prisma"]} '
    )
    print("即将调用 run.py 传递参数中")
    proc = subprocess.Popen(
        shlex.split(runpy_cmd),
        cwd=params["ns3_sim_path"],
        preexec_fn=os.setsid
    )
    print(f"启动了conweave的run.py，pid为：{proc.pid}")

    os.chdir(current_folder_path)
    return proc


"""
import os
import subprocess
import shlex
"""
#def run_ns3(params, configure=True):
"""
    Run the conweave-based ns3 simulator
    Args:
        params(dict): parameter dict, must include "conweave_path"
    Returns:
        proc: process id of the conweave simulator process
"""
"""
    current_folder_path = os.getcwd()

    # 切换目录到 conweave 项目
    os.chdir(params["conweave_path"])

    # 构造命令行参数
    cmd = (f'python3 run.py '
           f'--cc {params["cc"]} '
           f'--lb {params["lb"]} '
           f'--pfc {params["pfc"]} '
           f'--irn {params["irn"]} '
           f'--simul_time {params["simul_time"]} '
           f'--buffer {params["buffer"]} '
           f'--netload {params["netload"]} '
           f'--bw {params["bw"]} '
           f'--topo {params["topo"]} '
           f'--cdf {params["cdf"]} '
           f'--enforce_win {params["enforce_win"]} '
           f'--sw_monitoring_interval {params["sw_monitoring_interval"]}')
    run_conweave_command = shlex.split(cmd)

    # 启动 conweave 的 run.py 脚本
    proc = subprocess.Popen(run_conweave_command)

    print(f"通过run.py启动了conweave项目！ process id: {proc.pid}")

    os.chdir(current_folder_path)
    return proc.pid
"""
