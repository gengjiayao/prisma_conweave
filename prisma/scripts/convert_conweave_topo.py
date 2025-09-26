"""
将conweave拓扑转换为PRISMA使用的邻接矩阵的拓扑
"""

import argparse
import numpy as np

def parse_conweave_topo_file(topo_file_path):
    """
    读取conweave拓扑，生成邻接矩阵

    Args:
        topo_file_path(str):Direction of conweave topo file

    Return:
        adj_matrix(np.ndarry):adjacency matrix(N*N)
    """

    #打开文件
    with open(topo_file_path,'r') as f:
        lines = f.readlines()

    #解析文件头：节点数，交换机个数，链路数
    #例如：144 16 192
    header = lines[0].strip().split()
    num_switches = int(header[1])#交换机总数

    switch_ids = list(map(int, lines[1].strip().split()))
    switch_id_set = set(switch_ids)
    switch_id_to_matrix_idx = {nid: i for i, nid in enumerate(switch_ids)}#原始交换机id到矩阵索引

    #初始化邻接矩阵
    adj_matrix = np.zeros((num_switches,num_switches),dtype = int)
        
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < 2: #如果出现这种情况，该行不合格
            continue
        src = int(parts[0])
        dst = int(parts[1])

        if src in switch_id_set and dst in switch_id_set:
            i = switch_id_to_matrix_idx[src]
            j = switch_id_to_matrix_idx[dst]
            adj_matrix[i][j] = 1
            adj_matrix[j][i] = 1

    return adj_matrix, switch_ids

def main():
    parser = argparse.ArgumentParser(description="Convert conweave topology file into PRISMA adjacency matrix format.\n")

    parser.add_argument('--in', dest='input_path', type=str, required=True, help='conweave拓扑文件路径')
    parser.add_argument('--out', dest='output_path', type=str, required=True, help='PRISMA邻接矩阵输出路径')
    parser.add_argument('--map', dest='map_path',type=str,required=True,help='overlay_adjacency_matrix序号和交换机编号映射文件存放路径')

    args = parser.parse_args()

    adj_matrix,switch_ids = parse_conweave_topo_file(args.input_path)

    np.savetxt(args.output_path, adj_matrix,fmt='%d')

    print(f"已经生成邻接矩阵文件：{args.output_path},维度是：{adj_matrix.shape}")

    if args.map_path:
        with open(args.map_path,'w') as f:
            for idx, sid in enumerate(switch_ids):
                f.write(f"{idx} {sid}\n")
        print(f"已保存overlay_adjacency_matrix序号和交换机编号映射文件：{args.map_path}")

if __name__ == '__main__':
    main()
