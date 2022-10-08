import findspark
findspark.init() # this must be executed before the below import
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark import SparkFiles
import ray
import time
import rtree
from rtree import index
import pandas as pd
import numpy as np
from numpy import genfromtxt
import threading
import pyarrow as pa
import pyarrow.parquet as pq
from partition_tree import PartitionTree
conf = SparkConf().setAll([("spark.executor.memory", "24g"),("spark.driver.memory","24g"),
                           ("spark.memory.offHeap.enabled",True),("spark.memory.offHeap.size","16g"),
                          ("spark.driver.maxResultSize", "16g")])

sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
import os
os.environ['HADOOP_HOME'] = '/home/liupengju/hadoop'
os.environ['JAVA_HOME'] = '/home/liupengju/java/jdk1.8.0_281'
os.environ['ARROW_LIBHDFS_DIR'] = '/home/liupengju/hadoop/lib/native'

def process_chunk_row(row, used_dims, partition_tree, pid_data_dict, count, k):
    if count[0] % 100000 == 0:
        print('proces',k,'has routed',count[0],'rows')
    count[0] += 1
    row_numpy = row.to_numpy()
    row_point = row_numpy[used_dims].tolist()
    pids = [0]
    try:
        pids = partition_tree.get_pid_for_data_point(row_point)
    except:
        print(row_point)
    if isinstance(pids,list):
        for pid in pids:
            if pid in pid_data_dict:
                pid_data_dict[pid]+=[row_numpy.tolist()]
            else:
                pid_data_dict[pid]=[row_numpy.tolist()]

@ray.remote
def process_chunk(chunk, used_dims, partition_path, k, partition_tree):
    print("enter data routing process", k, '..')
    pid_data_dict = {}
    count = [0]
    chunk.apply(lambda row: process_chunk_row(row, used_dims, partition_tree, pid_data_dict, count, k), axis=1)
    dict_id = ray.put(pid_data_dict)
    print("exit data routing process", k, ".")
    return dict_id

@ray.remote
def merge_epochs(parameters):
    #fs = pa.hdfs.connect()
    pids, epoch_count, hdfs_path, fs, merge_process = parameters
    for pid in pids:
        parquets = []
        for epoch in range(epoch_count):
            path = hdfs_path + "epoch_" + str(epoch) + '/partition_' + str(pid)+'.parquet'
            #print(path)
            try:
                par = pq.read_table(path)
                parquets.append(par)
            except:
                continue
        print("process", merge_process, "pid", pid, " len parquets (epochs):", len(parquets))
        if len(parquets) == 0:
            continue
        merged_parquet = pa.concat_tables(parquets)
        merge_path = hdfs_path + 'merged/partition_' + str(pid)+'.parquet'
        with fs.open(merge_path,'wb') as f:
            pq.write_table(merged_parquet, f)
        # fw = fs.open(merge_path,'wb')
        # pq.write_table(merged_parquet, fw)
        # fw.close()
    print('exit merge process', merge_process)

def merge_dict(base_dict, new_dict):
    for key, val in new_dict.items():
        if key in base_dict:
            base_dict[key] += val
        else:
            base_dict[key] = val
    new_dict.clear()

def dump_dict_2_hdfs_epoch(merged_dict, column_names, hdfs_path, fs, epoch):
    #print('= = = start dumping in main thread = = =')
    for pid, val in merged_dict.items():
        #print("writing to pid:",pid)
        path = hdfs_path + 'epoch_'+ str(epoch) +'/partition_' + str(pid) + '.parquet'
        pdf = pd.DataFrame(val, columns=column_names)
        adf = pa.Table.from_pandas(pdf)
        #fw = fs.open(path, 'wb')
        with fs.open(path,'wb') as f:
            pq.write_table(adf, f)
        # fw = fs.open(path,'wb') # it seems the new version does not have the open function
        # pq.write_table(adf, fw)
        # fw.close()
    #print('= = = exit dumping = = =')


def batch_data_parallel(table_path, partition_path, chunk_size, used_dims, hdfs_path, num_dims, num_process,
                        hdfs_private_ip):
    begin_time = time.time()

    ray.init(num_cpus=num_process)

    # column names for pandas dataframe
    cols = [i for i in range(num_dims)]
    col_names = ['_c' + str(i) for i in range(num_dims)]
    # pyarrow parquent append
    # fs = pa.fs.HadoopFileSystem(hdfs_private_ip, port=9001, user='hdfs', replication=1)
    fs=pa.hdfs.connect(host=hdfs_private_ip, port=9001, user='liupengju')
    partition_tree = PartitionTree(len(used_dims))
    partition_tree.load_tree(partition_path)

    # chunks
    chunk_count = 0
    epoch_count = 0

    # collect object refs
    result_ids = []
    last_batch_ids = []
    first_loop = True

    for chunk in pd.read_table(table_path, delimiter='|', usecols=cols, names=col_names, chunksize=chunk_size):
        print('reading chunk: ', chunk_count)

        chunk_id = ray.put(chunk)
        result_id = process_chunk.remote(chunk_id, used_dims, partition_path, chunk_count, partition_tree)

        del chunk_id
        result_ids.append(result_id)
        del result_id

        # after all process allocated a chunk, process and dump the data
        if chunk_count % num_process == num_process - 1:

            if first_loop:
                first_loop = False
                last_batch_ids = result_ids.copy()
                result_ids.clear()
                chunk_count += 1
                continue
            else:
                print("= = = Process Dump For Chunk", chunk_count - 2 * num_process + 1, "to",
                      chunk_count - num_process, "= = =")
                base_dict = {}
                while len(last_batch_ids):
                    done_id, last_batch_ids = ray.wait(last_batch_ids)
                    dict_id = ray.get(done_id[0])
                    result_dict = ray.get(dict_id)
                    merge_dict(base_dict, result_dict)
                dump_dict_2_hdfs_epoch(base_dict, col_names, hdfs_path, fs,
                                       epoch_count)  # consider whether we should use another process
                epoch_count += 1
                base_dict.clear()
                print("= = = Finish Dump For Chunk", chunk_count - 2 * num_process + 1, "to", chunk_count - num_process,"= = =")
                last_batch_ids = result_ids.copy()
                result_ids.clear()

            current_time = time.time()
            time_elapsed = current_time - begin_time
            print("= = = TOTAL PROCESSED SO FAR:", (chunk_count - num_process + 1) * chunk_size, "ROWS. TIME SPENT:",time_elapsed, "SECONDS = = =")

        chunk_count += 1

    # process the last few batches
    print("= = = Process Dump For Last Few Chunks = = =")
    base_dict = {}
    while len(last_batch_ids):
        done_id, last_batch_ids = ray.wait(last_batch_ids)
        dict_id = ray.get(done_id[0])
        result_dict = ray.get(dict_id)
        merge_dict(base_dict, result_dict)
    dump_dict_2_hdfs_epoch(base_dict, col_names, hdfs_path, fs, epoch_count)
    epoch_count += 1
    base_dict.clear()
    last_batch_ids.clear()

    base_dict = {}
    while len(result_ids):
        done_id, result_ids = ray.wait(result_ids)
        dict_id = ray.get(done_id[0])
        result_dict = ray.get(dict_id)
        merge_dict(base_dict, result_dict)
    result_ids.clear()  # clear up the references
    dump_dict_2_hdfs_epoch(base_dict, col_names, hdfs_path, fs, epoch_count)
    epoch_count += 1
    base_dict.clear()
    result_ids.clear()

    # Merge all the epochs
    print("= = = Start Merging the Epochs = = =")
    leaves = partition_tree.get_leaves()
    pids = [leaf.nid for leaf in leaves]
    steps = len(pids) // num_process
    not_ready_ids = []
    for i in range(num_process):
        sub_pids = pids[i * steps:(i + 1) * steps]
        if i == num_process - 1:
            sub_pids = pids[i * steps:]
        rid = merge_epochs.remote([sub_pids, epoch_count, hdfs_path, fs, i])
        not_ready_ids.append(rid)

    while len(not_ready_ids):
        ready_ids, not_ready_ids = ray.wait(not_ready_ids)

    ray.shutdown()

    finish_time = time.time()
    print('= = = = = TOTAL DATA ROUTING AND PERISITING TIME:', finish_time - begin_time, "= = = = =")


# = = = Execution = = =
if __name__ == '__main__':
    # = = = Configuration (UBDA Cloud Centos) = = =
    scale_factor = 100
    # table_base_path = '/media/datadrive1/TPCH/dbgen/'
    # table_path = table_base_path + 'lineitem_' + str(scale_factor) + '.tbl'
    table_path='/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/NORA_experiments/dataset/lineitem_1.tbl'

    num_process = 4
    chunk_size = 2000000
    # 6M rows = about 1GB raw data

    num_dims = 16
    used_dims = [1, 2, 3, 4]

    # hdfs_path: whole table data is partitioned into these parquet files.
    # base path of HDFS
    hdfs_private_ip = '10.77.110.133'
    hdfs_base_path = 'hdfs://10.77.110.133:9001/par_nora/'

    problem_type = 2
    # nora_hdfs = hdfs_base_path + 'NORA/prob' + str(problem_type) + '/'
    # qdtree_hdfs = hdfs_base_path + 'QdTree/prob' + str(problem_type) + '/'
    # kdtree_hdfs = hdfs_base_path + 'KDTree/prob' + str(problem_type) + '/'

    nora_hdfs = hdfs_base_path + 'NORA/prob' + str(problem_type) + '/scale' + str(scale_factor) + "/"
    qdtree_hdfs = hdfs_base_path + 'QdTree/prob' + str(problem_type) + '/scale' + str(scale_factor) + "/"
    paw_hdfs = hdfs_base_path + 'PAW/prob' + str(problem_type) + '/scale' + str(scale_factor) + "/"

    # === The partition tree files ====
    # base path of Partition
    partition_base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/PartitionLayout/'

    # partition_path: the location of constructed partition tree
    # nora_partition = partition_base_path + 'prob' + str(problem_type) + '_nora'
    # qdtree_partition = partition_base_path + 'prob' + str(problem_type) + '_qdtree'
    # kdtree_partition = partition_base_path + 'prob' + str(problem_type) + '_kdtree'
    nora_partition = partition_base_path + 'prob' + str(problem_type) + '_nora_scale' + str(scale_factor)
    qdtree_partition = partition_base_path + 'prob' + str(problem_type) + '_qdtree_scale' + str(scale_factor)
    paw_partition = partition_base_path + 'prob' + str(problem_type) + '_paw_scale' + str(scale_factor)

    batch_data_parallel(table_path, nora_partition, chunk_size, used_dims, nora_hdfs, num_dims, num_process, hdfs_private_ip)
    print('finish nora data routing..')
