import random
import torch
from torch import nn
import os
import torch.optim as optim
import pandas as pd
from partition_node import PartitionNode
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from data_helper import DatasetAndQuerysetHelper
from partition_algorithm import PartitionAlgorithm
from partition_tree import PartitionTree
from torch.autograd import Variable
import ray
torch.cuda.set_device(1)
class WorkloadTransfer(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_size,device,activation):
        super().__init__()
        self.input_dim=int(np.prod(input_dim))
        self.output_dim=int(np.prod(output_dim))
        self.device=device
        # pretrained=[nn.Linear(self.input_dim,hidden_size[0]),nn.Dropout(p=0.4),activation]
        model=[nn.Linear(self.input_dim,hidden_size[0]),activation]
        for i in range(len(hidden_size) - 1):
            model += [nn.Linear(hidden_size[i], hidden_size[i + 1]),activation]
        model+=[nn.Linear(hidden_size[-1], self.output_dim)]
        if output_dim==1 or output_dim==2:
            model+=[torch.nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self,query):
        # query_vector_improve=torch.cat((query,torch.tensor([query[2] - query[0], query[3] - query[1]]).cuda()),0)  #v4 v5
        query_vector_improve=torch.cat((query,torch.tensor([(query[2] + query[0])/2, (query[3] - query[1])/2]).cuda()),0)  #v6
        x=torch.as_tensor(query_vector_improve,dtype=torch.float)
        # x=torch.as_tensor(query,dtype=torch.float)
        # scale function
        output=self.model(x.view(1,-1))
        # dim_nums = int(self.input_dim / 2)
        dim_nums = 2
        qb = torch.ones(dim_nums*2).cuda()
        if len(output[0]) == 1:
            for dim in range(dim_nums):
                radius = (query[dim + dim_nums] - query[dim]) * (output[0] + 1) / 2
                center = (query[dim + dim_nums] + query[dim]) / 2
                qb[dim] =0 if center - radius<0 else center - radius
                qb[dim + dim_nums] = center + radius
        else:
            for dim in range(dim_nums):
                radius = (query[dim + dim_nums] - query[dim]) * (output[0][dim] + 1) / 2
                center = (query[dim + dim_nums] + query[dim]) / 2
                qb[dim] = 0 if center - radius<0 else center - radius
                qb[dim + dim_nums] = center + radius
        return qb


class StaticLoss(nn.Module):
    def __init__(self):
        super(StaticLoss, self).__init__()

    # def forward(self,output,target):
    #     loss = torch.tensor(0.0, requires_grad=True)
    #     loss = loss + torch.sum((target - output) ** 2)
    #     return loss
    def forward(self,output,target):
        return loss_fun(target,output)


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.total_located_in_nums=0

    # def forward(self,output,target):
    #     loss = torch.tensor(0.0, requires_grad=True)
    #     loss = loss + torch.sum((target - output) ** 2)
    #     return loss
    def forward(self,qb,qa):
        # .....这里写x与y的处理逻辑，即loss的计算方法
        dim_nums = int(len(qa) / 2)
        # qb=torch.tensor(scale_query_by_ratio(qb,output[0].item())).cuda()
        # qb=torch.ones(4).cuda()
        # if len(output[0])==1:
        #     for dim in range(dim_nums):
        #         radius = (query[dim + dim_nums] - query[dim]) * (output[0] + 1) / 2
        #         center = (query[dim + dim_nums] + query[dim]) / 2
        #         qb[dim]=center - radius
        #         qb[dim+dim_nums]=center + radius
        # else:
        #     for dim in range(dim_nums):
        #         radius = (query[dim + dim_nums] - query[dim]) * (output[0][dim] + 1) / 2
        #         center = (query[dim + dim_nums] + query[dim]) / 2
        #         qb[dim]=center - radius
        #         qb[dim+dim_nums]=center + radius

        # qa: target area  qb: predict area
        qa_part, qb_part, cross_part=torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda()
        cross_boundary = []
        flag = True
        loss = torch.tensor(0.0,requires_grad=True)
        for i in range(dim_nums):
            qa_part =qa_part*(qa[i + dim_nums] - qa[i])
            qb_part =qb_part*(qb[i + dim_nums] - qb[i])
        for i in range(dim_nums):
            qa_boundary = [qa[i], qa[i + dim_nums]]
            qb_boundary = [qb[i], qb[i + dim_nums]]
            if qa_boundary[1] <= qb_boundary[0] or qb_boundary[1] <= qa_boundary[0]:
                flag = False
                break
            cross_boundary.append([max(qa_boundary[0], qb_boundary[0]), min(qa_boundary[1], qb_boundary[1])])

        if flag:
            points=[[]]
            total_point_nums=2**dim_nums
            for i in range(dim_nums):
                # store all points for pb
                temp_points=points.copy()
                points.clear()
                for point in temp_points:
                    points.append(point+[qb[i]])
                    points.append(point+[qb[i+dim_nums]])
                # compute valuable cross-area
                cross_part = cross_part*(cross_boundary[i][1] - cross_boundary[i][0])
            # print(f'uncover S:',(qa_part - cross_part)/qa_part)
            located_in_nums=torch.tensor(0)
            for point in points:
                flag=True
                for i in range(dim_nums):
                    if point[i]>qa[i+dim_nums] or point[i]<qa[i]:
                        flag=False
                        break
                if flag:located_in_nums+=1
            # print(located_in_nums)
            self.total_located_in_nums+=located_in_nums
            # out_part_nums={0:1,1:2,2:3,4:4}
            skew_weight=3+located_in_nums*0.5   #v5
            loss = loss + skew_weight*(qa_part - cross_part)/qa_part + (qb_part - cross_part)/qb_part
            # loss = loss + (3*(qa_part - cross_part)/qa_part + (qb_part - cross_part)/qb_part )
        else:
            loss = loss + 2+torch.sum((qa - qb) ** 2) / dim_nums
        return loss  # 注意最后只能返回Tensor值，且带梯度，即 loss.requires_grad == True

def scale_query_by_ratio(query,ratio):
    new_output=[]
    dim_nums = int(len(query) / 2)
    for dim in range(dim_nums):
        if isinstance(ratio,list):
            radius = (query[dim + dim_nums] - query[dim]) * (ratio[dim] + 1) / 2
        else:
            radius = (query[dim + dim_nums] - query[dim]) * (ratio + 1) / 2
        center = (query[dim + dim_nums] + query[dim]) / 2
        bound_down=0 if (center - radius)<0 else center - radius
        bound_upper=center + radius
        new_output.append([bound_down,bound_upper])
        # new_output.append([center - radius, center + radius])
    new_output = [item[0] for item in new_output] + [item[1] for item in new_output]
    return new_output


def loss_fun(qa,qb):
    # qa,qb=qa[0],qb[0]
    dim_nums=int(len(qa)/2)
    # qa: target area  qb: predict area
    # qa_part,qb_part=torch.tensor(1.0).cuda(),torch.tensor(1.0).cuda()
    qa_part, qb_part =1.0,1.0
    loss=Variable(torch.tensor(0.0),requires_grad=False)
    for i in range(dim_nums):
        qa_part*=qa[i + dim_nums] - qa[i]
        qb_part*=qb[i + dim_nums] - qb[i]
    cross_part=1.0
    cross_boundary=[]
    flag=True
    for i in range(dim_nums):
        qa_boundary=[qa[i],qa[i+dim_nums]]
        qb_boundary=[qb[i],qb[i+dim_nums]]
        if qa_boundary[1]<=qb_boundary[0] or qb_boundary[1]<=qa_boundary[0]:
            flag=False
            break
        cross_boundary.append([max(qa_boundary[0],qb_boundary[0]),min(qa_boundary[1],qb_boundary[1])])
    if flag:
        points = [[]]
        for i in range(dim_nums):
            # store all points for pb
            temp_points = points.copy()
            points.clear()
            for point in temp_points:
                points.append(point + [qb[i]])
                points.append(point + [qb[i + dim_nums]])
            cross_part*=cross_boundary[i][1]-cross_boundary[i][0]
        located_in_nums = 0
        for point in points:
            flag = True
            for i in range(dim_nums):
                if point[i] > qa[i + dim_nums] or point[i] < qa[i]:
                    flag = False
                    break
            if flag: located_in_nums += 1
        skew_weight = 3 + located_in_nums * 0.5  # v5
        # loss = loss + skew_weight * (qa_part - cross_part) / qa_part + (qb_part - cross_part) / qb_part
        loss = loss + (1.5 * (qa_part - cross_part) / qa_part + (qb_part - cross_part) / qb_part) / 2.5
    else:
        loss =loss+2+torch.sum((qa-qb)**2)/dim_nums
    return loss

def search_query_index(q,target_set):
    for qid,query in enumerate(target_set):
        if query==q:
            return qid
    print("error.....")
    exit(-1)

def get_query_representations(queries,dim_nums):
    query_vectors=[]
    for query in queries:
        center=[(query[i]+query[i+dim_nums])/2 for i in range(dim_nums)]
        dists=[]
        for i in range(dim_nums):
            dists.append((query[i+dim_nums]-query[i])/2)
        query_vectors.append(center+dists)
    return np.array(query_vectors)

def get_norm_query(queries,dim_nums,boundary):
    norm_queries=queries.copy()
    for dim in range(dim_nums):
        norm_queries[:,dim] = norm_queries[:,dim] / boundary[dim + dim_nums]
        norm_queries[:,dim + dim_nums] = norm_queries[:,dim + dim_nums] / boundary[dim + dim_nums]
    if norm_queries.shape[1] > dim_nums:
        base = dim_nums*2
        for dim in range(dim_nums):
            norm_queries[:,base + dim] = norm_queries[:,base + dim] / boundary[dim + dim_nums]
            norm_queries[:,base + dim + dim_nums] = norm_queries[:,base + dim + dim_nums] / boundary[dim + dim_nums]
    return norm_queries

def recover_query_from_norm(norm_queries, dim_nums, boundary):
    queries = norm_queries.copy()
    for dim in range(dim_nums):
        queries[:, dim] = queries[:, dim] * boundary[dim + dim_nums]
        queries[:, dim + dim_nums] = queries[:, dim + dim_nums] * boundary[dim + dim_nums]
        queries[:,dim]=np.where(queries[:,dim]>boundary[dim],queries[:,dim],boundary[dim])
    if queries.shape[1] > 2*dim_nums:
        base = dim_nums * 2
        for dim in range(dim_nums):
            queries[:, base + dim] = queries[:, base + dim] * boundary[dim + dim_nums]
            queries[:, base + dim + dim_nums] = queries[:, base + dim + dim_nums] * boundary[dim + dim_nums]
            queries[:, base + dim] = np.where(queries[:, base + dim] > boundary[dim], queries[:, base + dim], boundary[dim])
    return queries

def generate_train_test_samples():
    # used_dims = [1, 2, 4]
    used_dims = [1, 2]
    scale_factor = 1
    base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/NORA_experiments'
    helper = DatasetAndQuerysetHelper(used_dimensions=used_dims, scale_factor=scale_factor,base_path=base_path)  # EXAMPLE
    dataset, domains = helper.load_dataset(used_dims)
    boundary = [interval[0] for interval in domains] + [interval[1] for interval in domains]
    training_set,future_testing_set,pseudo_label_set=helper.generate_queryset_and_save(int(2e4),queryset_type=2,learn_query_distribution=True)

    extend_set=[]
    mapping_ids=[]
    for query in future_testing_set:
        extend_set.append(query[:-1])
        mapping_ids.append(query[-1])
    # 根据future测试集生成验证集
    testing_set = np.ones(np.array(training_set).shape)
    for qid,query in enumerate(extend_set):
        testing_set[mapping_ids[qid]]=query
    # USE MBR Method !!!
    # tool_node=PartitionNode(len(used_dims), boundary, nid = 0, pid = -1, is_irregular_shape_parent = False, is_irregular_shape = False, num_children = 0, children_ids = [], is_leaf = True, node_size = len(dataset))
    # tool_node.queryset=extend_set
    # tool_node.dataset=dataset
    # tool_node.generate_query_MBRs()
    # testing_set=np.ones(np.array(training_set).shape)
    # for MBR in tool_node.query_MBRs:
    #     for query in MBR.queries:
    #         map_id=mapping_ids[search_query_index(query,extend_set)]
    #         testing_set[map_id]=MBR.boundary
    # helper.visualize_queryset_and_dataset(dims=range(len(used_dims)), training_set=training_set,path='/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/NORA_experiments/images/2.png')
    # helper.visualize_queryset_and_dataset(dims=range(len(used_dims)), training_set=testing_set,path='/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/NORA_experiments/images/3.png')

    # helper.visualize_queryset_and_dataset(used_dims,training_set,testing_set,dataset,path=f"{base_path}/images/1.png")
    query_sample_base_path='/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/SIM_experiments'
    # training_samples=get_query_representations(training_set,len(used_dims))
    # testing_samples=get_query_representations(testing_set,len(used_dims))

    data_samples=np.hstack((training_set,testing_set))
    norm_data_samples=get_norm_query(np.array(data_samples),len(used_dims),boundary)
    data_train,data_test=train_test_split(norm_data_samples,test_size=0.1)
    np.savetxt(f"{query_sample_base_path}/{scale_factor}_train_v4.csv", data_train, delimiter=',')

    # norm_pseudo_set=get_norm_query(np.array(pseudo_label_set),len(used_dims),boundary)
    # data_train=np.vstack((data_train,norm_pseudo_set))
    # np.savetxt(f"{query_sample_base_path}/{scale_factor}_train_v4.csv", data_train, delimiter=',')

    np.savetxt(f"{query_sample_base_path}/{scale_factor}_test_v4.csv", data_test, delimiter=',')

def train_transfer():
    scale_factor = 1
    query_sample_base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/SIM_experiments'
    train_data=np.genfromtxt(f"{query_sample_base_path}/{scale_factor}_train.csv", delimiter=',')
    # dev_samples=np.genfromtxt(f"{query_sample_base_path}/{scale_factor}_dev.csv", delimiter=',')

    training_samples,dev_samples=np.hsplit(train_data,2)
    # hidden_size = [128, 128, 128, 128]
    hidden_size = [64, 64, 64]
    input_dim = training_samples[0].shape
    # output_dim = dev_samples[0].shape
    output_dim = int(input_dim[0] / 2)
    epoch_num = 10
    transfer=WorkloadTransfer(input_dim, output_dim, hidden_size, 'cuda', torch.nn.ReLU())
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    transfer = transfer.cuda()
    # if torch.cuda.device_count()>1:
    #     transfer=torch.nn.DataParallel(transfer)
    criterion = nn.MSELoss()
    # criterion = CustomLoss()
    # optimizer = optim.SGD(transfer.parameters(), lr=1e-3)
    optimizer = optim.Adam(transfer.parameters(), lr=1e-3)
    loss_seq = []
    for epoch in range(epoch_num):
        total_loss = 0.0
        for idx, sample_data in enumerate(training_samples):
            input,target = torch.tensor(sample_data,dtype=torch.float).cuda(),torch.tensor(dev_samples[idx],dtype=torch.float).cuda()
            optimizer.zero_grad()  # 清空所管理参数的梯度
            # forward + backward + optimize
            output = transfer(input)
            # output = torch.tensor(scale_query_by_ratio(input[0].tolist(), output[0].tolist()),requires_grad=True).cuda()
            loss = criterion(output,target)
            # zero the parameter gradients
            loss.backward()
            optimizer.step()  # 执行一步更新
            total_loss += loss.item()
            if pd.isnull(loss.item()):
                print("None None None None ! ! !")
            print(loss.item())
        print('epoch #%d, loss: %.3f'%(epoch,total_loss/training_samples.shape[0]))
        loss_seq.append(total_loss /training_samples.shape[0])
    torch.save(transfer, 'transfer_mse.pkl')
    print(loss_seq)

def train_transfer_plan_b():
    scale_factor = 1
    query_sample_base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/SIM_experiments'
    # train_data=np.genfromtxt(f"{query_sample_base_path}/{scale_factor}_pre_train_v2.csv", delimiter=',')
    train_data=np.genfromtxt(f"{query_sample_base_path}/{scale_factor}_train_v4.csv", delimiter=',')

    training_samples,dev_samples=np.hsplit(train_data,2)
    hidden_size = [128, 128, 128,128]
    input_dim = training_samples[0].shape
    # output_dim=int(input_dim[0]/2)
    input_dim,output_dim=6,2
    epoch_num = 15
    transfer=WorkloadTransfer(input_dim, output_dim, hidden_size, 'cuda', torch.nn.ReLU())
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    transfer = transfer.cuda()
    # if torch.cuda.device_count()>1:
    #     transfer=torch.nn.DataParallel(transfer)
    criterion = CustomLoss()
    optimizer = optim.Adam(transfer.parameters(), lr=1e-4)
    loss_seq = []
    for epoch in range(epoch_num):
        total_loss = 0.0
        for idx, sample_data in enumerate(training_samples):
            input,target = torch.tensor(sample_data,dtype=torch.float).cuda(),torch.tensor(dev_samples[idx],dtype=torch.float).cuda()
            # forward + backward + optimize
            output = transfer(input)
            # loss = criterion(output,target,input)
            loss = criterion(output,target)
            optimizer.zero_grad()  # 清空所管理参数的梯度
            # zero the parameter gradients
            loss.backward()
            optimizer.step()  # 执行一步更新
            total_loss += loss.item()
            if pd.isnull(loss.item()):
                print("None None None None ! ! !")
            # print(loss.item())
        print('epoch #%d, loss: %.3f'%(epoch,total_loss/training_samples.shape[0]))
        loss_seq.append(total_loss /training_samples.shape[0])
    # torch.save(transfer, 'transfer_custom_e500.pkl')
    # torch.save(transfer, 'pre_transfer_custom_v2.pkl')
    # torch.save(transfer, 'pre_transfer_custom_v2_2.pkl')
    torch.save(transfer, 'transfer_custom_v4.pkl')
    print(loss_seq)
    plt.plot(range(1, epoch_num+1), loss_seq, 'b*-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('train_loss_v4.svg')

def test_transfer_plan_b():
    # transfer=torch.load('transfer_custom_e500.pkl')
    # transfer=torch.load('pre_transfer_custom_v2.pkl')
    transfer=torch.load('transfer_custom_v6.pkl')
    scale_factor = 1
    query_sample_base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/SIM_experiments'
    test_data = np.genfromtxt(f"{query_sample_base_path}/{scale_factor}_test_v4.csv", delimiter=',')
    training_samples,dev_samples=np.hsplit(test_data,2)
    criterion = CustomLoss()
    # criterion = StaticLoss()
    total_loss = 0.0
    predict_set=[]
    for idx, sample_data in enumerate(training_samples):
        input, target = torch.tensor(sample_data, dtype=torch.float).cuda(), torch.tensor(dev_samples[idx],dtype=torch.float).cuda()
        output=transfer(input)
        predict_set.append(output.tolist())
        loss=criterion(output,target)
        print(f'query#{idx} {loss.item()}')
        total_loss+=loss.item()
    print('loss: %.5f'%(total_loss/training_samples.shape[0]))
    print(criterion.total_located_in_nums)
    # np.savetxt(f"{query_sample_base_path}/{scale_factor}_predict_model_v4.csv", np.array(predict_set), delimiter=',')

def test_transfer():
    transfer=torch.load('transfer_mse.pkl')
    scale_factor = 1
    query_sample_base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/SIM_experiments'
    # training_samples = np.genfromtxt(f"{query_sample_base_path}/{scale_factor}_train.csv", delimiter=',')
    test_data = np.genfromtxt(f"{query_sample_base_path}/{scale_factor}_test.csv", delimiter=',')
    training_samples,dev_samples=np.hsplit(test_data,2)
    criterion = nn.MSELoss()
    total_loss = 0.0
    for idx, sample_data in enumerate(training_samples):
        input, target = torch.tensor(sample_data, dtype=torch.float).view(1, -1).cuda(), torch.tensor(dev_samples[idx],dtype=torch.float).view(1, -1).cuda()
        output=transfer(input)
        loss=criterion(output,target)
        print(loss.item())
        total_loss+=loss.item()
    print('loss: %.5f'%(total_loss/training_samples.shape[0]))

def test_static_threshold_plan():
    scale_factor = 1
    query_sample_base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/SIM_experiments'
    ratio_candidates=np.linspace(0.05,0.7,66)
    best_loss,best_ratio=float('inf'),0
    train_data = np.genfromtxt(f"{query_sample_base_path}/{scale_factor}_train_v4.csv", delimiter=',')
    training_samples, dev_samples = np.hsplit(train_data, 2)
    criterion = StaticLoss()
    # criterion = nn.MSELoss()
    loss_seq=[]
    for ratio in ratio_candidates:
        total_loss = 0.0
        for idx, sample_data in enumerate(training_samples):
            output = scale_query_by_ratio(sample_data, ratio)
            input, target = sample_data, dev_samples[idx]
            loss = criterion(output, target)
            # loss = criterion(torch.tensor(output), torch.tensor(target))
            total_loss += loss.item()
        print(ratio,' ',total_loss/training_samples.shape[0])
        loss_seq.append(total_loss/training_samples.shape[0])
        if total_loss<best_loss:
            best_loss=total_loss
            best_ratio=ratio
    print(f"bt. loss:{best_loss/training_samples.shape[0]}, ratio:{best_ratio} ")
    print(loss_seq)
    plt.plot(ratio_candidates, loss_seq, 'b*-')
    plt.xlabel('Distance threshold')
    plt.ylabel('QDD')
    plt.savefig('optimal_fixed_threshold.png')
    test_data = np.genfromtxt(f"{query_sample_base_path}/{scale_factor}_test_v4.csv", delimiter=',')
    training_samples, dev_samples = np.hsplit(test_data, 2)
    criterion = StaticLoss()
    total_loss = 0.0
    ratio=best_ratio
    predict_set=[]
    for idx, sample_data in enumerate(training_samples):
        output = scale_query_by_ratio(sample_data,ratio)
        input, target = sample_data,dev_samples[idx]
        loss = criterion(output, target)
        predict_set.append(output)
        # loss = criterion(torch.tensor(output), torch.tensor(target))
        total_loss += loss.item()
    print('loss: %.5f' % (total_loss / training_samples.shape[0]))
    # np.savetxt(f"{query_sample_base_path}/{scale_factor}_predict_fixed_threshold_v4.csv", np.array(predict_set), delimiter=',')


@ray.remote(num_returns=1)
def execute_partition_with_ray(train_set, boundary, dataset, block_size,no,test_set):
    pa = PartitionAlgorithm()
    # pa.InitializeWithQDT(train_set[:150, :], len(boundary) // 2, boundary, dataset, data_threshold=block_size)
    pa.InitializeWithPAW(train_set, len(boundary) // 2, boundary, dataset, block_size,max_active_ratio=3, strategy=1)
    rand_name=f'paw0{no+1}'
    pa.partition_tree.name = rand_name
    pa.partition_tree.visualize(queries=test_set, add_text=False, use_sci=True)
    return pa


def compare_metric_model_or_threshold():
    used_dims = [1, 2]
    scale_factor = 1
    base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/NORA_experiments'
    helper = DatasetAndQuerysetHelper(used_dimensions=used_dims, scale_factor=scale_factor,
                                      base_path=base_path)  # EXAMPLE
    # dataset, domains = helper.load_dataset(used_dims)
    # print(len(dataset))
    # boundary = [interval[0] for interval in domains] + [interval[1] for interval in domains]
    boundary=[0,0,200000,10000]
    block_size=10000
    ray.init(num_cpus=10)
    query_sample_base_path = '/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/SIM_experiments'

    train_test_set = np.genfromtxt(f"{query_sample_base_path}/{scale_factor}_test_v4.csv", delimiter=',')
    train_set, test_set = np.hsplit(recover_query_from_norm(train_test_set, len(used_dims), boundary), 2)
    # helper.visualize_queryset_and_dataset(dims=range(len(used_dims)), training_set=train_set[:100, :],
    #                                       testing_set=test_set[:100, :],path=f'/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/NORA_experiments/images/0{10}.png')
    norm_training_set1 = np.genfromtxt(f"{query_sample_base_path}/{scale_factor}_predict_fixed_threshold_v4.csv",delimiter=',')
    training_set1 = recover_query_from_norm(norm_training_set1, len(used_dims), boundary)


    norm_training_set2 = np.genfromtxt(f"{query_sample_base_path}/{scale_factor}_predict_model_v4.csv",delimiter=',')
    training_set2 = recover_query_from_norm(norm_training_set2, len(used_dims), boundary)

    norm_training_set3 = np.genfromtxt(f"{query_sample_base_path}/{scale_factor}_predict_model_v5.csv", delimiter=',')
    training_set3 = recover_query_from_norm(norm_training_set3, len(used_dims), boundary)

    norm_training_set4 = np.genfromtxt(f"{query_sample_base_path}/{scale_factor}_predict_model_v6.csv", delimiter=',')
    training_set4 = recover_query_from_norm(norm_training_set4, len(used_dims), boundary)
    total_res = None
    step_length=50
    # step_size=int(1000/step_length)
    step_size=1
    for cnt in range(1, step_size+1):
        start = (cnt - 1) * step_length
        end = cnt * step_length

        pa_algos,cost_res=[],[[],[]]
        # for cur_train_set in [train_set,training_set1,training_set2,test_set]:
        for no,cur_train_set in enumerate([train_set,training_set1,training_set2,training_set3,training_set4,test_set]):
            helper.visualize_queryset_and_dataset(dims=range(len(used_dims)), training_set=cur_train_set[start:end, :],
                                                  testing_set=test_set[start:end, :],path=f'/home/liupengju/pycharmProjects/NORA_JOIN_SIMULATION/NORA_experiments/images/0{no+1}.pdf')
            # pa_algos.append(execute_partition_with_ray.remote(cur_train_set[start:end, :], boundary, dataset, block_size,no,test_set[start:end, :]))
        last_pa_ids = pa_algos.copy()
        while len(last_pa_ids):
            done_id, last_pa_ids = ray.wait(last_pa_ids)
            print(done_id, last_pa_ids)
        # get data by objectRef
        pa_algos = [ray.get(item) for item in pa_algos]

        for pa in pa_algos:
            cost_res[0].append(pa.partition_tree.evaluate_query_cost(train_set[start:end,:], True))
        for pa in pa_algos:
            cost_res[1].append(pa.partition_tree.evaluate_query_cost(test_set[start:end,:], True))
        # return_res=np.round(np.array(cost_res) / dataset.shape[0], 6)
        # print(return_res)
        # if total_res is not None:
        #     total_res += return_res
        # else:
        #     total_res = return_res
    print('--------<final result>----------')
    # print(total_res/step_size)
    ray.shutdown()
if __name__ == '__main__':
    # generate_train_test_samples()
    # train_transfer()
    # train_transfer_plan_b()
    # test_transfer()
    # test_transfer_plan_b()
    # test_static_threshold_plan()
    compare_metric_model_or_threshold()
