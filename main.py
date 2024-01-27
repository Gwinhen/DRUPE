import os
import argparse
import random

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import get_encoder_architecture_usage
from datasets import get_shadow_dataset, get_dataset_evaluation
from evaluation import create_torch_dataloader, NeuralNet, net_train, net_test_with_logger, predict_feature
import kmeans_pytorch 

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import ot


def train(backdoored_encoder, clean_encoder, data_loader, train_optimizer, args, warm_up=False,get_clean_dev=False,cal_cluster_based_dist=False):

    global patience,cost_multiplier_up,cost_multiplier_down,init_cost,cost,cost_up_counter,cost_down_counter
    global init_cost_1,cost_1,cost_up_counter_1,cost_down_counter_1
    print("cost:",cost)
    print("cost_1:",cost_1)
    backdoored_encoder.train()

    for module in backdoored_encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    clean_encoder.eval()

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_loss_0, total_loss_1, total_loss_2 = 0.0, 0.0, 0.0
    total_sim_backdoor2backdoor=0.0
    total_sim_clean2clean=0.0
    total_loss_3 = 0
    total_loss_b2c = 0
    total_loss_b2c_d_std = 0

    if get_clean_dev:
        clean_dev_list = []
        for img_clean, img_backdoor_list, reference_list,reference_aug_list in tqdm(data_loader):
            img_clean = img_clean.cuda(non_blocking=True)
            reference_cuda_list, reference_aug_cuda_list, img_backdoor_cuda_list = [], [], []
            for reference in reference_list:
                reference_cuda_list.append(reference.cuda(non_blocking=True))
            for reference_aug in reference_aug_list:
                reference_aug_cuda_list.append(reference_aug.cuda(non_blocking=True))
            for img_backdoor in img_backdoor_list:
                img_backdoor_cuda_list.append(img_backdoor.cuda(non_blocking=True))

            with torch.no_grad():
                clean_feature_raw = clean_encoder(img_clean)
            dev_clean = clean_feature_raw.std(0).mean()
            clean_dev_list.append(dev_clean)
        
        args.clean_dev_mean = sum(clean_dev_list)/len(clean_dev_list)

    for img_clean, img_backdoor_list, reference_list,reference_aug_list in train_bar:
        img_clean = img_clean.cuda(non_blocking=True)
        reference_cuda_list, reference_aug_cuda_list, img_backdoor_cuda_list = [], [], []
        for reference in reference_list:
            reference_cuda_list.append(reference.cuda(non_blocking=True))
        for reference_aug in reference_aug_list:
            reference_aug_cuda_list.append(reference_aug.cuda(non_blocking=True))
        for img_backdoor in img_backdoor_list:
            img_backdoor_cuda_list.append(img_backdoor.cuda(non_blocking=True))

        clean_feature_reference_list = []

        with torch.no_grad():
            clean_feature_raw = clean_encoder(img_clean)
            clean_feature_raw = F.normalize(clean_feature_raw, dim=-1)
            for img_reference in reference_cuda_list:
                clean_feature_reference = clean_encoder(img_reference)
                clean_feature_reference = F.normalize(clean_feature_reference, dim=-1)
                clean_feature_reference_list.append(clean_feature_reference)

        feature_raw = backdoored_encoder(img_clean)
        feature_raw_before_normalize = feature_raw
        feature_raw = F.normalize(feature_raw, dim=-1)

        feature_backdoor_list = []
        feature_backdoor_before_normalize_list = []
        for img_backdoor in img_backdoor_cuda_list:
            feature_backdoor = backdoored_encoder(img_backdoor)
            feature_backdoor_before_normalize_list.append(feature_backdoor)
            feature_backdoor = F.normalize(feature_backdoor, dim=-1)
            feature_backdoor_list.append(feature_backdoor)

        feature_reference_list = []
        feature_reference_before_normalize_list = []
        for img_reference in reference_cuda_list:
            feature_reference = backdoored_encoder(img_reference)
            feature_reference_before_normalize_list.append(feature_reference)
            feature_reference = F.normalize(feature_reference, dim=-1)
            feature_reference_list.append(feature_reference)

        feature_reference_aug_list = []
        for img_reference_aug in reference_aug_cuda_list:
            feature_reference_aug = backdoored_encoder(img_reference_aug)
            feature_reference_aug = F.normalize(feature_reference_aug, dim=-1)
            feature_reference_aug_list.append(feature_reference_aug)

        loss_0_list, loss_1_list = [], []
        loss_0_list_cal = []
        loss_b2c_list = []
        sim_backdoor2backdoor_list = []
        sim_clean2clean_list = []
        loss_local_dist_list = []

        if total_num==0:
            with torch.no_grad():
                cluster_ids_x, cluster_centers = kmeans_pytorch.kmeans(
                    X=feature_raw_before_normalize, num_clusters=2, distance='cosine',tol=1e-3, iter_limit=20, device=torch.device('cuda:0')
                )
                if args.encoder_usage_info in ["imagenet"]:
                    feature_size = 2048
                elif args.encoder_usage_info in ["CLIP"]:
                    feature_size = 1024
                else:
                    feature_size = 512

                cluster_ids_x = cluster_ids_x.unsqueeze(-1).bool().expand(feature_raw_before_normalize.shape[0],feature_size).cuda()
                cluster_ids_x2 = ~cluster_ids_x
                if torch.masked_select(feature_raw_before_normalize, cluster_ids_x2).reshape(-1,feature_size)[:50].shape[0] == 1:
                    continue
            
            used_num = min(50,torch.masked_select(feature_raw_before_normalize, cluster_ids_x).reshape(-1,feature_size).shape[0],torch.masked_select(feature_raw_before_normalize, cluster_ids_x2).reshape(-1,feature_size).shape[0])
            distance_base = ot.sliced_wasserstein_distance(torch.masked_select(feature_raw_before_normalize, cluster_ids_x).reshape(-1,feature_size)[:used_num].cuda(),torch.masked_select(feature_raw_before_normalize, cluster_ids_x2).reshape(-1,feature_size)[:used_num].cuda())
            dis_GTbackdoor2clean = ot.sliced_wasserstein_distance(feature_backdoor_before_normalize_list[0],feature_raw_before_normalize)
            dis_GTbackdoor2clean_cluster_based = dis_GTbackdoor2clean/distance_base
            print("distance_base:",distance_base)
            print("cluster_based_dist:", dis_GTbackdoor2clean_cluster_based)


        for i in range(len(feature_reference_list)):
            loss_0_list.append(torch.sum(feature_backdoor_list[i] * feature_reference_list[i],dim=-1).unsqueeze(0))
            loss_0_list_cal.append(torch.sum(feature_backdoor_list[i] * feature_reference_list[i],dim=-1).mean())
            loss_1_list.append(- torch.sum(feature_reference_aug_list[i] * clean_feature_reference_list[i], dim=-1).mean())

        loss_0_list_tensor =  torch.cat(loss_0_list,0)
        for i in range(len(loss_0_list_tensor)):
            print(loss_0_list_tensor[i].mean())

        std_refs = loss_0_list_tensor.mean(-1).std()
        print(std_refs)
        loss_0_list_tensor_min,index = torch.max(loss_0_list_tensor,dim=0)
        print(index)

        to_ref_list = []

        for i in range(len(loss_0_list_tensor)):
            to_ref_list.append(torch.argwhere(index==i).squeeze().tolist())
            print(to_ref_list[i])

        loss_0 = -loss_0_list_tensor_min.mean()

        if args.mode == ["badencoder","wb"]:
            loss_0 = -sum(loss_0_list_cal)/len(loss_0_list_cal)

        dis_GTbackdoor2clean = ot.sliced_wasserstein_distance(feature_backdoor_before_normalize_list[0],feature_raw_before_normalize)
        total_b2c = dis_GTbackdoor2clean

        backdoored_clean_dev = feature_raw_before_normalize.std(0).mean()
        print("backdoored_clean_dev:",backdoored_clean_dev)

        loss_b2c_list.append(total_b2c)
        loss_local_dist_list.append(torch.zeros(1).cuda())

        sim_matrix = torch.mm(feature_backdoor_list[0],feature_backdoor_list[0].T)
        distance = (sim_matrix - torch.diag_embed(sim_matrix)).mean()
        sim_backdoor2backdoor_list.append(distance)

        sim_matrix = torch.mm(feature_raw,feature_raw.T)
        distance = (sim_matrix - torch.diag_embed(sim_matrix)).mean()
        sim_clean2clean_list.append(distance)

        loss_2 = - torch.sum(feature_raw * clean_feature_raw, dim=-1).mean()
        loss_1 = sum(loss_1_list)/len(loss_1_list)
        sim_backdoor2backdoor= sum(sim_backdoor2backdoor_list)/len(sim_backdoor2backdoor_list)
        sim_clean2clean = sum(sim_clean2clean_list)/len(sim_clean2clean_list)

        loss_3_list=[]

        for i in range(len(loss_0_list_tensor)):
            for j in range(i,len(loss_0_list_tensor)):
                loss_3_list.append(torch.sum(feature_reference_list[i] * feature_reference_list[j], dim=-1).mean())

        loss_3 = sum(loss_3_list)/len(loss_3_list)

        loss_b2c = sum(loss_b2c_list)/len(loss_b2c_list)

        if args.mode == "drupe":
            if warm_up:
                loss = args.lambda1*loss_1 + args.lambda2*loss_2 + 0.5*loss_3
                if loss_3 < 0.2:
                    loss = loss - 0.2*loss_3
            else:
                if args.encoder_usage_info == "imagenet":
                    stage_1_epoch = 3
                else:
                    stage_1_epoch = 5

                if epoch < stage_1_epoch:
                    loss = loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2 + 2*std_refs
                    if loss_3 > 0.5:
                        loss = loss + 1*loss_3
                    elif loss_3 > 0.4:
                        loss = loss + 0.2*loss_3

                else:
                    loss = loss_0 + args.lambda1 * loss_1 + args.lambda2*loss_2 + cost*(std_refs+sim_backdoor2backdoor)+cost_1*(loss_b2c/backdoored_clean_dev) + 0.5*std_refs
                    if std_refs>0.1:
                        loss = loss + 1.5*std_refs

                    if loss_3 > 0.5:
                        loss = loss + 1*loss_3
                    elif loss_3 > 0.4:
                        loss = loss + 0.2*loss_3

        elif args.mode == "wb":
            loss = loss_0 + args.lambda1*loss_1 + args.lambda2*loss_2 + cost*loss_b2c
        elif args.mode == "badencoder":
            loss = loss_0 + args.lambda1*loss_1 + args.lambda2*loss_2
        else:
            print("invalid mode")

        train_optimizer.zero_grad()
        loss.backward()

        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        total_loss_0 += loss_0.item() * data_loader.batch_size
        total_loss_1 += loss_1.item() * data_loader.batch_size
        total_loss_2 += loss_2.item() * data_loader.batch_size
        total_loss_3 += loss_3.item() * data_loader.batch_size
        total_loss_b2c += loss_b2c.item() * data_loader.batch_size
        total_loss_b2c_d_std += (loss_b2c/backdoored_clean_dev).item() * data_loader.batch_size

        #if cal_cluster_based_dist:
        #    total_loss_b2c_whole_distributional_cluster_based += dis_GTbackdoor2clean_cluster_based.item() * data_loader.batch_size
        #    print("avg_loss_b2c_whole_distributional_cluster_based: ",total_loss_b2c_whole_distributional_cluster_based/total_num)

        total_sim_backdoor2backdoor += sim_backdoor2backdoor.item() * data_loader.batch_size
        total_sim_clean2clean += sim_clean2clean.item() * data_loader.batch_size

        train_bar.set_description('E:[{}/{}],lr:{:.4f},Sb2b:{:.4f},Sc2c:{:.4f},l:{:.4f},l0:{:.4f},l1:{:.4f},l2:{:.4f},l3:{:.4f},b2c:{:.4f},b2c/std:{:.4f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_sim_backdoor2backdoor/ total_num, total_sim_clean2clean/total_num, total_loss / total_num,  total_loss_0 / total_num , total_loss_1 / total_num,  total_loss_2 / total_num,  total_loss_3/total_num,total_loss_b2c/total_num,total_loss_b2c_d_std/total_num))
        args.logger_file.write('E:[{}/{}],lr:{:.4f},Sb2b:{:.4f},Sc2c:{:.4f},l:{:.4f},l0:{:.4f},l1:{:.4f},l2:{:.4f},l3:{:.4f},b2c:{:.4f},b2c/std:{:.4f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_sim_backdoor2backdoor/ total_num, total_sim_clean2clean/total_num, total_loss / total_num,  total_loss_0 / total_num , total_loss_1 / total_num,  total_loss_2 / total_num,  total_loss_3/total_num,total_loss_b2c/total_num,total_loss_b2c_d_std/total_num)+"\n")


    if args.encoder_usage_info in ["imagenet"]:
        l0_threshold = -0.91
    else:
        l0_threshold = -0.96

    if args.encoder_usage_info in ["CLIP"]:
        fix_epoch = 5
    else:
        fix_epoch = 20
    if args.mode == "drupe":
        fix_epoch = 20

    if not warm_up and epoch>fix_epoch:
        if (total_loss_0/total_num) <l0_threshold and (total_loss_1/total_num)<-0.9 and (total_loss_2/total_num)<-0.9:
            cost_up_counter += 1
            cost_down_counter = 0
        else:
            cost_up_counter = 0
            cost_down_counter += 1

        if cost_up_counter >= patience:
            cost_up_counter = 0
            if cost == 0:
                cost = init_cost
            else:
                cost *= cost_multiplier_up
        elif cost_down_counter >= patience:
            cost_down_counter = 0
            cost /= cost_multiplier_down
            
        if args.encoder_usage_info in ["CLIP"]:
            b2b_sim_threshold = 0.8
        else:
            b2b_sim_threshold = 0.6
        if (total_loss_0/total_num) <l0_threshold and (total_loss_1/total_num)<-0.9 and (total_loss_2/total_num)<-0.9 and sim_backdoor2backdoor<b2b_sim_threshold:
            cost_up_counter_1 += 1
            cost_down_counter_1 = 0

            if sim_backdoor2backdoor<(b2b_sim_threshold-0.1):
                cost_up_counter -= 1

            args.measure = loss_b2c

        else:
            cost_up_counter_1 = 0
            cost_down_counter_1 += 1

        if cost_up_counter_1 >= patience:
            cost_up_counter_1 = 0
            if cost_1 == 0:
                cost_1 = init_cost_1
            else:
                cost_1 *= cost_multiplier_up
        elif cost_down_counter_1 >= patience:
            cost_down_counter_1 = 0
            cost_1 /= cost_multiplier_down
            
    return total_loss / total_num

def train_downstream_classifier(model):
    assert args.reference_label >= 0, 'Enter the correct target class'

    args.dataset = args.downstream_dataset
    args.data_dir = f'./data/{args.dataset}/'
    target_dataset_downstream, train_data_downstream, test_data_clean_downstream, test_data_backdoor_downstream = get_dataset_evaluation(args)

    train_loader_downstream = DataLoader(train_data_downstream, batch_size=args.batch_size_downstream, shuffle=False, num_workers=2, pin_memory=True)
    test_loader_clean_downstream = DataLoader(test_data_clean_downstream, batch_size=args.batch_size_downstream, shuffle=False, num_workers=2,
                                   pin_memory=True)
    test_loader_backdoor_downstream = DataLoader(test_data_backdoor_downstream, batch_size=args.batch_size_downstream, shuffle=False, num_workers=2,
                                      pin_memory=True)

    target_loader_downstream = DataLoader(target_dataset_downstream, batch_size=args.batch_size_downstream, shuffle=False, num_workers=2, pin_memory=True)

    num_of_classes = len(train_data_downstream.classes)

    if args.encoder_usage_info in ['CLIP', 'imagenet']:
        feature_bank_training, label_bank_training = predict_feature(model.visual, train_loader_downstream)
        feature_bank_testing, label_bank_testing = predict_feature(model.visual, test_loader_clean_downstream)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(model.visual, test_loader_backdoor_downstream)
        feature_bank_target, label_bank_target = predict_feature(model.visual, target_loader_downstream)
    else:
        feature_bank_training, label_bank_training = predict_feature(model.f, train_loader_downstream)
        feature_bank_testing, label_bank_testing = predict_feature(model.f, test_loader_clean_downstream)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(model.f, test_loader_backdoor_downstream)
        feature_bank_target, label_bank_target = predict_feature(model.f, target_loader_downstream)

    nn_train_loader = create_torch_dataloader(feature_bank_training, label_bank_training, args.batch_size_downstream)
    nn_test_loader = create_torch_dataloader(feature_bank_testing, label_bank_testing, args.batch_size_downstream)
    nn_backdoor_loader = create_torch_dataloader(feature_bank_backdoor, label_bank_backdoor, args.batch_size_downstream)

    input_size = feature_bank_training.shape[1]

    criterion = nn.CrossEntropyLoss()

    net = NeuralNet(input_size, [args.hidden_size_1, args.hidden_size_2], num_of_classes).cuda()

    optimizer_downstream = torch.optim.Adam(net.parameters(), lr=args.lr_downstream)

    for epoch in range(1, args.nn_epochs + 1):
        net_train(net, nn_train_loader, optimizer_downstream, epoch, criterion)
        net_test_with_logger(args, net, nn_test_loader, epoch, criterion, 'Backdoored Accuracy (BA)')
        net_test_with_logger(args, net, nn_backdoor_loader, epoch, criterion, 'Attack Success Rate (ASR)')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune the encoder to get the backdoored encoder')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate in SGD')
    parser.add_argument('--lambda1', default=1.0, type=np.float64, help='value of labmda1')
    parser.add_argument('--lambda2', default=1.0, type=np.float64, help='value of labmda2')
    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the shadow dataset to inject the backdoor')

    parser.add_argument('--reference_file', default='', type=str, help='path to the reference inputs')
    parser.add_argument('--trigger_file', default='', type=str, help='path to the trigger')
    parser.add_argument('--shadow_dataset', default='cifar10', type=str,  help='shadow dataset')
    parser.add_argument('--pretrained_encoder', default='', type=str, help='path to the clean encoder used to finetune the backdoored encoder')
    parser.add_argument('--encoder_usage_info', default='cifar10', type=str, help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
    #parser.add_argument('--reference_label', default=-1, type=int, help='target class in the target downstream task')

    parser.add_argument('--shadow_fraction', default=0.2, type=float, help='learning rate in SGD')

    parser.add_argument('--downstream_dataset', default='gtsrb', type=str,  help='downstream_dataset')
    parser.add_argument('--lr_downstream', default=0.0001, type=float, help='learning rate in SGD')
    

    parser.add_argument('--mode', default='badencoder', type=str, help='')#['badencoder','ada_badencoder_nosim','ada_badencoder_nodis','ada_badencoder']
    parser.add_argument('--target_label', default=0, type=int, help='')
    parser.add_argument('--n_ref', default=3, type=int, help='')

    parser.add_argument('--nn_epochs', default=500, type=int)
    parser.add_argument('--hidden_size_1', default=512, type=int)
    parser.add_argument('--hidden_size_2', default=256, type=int)
    parser.add_argument('--batch_size_downstream', default=64, type=int, help='Number of images in each mini-batch')

    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')
    args = parser.parse_args()

    args.results_dir = "/data/local/wzt/model_fix/BadEncoder/DRUPE_results/" \
                     +args.mode+"/pretrain_"+args.encoder_usage_info+"_sf"+str(args.shadow_fraction)+"/downstream_" \
                     +args.downstream_dataset+"_t"+str(args.target_label)+"/"

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    if args.reference_file == '':
        if args.encoder_usage_info in ["imagenet","CLIP"]:
            args.reference_file="./reference/{}_l{}_n{}_224.npz".format(args.downstream_dataset,str(args.target_label),str(args.n_ref))
        else:
            args.reference_file="./reference/{}_l{}_n{}.npz".format(args.downstream_dataset,str(args.target_label),str(args.n_ref))

    args.pretraining_dataset = args.encoder_usage_info

    logger_path = args.results_dir + "log.txt"
    args.logger_file = open(logger_path, 'w')

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args.data_dir = f'./data/{args.shadow_dataset.split("_")[0]}/'
    args.knn_k = 200
    args.knn_t = 0.5
    args.reference_label = args.target_label
    print(args)

    shadow_data, memory_data, test_data_clean, test_data_backdoor = get_shadow_dataset(args)
    train_loader = DataLoader(shadow_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    clean_model = get_encoder_architecture_usage(args).cuda()
    model = get_encoder_architecture_usage(args).cuda()

    print("Optimizer: SGD")
    if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
        # note that the following three dataloaders are used to monitor the finetune of the pre-trained encoder, they are not required by our BadEncoder. They can be ignored if you do not need to monitor the finetune of the pre-trained encoder
        memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader_clean = DataLoader(test_data_clean, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader_backdoor = DataLoader(test_data_backdoor, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        optimizer = torch.optim.SGD(model.f.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.visual.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)

    # Initialize the BadEncoder and load the pretrained encoder
    if args.pretrained_encoder != '':
        print(f'load the clean model from {args.pretrained_encoder}')
        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            checkpoint = torch.load(args.pretrained_encoder)
            clean_model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
        elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
            checkpoint = torch.load(args.pretrained_encoder)
            clean_model.visual.load_state_dict(checkpoint['state_dict'])
            model.visual.load_state_dict(checkpoint['state_dict'])
        else:
            raise NotImplementedError()

    patience = 1
    cost_multiplier_up   = 1.25
    cost_multiplier_down = 1.25 ** 1.25
    init_cost = 0.1
    cost = init_cost
    cost_up_counter = 0
    cost_down_counter = 0
    
    init_cost_1 = 0.001
    cost_1 = init_cost_1
    cost_up_counter_1 = 0
    cost_down_counter_1 = 0

    if args.encoder_usage_info in ["CLIP"]:
        init_cost = 0.01
        init_cost_1 = 0.0001

    args.measure = 0
    measure_best = float('inf')

    # training loop
    for epoch in range(1, args.epochs + 1):
        print("=================================================")
        if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
            if epoch<2:
                warm_up = True
            else:
                warm_up = False

            if epoch==1:
                get_clean_dev=True
            else:
                get_clean_dev=False

            if epoch%10==0:
                cal_cluster_based_dist=True
            else:
                cal_cluster_based_dist=False
            
            train_loss = train(model.f, clean_model.f, train_loader, optimizer, args, warm_up=warm_up, get_clean_dev=get_clean_dev,cal_cluster_based_dist=cal_cluster_based_dist)
            # the test code is used to monitor the finetune of the pre-trained encoder, it is not required by our BadEncoder. It can be ignored if you do not need to monitor the finetune of the pre-trained encoder
            #_ = test(model.f, memory_loader, test_loader_clean, test_loader_backdoor,epoch, args)
        elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
            if args.encoder_usage_info == 'imagenet':
                warm_up_epoch = 2
            elif args.encoder_usage_info == 'CLIP':
                warm_up_epoch = 1

            if epoch<warm_up_epoch:
                warm_up = True
            else:
                warm_up = False

            if epoch==1:
                get_clean_dev=True
            else:
                get_clean_dev=False

            if epoch%10==0:
                cal_cluster_based_dist=True
            else:
                cal_cluster_based_dist=False

            train_loss = train(model.visual, clean_model.visual, train_loader, optimizer, args, warm_up=warm_up, get_clean_dev=get_clean_dev,cal_cluster_based_dist=cal_cluster_based_dist)
        else:
            raise NotImplementedError()

        print("args.measure:",args.measure)
        print("measure_best:",measure_best)
        if epoch > 24 and args.measure < measure_best:
            measure_best = args.measure
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir+'/best.pth')

        if epoch>= (args.epochs-int(0.5*args.epochs)) and epoch % 10 == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir+'/epoch'+str(epoch)+'.pth')
            reference_label=str(args.target_label)

            try:
                model.f.eval()
            except:
                model.visual.eval()
            train_downstream_classifier(model)


