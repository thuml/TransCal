#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import csv
import pre_process as prep
from torch.utils.data import DataLoader
from data_list import ImageList
import multiprocessing
import sys
pytorch_path = os.path.abspath(os.path.dirname(__file__))
project_path = os.path.dirname(pytorch_path)
sys.path.extend([project_path])


def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def generate_feature_wrapper(loader, model,dir,output_name=None):
    def gather_outputs(selected_loader, output_name):
        if 'MDD' in output_name:
            base_network, bottleneck_layer, classifier_layer = model[0], model[1], model[2]
        elif 'MCD' in output_name:
            feature_extractor, C1, C2 = model[0],model[1], model[2]
        with torch.no_grad():
            start_test = True
            iter_loader = iter(selected_loader)
            for i in range(len(selected_loader)):
                inputs, labels, _ = iter_loader.next()
                inputs = inputs.cuda()
                if 'MDD' in output_name:
                    conv_features = base_network(inputs)
                    fc_features = bottleneck_layer(conv_features)
                    logit = classifier_layer(fc_features)
                elif 'MCD' in output_name:
                    fc_features = feature_extractor(inputs)
                    output_C1 = C1(fc_features)
                    output_C2 = C2(fc_features)
                    logit = (output_C1 + output_C2) / 2.0
                else:
                    fc_features, logit = model(inputs)

                if start_test:
                    features_ = fc_features.float().cpu()
                    outputs_ = logit.float().cpu()
                    labels_ = labels
                    start_test = False
                else:
                    features_ = torch.cat((features_, fc_features.float().cpu()), 0)
                    outputs_ = torch.cat((outputs_, logit.float().cpu()), 0)
                    labels_ = torch.cat((labels_, labels), 0)
            return features_, outputs_, labels_

    COMBINE = False
    if COMBINE:
        def save_result(result, name):
            with open(os.path.join(dir, name + '.csv'), 'w') as file:
                writer = csv.writer(file)
                writer.writerow(result)

        def save_results(results, name):
            with open(os.path.join(dir, name + '.csv'), 'w') as file:
                writer = csv.writer(file)
                writer.writerows(results)
        features_source_train, outputs_source_train, labels_source_train = gather_outputs(loader['source_train'], output_name)
        features_source_val, outputs_source_val, labels_source_val = gather_outputs(loader['source_val'], output_name)
        features_target, outputs_target, labels_target = gather_outputs(loader['target'], output_name)

        N_s, _ = features_source_train.shape
        N_t, _ = features_target.shape
        N_v, _ = features_source_val.shape
        num = np.array([N_s, N_t, N_v])
        save_result(num, output_name + '_num')

        total_feature = np.concatenate((features_source_train, features_target, features_source_val))
        save_results(total_feature, output_name + '_feature')

        total_outputs = np.concatenate((outputs_source_train, outputs_target, outputs_source_val))
        save_results(total_outputs, output_name + '_output')

        total_labels = np.concatenate((labels_source_train, labels_target, labels_source_val))
        save_result(total_labels, output_name + '_label')
    else:
        def save(loader, output_name, data_name):
            features, outputs, labels = gather_outputs(loader, output_name)
            np.save(dir + '/' + output_name + '_' + data_name + '_feature.npy', features)
            np.save(dir + '/' + output_name + '_' + data_name + '_output.npy', outputs)
            np.save(dir + '/' + output_name + '_' + data_name + '_label.npy', labels)

        print("-----------------Saving:source_train-----------")
        save(loader['source_train'], output_name, 'source_train')
        print("-----------------Saving:source_val-----------")
        save(loader['source_val'], output_name, 'source_val')
        print("-----------------Saving:target-----------")
        save(loader['target'], output_name, 'target')


def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source_train"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source_train"] = ImageList(open(data_config["source_train"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
    dset_loaders["source_train"] = DataLoader(dsets["source_train"], batch_size=train_bs, \
            shuffle=False, num_workers=4)
    dsets["source_val"] = ImageList(open(data_config["source_val"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
    dset_loaders["source_val"] = DataLoader(dsets["source_val"], batch_size=train_bs, \
                                        shuffle=False, num_workers=4)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=False, num_workers=4)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                shuffle=False, num_workers=4) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=4)

    backbone = torch.load('snapshot/' + config["dataset"] + '/' + config['method'] + '/' + config['task'] + '/iter_20000_model.pth.tar')
    backbone = backbone.cuda()
    backbone.train(False)

    feature_dir = os.path.join('../features', config["dataset"], config['method'])
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    output_name = config["dataset"] + '_' + config['method'] + '_' + config['task']
    print(output_name)

    generate_feature_wrapper(dset_loaders, backbone, feature_dir, output_name=output_name)

if __name__ == "__main__":

    def generate_feature_process(method, gpu_id, dataset, source, target):
        parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
        parser.add_argument('--method', type=str, default= method, choices=['CDAN', 'CDAN+E', 'DANN'])
        parser.add_argument('--gpu_id', type=str, nargs='?', default=gpu_id, help="device id to run")
        parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"])

        parser.add_argument('--dset', type=str, default=dataset, choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset or source dataset used")
        parser.add_argument('--s_train_dset_path', type=str, default='../data/' + dataset + '/' + source + '_train_list.txt', help="The source train dataset path list")

        if dataset == 'domainnet':
            parser.add_argument('--s_val_dset_path', type=str,
                                default='../data/' + dataset + '/' + source + '_test_list.txt',
                                help="The source validation dataset path list")
            parser.add_argument('--t_dset_path', type=str, default='../data/' + dataset + '/' + target + '_test_list.txt',
                                help="The target dataset path list")
        else:
            parser.add_argument('--s_val_dset_path', type=str,
                                default='../data/' + dataset + '/' + source + '_val_list.txt',
                                help="The source validation dataset path list")
            parser.add_argument('--t_dset_path', type=str, default='../data/' + dataset + '/' + target + '_list.txt',
                                help="The target dataset path list")

        parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
        parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
        parser.add_argument('--output_dir', type=str, default='ece', help="output directory of our model (in ../snapshot directory)")
        parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
        parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
        args = parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        # train config
        config = {}
        config['method'] = args.method
        config['task'] = os.path.basename(args.s_train_dset_path)[0].upper() + '2' + os.path.basename(args.t_dset_path)[0].upper()


        config["prep"] = {"test_10crop":False, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
        config["loss"] = {"trade_off":1.0}
        if "AlexNet" in args.net:
            config["prep"]['params']['alexnet'] = True
            config["prep"]['params']['crop_size'] = 227
            config["network"] = {"name":network.AlexNetFc, \
                "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
        elif "ResNet" in args.net:
            config["network"] = {"name":network.ResNetFc, \
                "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
        elif "VGG" in args.net:
            config["network"] = {"name":network.VGGFc, \
                "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
        config["loss"]["random"] = args.random
        config["loss"]["random_dim"] = 1024

        config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                               "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                               "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

        config["dataset"] = args.dset
        config["data"] = {"source_train":{"list_path":args.s_train_dset_path, "batch_size":36}, \
                          "source_val": {"list_path": args.s_val_dset_path, "batch_size": 36}, \
                          "target":{"list_path":args.t_dset_path, "batch_size":36}, \
                          "test":{"list_path":args.t_dset_path, "batch_size":4}}

        if config["dataset"] == "office":
            config["network"]["params"]["class_num"] = 31
        elif config["dataset"] == "image-clef":
            config["network"]["params"]["class_num"] = 12
        elif config["dataset"] == "visda":
            config["network"]["params"]["class_num"] = 12
        elif config["dataset"] == "office-home":
            config["network"]["params"]["class_num"] = 65
        elif config["dataset"] == "domainnet":
            config["network"]["params"]["class_num"] = 345
        else:
            raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
        train(config)

    dataset = 'office-home'
    method_list = ['CDAN+E', 'MCD', 'MDD']

    for method in method_list:
        p1 = multiprocessing.Process(target=generate_feature_process, args=(method, '0', dataset, 'Art', 'Clipart'))
        p2 = multiprocessing.Process(target=generate_feature_process, args=(method, '1', dataset, 'Art', 'Product'))
        p3 = multiprocessing.Process(target=generate_feature_process, args=(method, '2', dataset, 'Art', 'Real_World'))
        p4 = multiprocessing.Process(target=generate_feature_process, args=(method, '3', dataset, 'Clipart', 'Art'))
        p5 = multiprocessing.Process(target=generate_feature_process, args=(method, '4', dataset, 'Clipart', 'Product'))
        p6 = multiprocessing.Process(target=generate_feature_process, args=(method, '5', dataset, 'Clipart', 'Real_World'))

        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        p6.join()