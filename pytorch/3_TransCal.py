#!/usr/bin/env python
#coding:utf-8

import numpy as np
import multiprocessing
import torch
import torch.nn as nn
import csv
import os
from utils import get_weight, ECELoss, VectorOrMatrixScaling, TempScaling, CPCS, TransCal, Oracle

def cal_acc_error(logit, label):
    softmaxes = nn.Softmax(dim=1)(logit)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(label)
    accuracy = accuracies.float().mean()
    confidence = confidences.float().mean()
    error = 1 - accuracies.float()
    error = error.view(len(error), 1).float().numpy()
    return accuracy, confidence, error


"Vanilla, MatrixScaling, VectorScaling, TempScaling, CPCS, TransCal, Oracle"
def calibration_in_DA(logits_source_val, labels_source_val, logits_target, labels_target, cal_method=None, weight=None, bias_term=True, variance_term=True):
    ece_criterion = ECELoss()
    if cal_method == 'VectorScaling' or cal_method == 'MatrixScaling':
        ece = VectorOrMatrixScaling(logits_source_val, labels_source_val, logits_target, labels_target, cal_method=cal_method)
        optimal_temp = 0.0
    else:
        if cal_method == 'TempScaling':
            cal_model = TempScaling()
            optimal_temp = cal_model.find_best_T(logits_source_val, labels_source_val)
        elif cal_method == 'CPCS':
            cal_model = CPCS()
            optimal_temp = cal_model.find_best_T(logits_source_val, labels_source_val, torch.from_numpy(weight))
        elif cal_method == 'TransCal':
            """calibrate the source model first and then attain the optimal temperature for the source dataset"""
            cal_model = TempScaling()
            optimal_temp_source = cal_model.find_best_T(logits_source_val, labels_source_val)
            _, source_confidence, error_source_val = cal_acc_error(logits_source_val / optimal_temp_source, labels_source_val)

            cal_model = TransCal(bias_term, variance_term)
            optimal_temp = cal_model.find_best_T(logits_target.numpy(), weight, error_source_val, source_confidence.item())
        elif cal_method == 'Oracle':
            cal_model = Oracle()
            optimal_temp = cal_model.find_best_T(logits_target, labels_target)
        ece = ece_criterion(logits_target / optimal_temp, labels_target).item()
    print(cal_method, ece)
    return ece, optimal_temp

def estimate_ece(method, dataset, task, root_dir = '../features/'):
    feature_path = os.path.join(root_dir, dataset, method, dataset + '_' + method + '_' + task)
    output_name = method + "_" + task
    print(output_name)
    print(feature_path)

    if os.path.exists(feature_path + '_feature.csv'):
        COMBINE = True
    else:
        COMBINE = False

    if COMBINE:
        num = np.genfromtxt(feature_path + '_num.csv', delimiter=',')
        n_tr, n_t, n_v = int(num[0]), int(num[1]), int(num[2])
        # print(n_tr, n_t, n_v)

        features = np.genfromtxt(feature_path + '_feature.csv', delimiter=',')
        outputs = np.genfromtxt(feature_path + '_output.csv', delimiter=',')
        labels = np.genfromtxt(feature_path + '_label.csv', delimiter=',')

        features_source_train, logits_source_train, labels_source_train = \
            features[:n_tr], torch.from_numpy(outputs[:n_tr]), torch.from_numpy(labels[:n_tr]).long()
        features_target, logits_target, labels_target = \
            features[n_tr:n_tr+n_t], torch.from_numpy(outputs[n_tr:n_tr+n_t]), torch.from_numpy(labels[n_tr:n_tr+n_t]).long()
        features_source_val, logits_source_val, labels_source_val = \
            features[-n_v:], torch.from_numpy(outputs[-n_v:]), torch.from_numpy(labels[-n_v:]).long()
    else:
        features_source_train = np.load(feature_path + '_source_train_feature.npy')
        logits_source_train = torch.from_numpy(np.load(feature_path + '_source_train_output.npy'))
        labels_source_train = torch.from_numpy(np.load(feature_path + '_source_train_label.npy')).long()

        features_target = np.load(feature_path + '_target_feature.npy')
        logits_target = torch.from_numpy(np.load(feature_path + '_target_output.npy'))
        labels_target = torch.from_numpy(np.load(feature_path + '_target_label.npy')).long()

        features_source_val = np.load(feature_path + '_source_val_feature.npy')
        logits_source_val = torch.from_numpy(np.load(feature_path + '_source_val_output.npy'))
        labels_source_val = torch.from_numpy(np.load(feature_path + '_source_val_label.npy')).long()

    accuracy_source_train, confidence_source_train, error_source_train = cal_acc_error(logits_source_train, labels_source_train)
    accuracy_target, confidence_target, error_target = cal_acc_error(logits_target, labels_target)
    accuracy_source_val, confidence_source_val, error_source_val = cal_acc_error(logits_source_val, labels_source_val)

    print("accuracy_source_train: " + str(accuracy_source_train))
    print("accuracy_source_val: " + str(accuracy_source_val))
    print("accuracy_target: " + str(accuracy_target))

    ece_criterion = ECELoss()
    source_train_ece = ece_criterion(logits_source_train, labels_source_train).item()
    source_val_ece = ece_criterion(logits_source_val, labels_source_val).item()
    print("source_train_ece: {:4f}".format(source_train_ece))
    print("source_val_ece: {:4f}".format(source_val_ece))

    "Method 1: Vanilla Model (before calibration)"
    vanilla_target_ece = ece_criterion(logits_target, labels_target).item()
    print("vanilla_target_ece: {:4f}".format(vanilla_target_ece))


    repeat_times = 10
    for _idx_ in range(repeat_times):
        weight = get_weight(features_source_train, features_target, features_source_val)

        "Method 2: Matrix Scaling"
        ece_matrix_scaling, optimal_temp_matrix_scaling = calibration_in_DA(logits_source_val, labels_source_val, logits_target, labels_target, cal_method='MatrixScaling')

        "Method 3: Vector Scaling"
        ece_vector_scaling, optimal_temp_vector_scaling = calibration_in_DA(logits_source_val, labels_source_val, logits_target, labels_target, cal_method='VectorScaling')

        "Method 4: Temperature Scaling"
        ece_temp_scaling, optimal_temp_temp_scaling = calibration_in_DA(logits_source_val, labels_source_val, logits_target, labels_target, cal_method='TempScaling')

        "Method 5: CPCS"
        ece_CPCS, optimal_temp_CPCS = calibration_in_DA(logits_source_val,labels_source_val, logits_target,labels_target, cal_method='CPCS', weight=weight)

        "Method 6: TransCal"
        ece_TransCal, optimal_temp_TransCal = calibration_in_DA(logits_source_val, labels_source_val, logits_target, labels_target, cal_method='TransCal', weight=weight)

        "Method 7: Oracle (Assume Labels in the target domain are aviable)"
        ece_oracle, optimal_temp_oracle = calibration_in_DA(logits_source_val, labels_source_val, logits_target, labels_target, cal_method='Oracle')

        result = [output_name, source_train_ece, source_val_ece, vanilla_target_ece,
                  ece_matrix_scaling, ece_vector_scaling, ece_temp_scaling, ece_CPCS, ece_TransCal, ece_oracle]
        result = np.array(result)
        with open('calibration_in_DA.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(result)


dataset = 'office-home'
method_list = ['CDAN+E','MCD', 'MDD']

DEBUG = True
if DEBUG:
    estimate_ece('CDAN+E', 'office-home', 'A2C')
    estimate_ece('CDAN+E', 'office-home', 'A2P')
    estimate_ece('CDAN+E', 'office-home', 'A2R')
else:
    if dataset == 'domainnet':
        for method in method_list:
            p1 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'S2I'))
            p2 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'S2R'))
            p3 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'I2R'))
            p4 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'I2S'))
            p5 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'R2I'))
            p6 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'R2S'))

            p1 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'S2I'))
            p2 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'S2Q'))
            p3 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'S2R'))
            p4 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'I2R'))
            p5 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'I2Q'))
            p6 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'I2S'))

            p7 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'Q2I'))
            p8 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'Q2R'))
            p9 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'Q2S'))
            p10 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'R2I'))
            p11 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'R2Q'))
            p12 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'R2S'))


            p1.start()
            p2.start()
            p3.start()
            p4.start()
            p5.start()
            p6.start()
            p7.start()
            p8.start()
            p9.start()
            p10.start()
            p11.start()
            p12.start()

            p1.join()
            p2.join()
            p3.join()
            p4.join()
            p5.join()
            p6.join()
            p7.join()
            p8.join()
            p9.join()
            p10.join()
            p11.join()
            p12.join()

    elif dataset == 'office-home':
        for method in method_list:
            p1 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'A2C'))
            p2 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'A2P'))
            p3 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'A2R'))
            p4 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'C2A'))
            p5 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'C2P'))
            p6 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'C2R'))

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

            p7 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'P2A'))
            p8 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'P2C'))
            p9 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'P2R'))
            p10 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'R2A'))
            p11 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'R2C'))
            p12 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'R2P'))

            p7.start()
            p8.start()
            p9.start()
            p10.start()
            p11.start()
            p12.start()

            p7.join()
            p8.join()
            p9.join()
            p10.join()
            p11.join()
            p12.join()

    elif dataset == 'visda':
        p1 = multiprocessing.Process(target=estimate_ece, args=('MDD', 'visda', 'T2V'))
        p2 = multiprocessing.Process(target=estimate_ece, args=('MCD', 'visda', 'T2V'))
        p3 = multiprocessing.Process(target=estimate_ece, args=('CDAN+E', 'visda', 'T2V'))

        p1.start()
        p2.start()
        p3.start()


        p1.join()
        p2.join()
        p3.join()


    elif dataset == 'sketch':
        p1 = multiprocessing.Process(target=estimate_ece, args=('MDD', 'sketch', 'I2S'))
        p2 = multiprocessing.Process(target=estimate_ece, args=('MCD', 'sketch', 'I2S'))
        p3 = multiprocessing.Process(target=estimate_ece, args=('CDAN+E', 'sketch', 'I2S'))

        p1.start()
        p2.start()
        p3.start()


        p1.join()
        p2.join()
        p3.join()


    elif dataset == 'office':
        for method in method_list:
            p1 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'A2W'))
            p2 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'A2D'))
            p3 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'D2A'))
            p4 = multiprocessing.Process(target=estimate_ece, args=(method, dataset, 'D2W'))

            p1.start()
            p2.start()
            p3.start()
            p4.start()

            p1.join()
            p2.join()
            p3.join()
            p4.join()


