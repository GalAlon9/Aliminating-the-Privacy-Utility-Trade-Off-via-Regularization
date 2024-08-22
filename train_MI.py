#!/usr/bin/env python3
import argparse
import json
import os
import time
import copy
import random
from collections import defaultdict

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed

import models
import datasets_multiclass as datasets
from utils import *
# from logger import Logger
# import wandb

from thirdparty.repdistiller.helper.loops import train_distill, train_distill_hide, train_distill_linear, train_vanilla, train_negrad, train_bcu, train_bcu_distill
from thirdparty.repdistiller.helper.pretrain import init
from thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate

def adjust_learning_rate(optimizer, epoch):
    if args.step_size is not None:lr = args.lr * 0.1 ** (epoch//args.step_size)
    else:lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def l2_penalty(model,model_init,weight_decay):
    l2_loss = 0
    for (k,p),(k_init,p_init) in zip(model.named_parameters(),model_init.named_parameters()):
        if p.requires_grad:
            l2_loss +=  (p-p_init).pow(2).sum()
    l2_loss *= (weight_decay/2.)
    return l2_loss
    
def run_epoch(args, model, model_init, train_loader, test_loader, criterion=torch.nn.CrossEntropyLoss(), optimizer=None, scheduler=None, epoch=0, weight_decay=0.0, mode='train', quiet=False):
    if mode == 'train':
        model.train()
    elif mode == 'test':
        model.eval()
    elif mode == 'dry_run':
        model.eval()
        set_batchnorm_mode(model, train=True)
    else:
        raise ValueError("Invalid mode.")
    
    if args.disable_bn:
        set_batchnorm_mode(model, train=False)
    
    mult=0.5 if args.lossfn=='mse' else 1
    metrics = AverageMeter()

    with torch.set_grad_enabled(mode != 'test'):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            
            if args.lossfn=='mse':
                target=(2*target-1)
                target = target.type(torch.cuda.FloatTensor).unsqueeze(1)
                
            if 'mnist' in args.dataset:
                data=data.view(data.shape[0],-1)
                
            output = model(data)
            loss = mult*criterion(output, target) + l2_penalty(model,model_init,weight_decay)
            
            if args.l1:
                l1_loss = sum([p.norm(1) for p in model.parameters()])
                loss += args.weight_decay * l1_loss

            if not quiet:
                metrics.update(n=data.size(0), loss=loss.item(), error=get_error(output, target))
            
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    log_metrics(mode, metrics, epoch)
    # logger.append('train' if mode=='train' else 'test', epoch=epoch, loss=metrics.avg['loss'], error=metrics.avg['error'], 
    #               lr=optimizer.param_groups[0]['lr'])
    print('Learning Rate : {}'.format(optimizer.param_groups[0]['lr']))
    
    return metrics.avg['error'], metrics.avg['loss']  # Return the average error and loss



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score
import random
import torch.nn as nn

def cm_score(estimator, X, y):
    y_pred = estimator.predict(X)
    cnf_matrix = confusion_matrix(y, y_pred)
    
    FP = cnf_matrix[0][1] 
    FN = cnf_matrix[1][0] 
    TP = cnf_matrix[0][0] 
    TN = cnf_matrix[1][1]


    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print (f"FPR:{FPR:.2f}, FNR:{FNR:.2f}, FP{FP:.2f}, TN{TN:.2f}, TP{TP:.2f}, FN{FN:.2f}")
    return ACC


def evaluate_attack_model(sample_loss,
                          members,
                          n_splits = 5,
                          random_state = None):
  """Computes the cross-validation score of a membership inference attack.
  Args:
    sample_loss : array_like of shape (n,).
      objective function evaluated on n samples.
    members : array_like of shape (n,),
      whether a sample was used for training.
    n_splits: int
      number of splits to use in the cross-validation.
    random_state: int, RandomState instance or None, default=None
      random state to use in cross-validation splitting.
  Returns:
    score : array_like of size (n_splits,)
  """

  unique_members = np.unique(members)
  if not np.all(unique_members == np.array([0, 1])):
    raise ValueError("members should only have 0 and 1s")

  attack_model = LogisticRegression()
  cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)
  cv_scores = cross_val_score(attack_model, sample_loss, members, cv=cv, scoring=cm_score)

  # Train the logistic regression model on the entire dataset
  attack_model.fit(sample_loss, members)
  
  # Calculate the accuracy on the same dataset (not cross-validation)
  y_pred = attack_model.predict(sample_loss)
  logistic_model_accuracy = accuracy_score(members, y_pred)
  
  return cv_scores, logistic_model_accuracy

def membership_inference_attack(model, t_loader, f_loader, seed):
    import matplotlib.pyplot as plt
    import seaborn as sns
    

    fgt_cls = list(np.unique(f_loader.dataset.targets))
    indices = [i in fgt_cls for i in t_loader.dataset.targets]
    t_loader.dataset.data = t_loader.dataset.data[indices]
    t_loader.dataset.targets = t_loader.dataset.targets[indices]

    
    cr = nn.CrossEntropyLoss(reduction='none')
    test_losses = []
    forget_losses = []
    model.eval()
    mult = 0.5 if args.lossfn=='mse' else 1
    dataloader = torch.utils.data.DataLoader(t_loader.dataset, batch_size=128, shuffle=False)
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(args.device), target.to(args.device)            
        if args.lossfn=='mse':
            target=(2*target-1)
            target = target.type(torch.cuda.FloatTensor).unsqueeze(1)
        if 'mnist' in args.dataset:
            data=data.view(data.shape[0],-1)
        output = model(data)
        loss = mult*cr(output, target)
        test_losses = test_losses + list(loss.cpu().detach().numpy())
    del dataloader
    dataloader = torch.utils.data.DataLoader(f_loader.dataset, batch_size=128, shuffle=False)
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(args.device), target.to(args.device)            
        if args.lossfn=='mse':
            target=(2*target-1)
            target = target.type(torch.cuda.FloatTensor).unsqueeze(1)
        if 'mnist' in args.dataset:
            data=data.view(data.shape[0],-1)
        output = model(data)
        loss = mult*cr(output, target)
        forget_losses = forget_losses + list(loss.cpu().detach().numpy())
    del dataloader

    np.random.seed(seed)
    random.seed(seed)
    if len(forget_losses) > len(test_losses):
        forget_losses = list(random.sample(forget_losses, len(test_losses)))
    elif len(test_losses) > len(forget_losses):
        test_losses = list(random.sample(test_losses, len(forget_losses)))
    


    test_labels = [0]*len(test_losses)
    forget_labels = [1]*len(forget_losses)
    features = np.array(test_losses + forget_losses).reshape(-1,1)
    labels = np.array(test_labels + forget_labels).reshape(-1)
    features = np.clip(features, -100, 100)
    score, accuracy = evaluate_attack_model(features, labels, n_splits=5, random_state=seed)

    return score, accuracy
    
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, choices=['train', 'forget'])
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Use data augmentation')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Use data augmentation')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--dataroot', type=str, default='data/')
    parser.add_argument('--disable-bn', action='store_true', default=False,
                        help='Put batchnorm in eval mode and don\'t update the running averages')
    parser.add_argument('--epochs', type=int, default=4000, metavar='N',
                        help='number of epochs to train (default: 31)')
    parser.add_argument('--filters', type=float, default=1.0,
                        help='Percentage of filters')
    parser.add_argument('--forget-class', type=str, default=None,
                        help='Class to forget')
    parser.add_argument('--l1', action='store_true', default=False,
                        help='uses L1 regularizer instead of L2')
    parser.add_argument('--lossfn', type=str, default='ce',
                        help='Cross Entropy: ce or mse')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--model', default='mlp')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of Classes')
    parser.add_argument('--num-to-forget', type=int, default=None,
                        help='Number of samples of class to forget')
    parser.add_argument('--confuse-mode', action='store_true', default=False,
                        help="enables the interclass confusion test")
    parser.add_argument('--name', default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--step-size', default=None, type=int, help='learning rate scheduler')
    parser.add_argument('--unfreeze-start', default=None, type=str, help='All layers are freezed except the final layers starting from unfreeze-start')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='M',
                        help='Weight decay (default: 0)')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,30,30', help='lr decay epochs')
    parser.add_argument('--sgda-learning-rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--print_freq', type=int, default=500, help='print frequency')
    parser.add_argument('--job_array',action='store_true', default =False, help='if we need to turn off running with slurm job array')

    args = parser.parse_args()
    
    
    print(args.job_array)
    if  not args.job_array:
    #remove this if running without slurm job array
        k= os.environ['SLURM_ARRAY_TASK_ID']
        k = float(k)/10
        args.filters = k
    #### 

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    
    if args.forget_class is not None:
        clss = args.forget_class.split(',')
        args.forget_class = list([])
        for c in clss:
            args.forget_class.append(int(c))

    manual_seed(args.seed)
    
    if args.step_size==None:args.step_size=args.epochs+1
    
    if args.name is None:
        args.name = f"{args.dataset}_{args.model}_{str(args.filters).replace('.','_')}"
        if args.split == 'train':
            args.name += f"_forget_{None}"
        else:
            args.name += f"_forget_{args.forget_class}"
            if args.num_to_forget is not None:
                args.name += f"_num_{args.num_to_forget}"
        if args.unfreeze_start is not None:
            args.name += f"_unfreeze_from_{args.unfreeze_start.replace('.','_')}"
        if args.augment:
            args.name += f"_augment"
        if args.l1:
            args.name += f"_l1"
        args.name+=f"_lr_{str(args.lr).replace('.','_')}"
        args.name+=f"_bs_{str(args.batch_size)}"
        args.name+=f"_ls_{args.lossfn}"
        args.name+=f"_wd_{str(args.weight_decay).replace('.','_')}"
        args.name+=f"_seed_{str(args.seed)}"
        
    print(f'Checkpoint name: {args.name}')
    
    mkdir('logs')

    # logger = Logger(index=args.name+'_training')
    # logger['args'] = args
    # logger['checkpoint'] = os.path.join('models/', logger.index+'.pth')
    # logger['checkpoint_step'] = os.path.join('models/', logger.index+'_{}.pth')

    # print("[Logging in {}]".format(logger.index))
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    os.makedirs('checkpoints', exist_ok=True)

    train_loader, valid_loader, test_loader = datasets.get_loaders(args.dataset, class_to_replace=args.forget_class,
                                                     num_indexes_to_replace=args.num_to_forget, confuse_mode=args.confuse_mode,
                                                     batch_size=args.batch_size, split=args.split, seed=args.seed,
                                                    root=args.dataroot, augment=args.augment)
    
    num_classes = max(train_loader.dataset.targets) + 1 if args.num_classes is None else args.num_classes
    args.num_classes = num_classes
    print(f"Number of Classes: {num_classes}")
    model = models.get_model(args.model, num_classes=num_classes, filters_percentage=args.filters).to(args.device)
    
    if args.model=='allcnn':classifier_name='classifier.'
    elif 'resnet' in args.model:classifier_name='linear.'
    
    if args.resume is not None:
        state = torch.load(args.resume)
        state = {k: v for k, v in state.items() if not k.startswith(classifier_name)}
        incompatible_keys = model.load_state_dict(state, strict=False)
        assert all([k.startswith(classifier_name) for k in incompatible_keys.missing_keys])
    model_init = copy.deepcopy(model)
    

    torch.save(model.state_dict(), f"MIA/checkpoints/{args.name}_init.pt")
    
    parameters = model.parameters()
    if args.unfreeze_start is not None:
        parameters = []
        layer_index = 1e8
        for i, (n,p) in enumerate(model.named_parameters()):
            if (args.unfreeze_start in n) or (i > layer_index):
                layer_index = i
                parameters.append(p)
        
    weight_decay = args.weight_decay if not args.l1 else 0.
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer = optim.Adam(parameters,lr=args.lr,weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss().to(args.device) if args.lossfn=='ce' else torch.nn.MSELoss().to(args.device)
    #optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=0.1, last_epoch=-1)

    train_time = 0
    all_metrics = []  # To store the metrics for each epoch
    metrics_file_path = f"MIA/{args.name}_metrics.json"
    
    with open(metrics_file_path, 'a') as metrics_file:
        for epoch in range(args.epochs):
            # adjust_learning_rate(optimizer, epoch)
            sgda_adjust_learning_rate(epoch, args, optimizer)

            t1 = time.time()

            # Run training phase
            train_error, train_loss = run_epoch(args, model, model_init, train_loader, test_loader, criterion, optimizer, scheduler, epoch, weight_decay, mode='train', quiet=args.quiet)

            # Run testing phase
            test_error, test_loss = run_epoch(args, model, model_init, test_loader, test_loader, criterion, optimizer, scheduler, epoch, weight_decay, mode='test', quiet=args.quiet)

            # Evaluate MIA Score
            mia_score, mia_accuracy = membership_inference_attack(model, train_loader, test_loader, args.seed)
            mia_score = mia_score.mean()  # Assuming you want the average score

            # Save the metrics for the current epoch
            metrics = {
                'epoch': epoch,
                'train_error': train_error,
                'train_loss': train_loss,
                'test_error': test_error,
                'test_loss': test_loss,
                'mia_score': mia_score,
                'mia_train_accuracy': mia_accuracy
            }
            all_metrics.append(metrics)

            # Write the metrics to the file, one per line
            metrics_file.write(json.dumps(metrics) + '\n')
            # save the changes to the file
            metrics_file.flush()

            # Save the model checkpoint 4 times during training (make sure to include the first and last epochs)
            if epoch == 0 or epoch == args.epochs // 4 or epoch == args.epochs // 2 or epoch == args.epochs - 1:
                torch.save(model.state_dict(), f"MIA/checkpoints/{args.name}_{epoch}.pt")
                print(f"Checkpoint saved at epoch {epoch}")

            t2 = time.time()
            train_time += np.round(t2-t1, 2)
            print(f'Epoch Time: {np.round(t2-t1,2)} sec')

    print(f'Pure training time: {train_time} sec')


# if __name__ == '__main__':
#     main()

