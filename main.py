from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import random
import threading

random.seed(1)
import numpy as np
import shutil

np.random.seed(1)

import argparse
from model.digit5 import CNN, Classifier
from model.officecaltech10 import OfficeCaltechNet, OfficeCaltechClassifier
from model.officehome import OfficehomeNet, OfficehomeClassifier

from model.domainnet import DomainNet, DomainNetClassifier
from datasets.DigitFive import digit5_dataset_read
from lib.utils.federated_utils import *
from lib.utils.loss import Icvar_p
from train.train import source_train, target_train,\
    visual_all, visual_source, visual_test, test, validate_target
from datasets.OfficeCaltech10 import get_office_caltech10_dloader
from datasets.OfficeHome import get_office_home_dloader
from datasets.DomainNet import get_domainnet_dloader

import torch.nn.functional as F

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import sys
import yaml
from os import path
import torch.multiprocessing
from lib.utils.loggings import get_logger

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="OfficeCaltech10.yaml")
    parser.add_argument('--dataset', default="OfficeCaltech10", type=str,
                        help="The target domain we want to perform domain adaptation")
    parser.add_argument('--gpu', default="0,1", type=str, help='GPU to use, -1 for CPU training')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--warm_target', type=int, default=5, help='the start epochs to train the target networks')
    parser.add_argument('--beta', type=float, default=0.3, help='parameter for beta')
    parser.add_argument('--communication_rounds', type=float, default=1, help='parameter for communication_rounds')
    parser.add_argument('--alpha_z', type=float, default=0.6, help='parameter for time integrated')

    parser.add_argument('--p_icvar', type=float, default=5, help='parameter for icvar')
    parser.add_argument('--alpha_mix', type=float, default=1.0, help='parameter for mixup learning loss')
    parser.add_argument('--num_gradual', type=int, default=5,
                        help='how many epochs for linear drop rate, can be 5, 10, 15.')
    parser.add_argument('--exponent', type=float, default=1,
                        help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
    parser.add_argument('-bp', '--base-path', default="/media/lxh/document/")

    parser.add_argument('--target-domain', default="amazon", type=str,
                        help="The target domain we want to perform domain adaptation")

    parser.add_argument('--source-domains', type=str, nargs="+", help="The source domains we want to use")
    parser.add_argument('-j', '--workers', default=4, metavar='N',
                        help='number of data loading workers (default: 8)')
    # Train Strategy Parameters
    parser.add_argument('-t', '--train-time', default=1, type=str,
                        metavar='N', help='the x-th time of training')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-dp', '--data-parallel', action='store_false', help='Use Data Parallel')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--forget_rate', type=float, default=0.06, help='forget rate')
    parser.add_argument('--p_threshold', type=float, default=0.5, help='clean probability threshold')

    # Optimizer Parameters2
    parser.add_argument('--optimizer', default="SGD", type=str, metavar="Optimizer Name")
    parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M', help='Momentum in SGD')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float)
    parser.add_argument('-bm', '--bn-momentum', type=float, default=0.1, help="the batchnorm momentum parameter")
    parser.add_argument('--outf', default='results', help='folder to output images and model checkpoints')

    opt = parser.parse_args()

    # Creating log directory
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    try:
        os.makedirs(os.path.join(opt.outf, 'visualization'))
    except OSError:
        pass
    try:
        os.makedirs(os.path.join(opt.outf, 'models'))
    except OSError:
        pass

    try:
        os.makedirs(os.path.join('log'))
    except OSError:
        pass

    file = open(r"./config/{}".format(opt.config))
    configs = yaml.full_load(file)
    # set the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    cudnn.benchmark = True

    # Setting random seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    # GPU/CPU flags

def main(args=opt, configs=configs):
    # set the dataloader list, model list, optimizer list, optimizer schedule list
    train_dloaders = []
    test_dloaders = []
    models = []
    classifiers = []
    optimizers = []
    classifier_optimizers = []
    optimizer_schedulers = []
    classifier_optimizer_schedulers = []

    # Creating data loaders
    if configs["DataConfig"]["dataset"] == "DigitFive":
        domains = ['mnistm', 'mnist', 'syn', 'usps', 'svhn']
        i = 0
        # [0,1]: target dataset, target backbone, [2:-1]: source dataset, source backbone
        # generate dataset for train and target
        print("load target domain {}".format(args.target_domain))
        target_train_dloader, target_test_dloader = digit5_dataset_read(args.base_path,
                                                                        args.target_domain,
                                                                        configs["TrainingConfig"]["batch_size"])
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        # generate CNN and Classifier for target domain
        models.append(CNN(args.data_parallel).cuda())
        classifiers.append(Classifier(args.data_parallel).cuda())
        models.append(CNN(args.data_parallel).cuda())
        classifiers.append(Classifier(args.data_parallel).cuda())
        domains.remove(args.target_domain)
        args.source_domains = domains
        print("target domain {} loaded".format(args.target_domain))
        # create DigitFive dataset
        print("Source Domains :{}".format(domains))
        for domain in domains:
            i += 1
            # generate dataset for source domain
            source_train_dloader, source_test_dloader = digit5_dataset_read(args.base_path, domain,
                                                                            configs["TrainingConfig"]["batch_size"])
            train_dloaders.append(source_train_dloader)
            test_dloaders.append(source_test_dloader)

            # generate CNN and Classifier for source domain
            models.append(CNN(args.data_parallel).cuda())
            classifiers.append(Classifier(args.data_parallel).cuda())
            print("Domain {} Preprocess Finished".format(domain))
        num_classes = 10

    elif configs["DataConfig"]["dataset"] == "OfficeCaltech10":
        domains = ['amazon','webcam', 'dslr', "caltech"]
        i = 0
        target_train_dloader, target_test_dloader = get_office_caltech10_dloader(args.base_path,
                                                                                 args.target_domain,
                                                                                 configs["TrainingConfig"]["batch_size"]
                                                                                 , args.workers)
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        models.append(
            OfficeCaltechNet(configs["ModelConfig"]["tarbackbone"], bn_momentum=args.bn_momentum,
                             pretrained=args.target_pretrain,
                             data_parallel=args.data_parallel).cuda())
        classifiers.append(
            OfficeCaltechClassifier(configs["ModelConfig"]["tarbackbone"], 10, args.data_parallel).cuda()
        )
        models.append(
            OfficeCaltechNet(configs["ModelConfig"]["tarbackbone"],bn_momentum=args.bn_momentum,
                             pretrained=args.target_pretrain,
                             data_parallel=args.data_parallel).cuda())
        classifiers.append(
            OfficeCaltechClassifier(configs["ModelConfig"]["tarbackbone"], 10, args.data_parallel).cuda()
        )
        domains.remove(args.target_domain)
        args.source_domains = domains
        for domain in domains:
            i += 1
            source_train_dloader, source_test_dloader = get_office_caltech10_dloader(args.base_path, domain,
                                                                                     configs["TrainingConfig"][
                                                                                         "batch_size"], args.workers)
            train_dloaders.append(source_train_dloader)
            test_dloaders.append(source_test_dloader)

            models.append(
                OfficeCaltechNet(configs["ModelConfig"]["backbone"], args.bn_momentum,
                                 pretrained=configs["ModelConfig"]["pretrained"],
                                 data_parallel=args.data_parallel).cuda())
            classifiers.append(
                OfficeCaltechClassifier(configs["ModelConfig"]["backbone"], 10, args.data_parallel).cuda()
            )

        num_classes = 10

    elif configs["DataConfig"]["dataset"] == "OfficeHome":
        domains = ['Art', 'Clipart', 'Product', 'Real World']
        i = 0
        target_train_dloader, target_test_dloader = get_office_home_dloader(args.base_path,
                                                                                 args.target_domain,
                                                                                 configs["TrainingConfig"]["batch_size"],
                                                                                args.workers)
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)

        models.append(
            OfficehomeNet(configs["ModelConfig"]["tarbackbone"], bn_momentum=args.bn_momentum,
                             pretrained=args.target_pretrain,
                             data_parallel=args.data_parallel).cuda())
        classifiers.append(
            OfficehomeClassifier(configs["ModelConfig"]["tarbackbone"], 65, args.data_parallel).cuda()
        )
        models.append(
            OfficehomeNet(configs["ModelConfig"]["tarbackbone"], bn_momentum=args.bn_momentum,
                             pretrained=args.target_pretrain,
                             data_parallel=args.data_parallel).cuda())
        classifiers.append(
            OfficehomeClassifier(configs["ModelConfig"]["tarbackbone"], 65, args.data_parallel).cuda()
        )
        domains.remove(args.target_domain)
        args.source_domains = domains
        for domain in domains:
            i += 1
            source_train_dloader, source_test_dloader = get_office_home_dloader(args.base_path, domain,
                                                                                     configs["TrainingConfig"][
                                                                                         "batch_size"], args.workers)
            train_dloaders.append(source_train_dloader)
            test_dloaders.append(source_test_dloader)
            models.append(
                OfficehomeNet(configs["ModelConfig"]["backbone"],args.bn_momentum,
                                 pretrained=configs["ModelConfig"]["pretrained"],
                                 data_parallel=args.data_parallel).cuda())
            classifiers.append(
                OfficehomeClassifier(configs["ModelConfig"]["backbone"], 65, args.data_parallel).cuda()
            )
        num_classes = 65
    elif configs["DataConfig"]["dataset"] == "DomainNet":
        domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        i = 0

        target_train_dloader, target_test_dloader = get_domainnet_dloader(args.base_path,
                                                                          args.target_domain,
                                                                          configs["TrainingConfig"]["batch_size"],
                                                                          args.workers)
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)

        models.append(
            DomainNet(configs["ModelConfig"]["tarbackbone"], args.bn_momentum, args.target_pretrain,
                      args.data_parallel).cuda())
        classifiers.append(DomainNetClassifier(configs["ModelConfig"]["tarbackbone"], 345, args.data_parallel).cuda())

        models.append(
            DomainNet(configs["ModelConfig"]["tarbackbone"], args.bn_momentum, args.target_pretrain,
                      args.data_parallel).cuda())
        classifiers.append(DomainNetClassifier(configs["ModelConfig"]["tarbackbone"], 345, args.data_parallel).cuda())

        domains.remove(args.target_domain)
        args.source_domains = domains
        for domain in domains:
            i += 1
            source_train_dloader, source_test_dloader = get_domainnet_dloader(args.base_path, domain,
                                                                              configs["TrainingConfig"]["batch_size"],
                                                                              args.workers)
            train_dloaders.append(source_train_dloader)
            test_dloaders.append(source_test_dloader)

            models.append(DomainNet(configs["ModelConfig"]["backbone"], args.bn_momentum,
                                    pretrained=configs["ModelConfig"]["pretrained"],
                                    data_parallel=args.data_parallel).cuda())
            classifiers.append(DomainNetClassifier(configs["ModelConfig"]["backbone"], 345, args.data_parallel).cuda())
        num_classes = 345
    else:
        raise NotImplementedError("Dataset {} not implemented".format(configs["DataConfig"]["dataset"]))

    for model in models:
        optimizers.append(
            torch.optim.SGD(model.parameters(), momentum=args.momentum,
                            lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd))
    for classifier in classifiers:
        classifier_optimizers.append(
            torch.optim.SGD(classifier.parameters(), momentum=args.momentum,
                            lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=args.wd))
    # create the optimizer scheduler with cosine annealing schedule
    for optimizer in optimizers:
        optimizer_schedulers.append(
            CosineAnnealingLR(optimizer, configs["TrainingConfig"]["total_epochs"],
                              eta_min=configs["TrainingConfig"]["learning_rate_end"]))
    for classifier_optimizer in classifier_optimizers:
        classifier_optimizer_schedulers.append(
            CosineAnnealingLR(classifier_optimizer, configs["TrainingConfig"]["total_epochs"],
                              eta_min=configs["TrainingConfig"]["learning_rate_end"]))

    writer_log_dir = path.join(args.base_path, configs["DataConfig"]["dataset"], "normal_mixup",
                               "train_time:{}".format(args.train_time) + "_" +
                               args.target_domain + "_" + "_".join(args.source_domains))

    print("create writer in {}".format(writer_log_dir))
    writer = SummaryWriter(log_dir=writer_log_dir)

    # begin train
    print("Begin the {} time's training, Dataset:{}, Source Domains {}, Target Domain {}".format(args.train_time,
                                                                                                 configs[
                                                                                                     "DataConfig"][
                                                                                                     "dataset"],
                                                                                                 args.source_domains,
                                                                                                 args.target_domain))
    # train model
    logger = get_logger(args.dataset+args.target_domain)
    batch_per_epoch, total_epochs = decentralized_training_strategy(
        communication_rounds=args.communication_rounds,
        epoch_samples=configs["TrainingConfig"]["epoch_samples"],
        batch_size=configs["TrainingConfig"]["batch_size"],
        total_epochs=configs["TrainingConfig"]["total_epochs"])
    maxaccA = 0
    maxaccB = 0
    maxaccAB = 0
    communication_rounds = configs["UMDAConfig"]["communication_rounds"]

    # define drop rate schedule
    rate_sche = np.ones(total_epochs) * args.forget_rate
    rate_sche[:(args.warm_target+args.num_gradual)] = np.linspace(0, args.forget_rate, (args.warm_target+args.num_gradual))

    source_num = len(models[1:]) - 1
    ntrain = len(train_dloaders[0].dataset.data_labels)

    Z_s0 = torch.zeros(source_num, ntrain, num_classes).float().cuda()  # intermediate values

    for epoch in range(args.start_epoch, total_epochs):

        logger.info('Start Validation of epoch {}.'.format(epoch))
        logger.info('Epoch:' + str(epoch + 1) + '/' + str(configs["TrainingConfig"]["total_epochs"]))
        if communication_rounds in [0.1,0.2,0.25, 0.5]:
            model_aggregation_frequency = round(1 / communication_rounds)
        else:
            model_aggregation_frequency = 1
        for f in range(model_aggregation_frequency):
            models, classifiers = source_train(logger,rate_sche[epoch], train_dloaders, test_dloaders, args, models,
                                                     classifiers, batch_per_epoch, total_epochs,
                                                     optimizers,
                                                     classifier_optimizers, epoch, writer, num_classes=num_classes,
                                                     source_domains=args.source_domains,
                                                     batch_size=configs["TrainingConfig"]["batch_size"])

        intra_domain0, _, Z_s0 = visual_source(logger, epoch + 1, train_dloaders[0], args, batch_per_epoch, Z_s0,
                                               models[2:], classifiers[2:],
                                               num_classes=num_classes, source_domains=args.source_domains,
                                               batch_size=configs["TrainingConfig"]["batch_size"])  # source free only

        intra_domain_weight = Icvar_p(args, intra_domain0)

        visual_all(logger, epoch+1, train_dloaders[0], args, num_classes, models[2:], classifiers[2:], intra_domain_weight,
                   batch_size=configs["TrainingConfig"]["batch_size"])
        visual_test(logger, epoch+1, train_dloaders[0], args, num_classes, models[2:], classifiers[2:], Z_s0, intra_domain_weight,
                   batch_size=configs["TrainingConfig"]["batch_size"])

        test(args, args.target_domain, args.source_domains, train_dloaders, Z_s0, epoch,intra_domain_weight, writer,num_classes)

        if epoch >= args.warm_target:
            maxaccA, maxaccB, maxaccAB = validate_target(logger, epoch, test_dloaders[0], maxaccA, maxaccB, maxaccAB,
                                                         args, models, classifiers)

            target_train(logger, args, args.target_domain, rate_sche[epoch], train_dloaders[0], test_dloaders[0], Z_s0,
                         maxaccA, models[0], classifiers[0],
                         batch_per_epoch, epoch, optimizers[0], classifier_optimizers[0],
                         models[1], classifiers[1], models, classifiers,
                         writer, num_classes, intra_domain_weight,
                         total_epochs=total_epochs,
                         batch_size=configs["TrainingConfig"]["batch_size"],
                         epoch_samples=configs["TrainingConfig"]["epoch_samples"])

            target_train(logger, args, args.target_domain, rate_sche[epoch], train_dloaders[0], test_dloaders[0], Z_s0,
                         maxaccB, models[1], classifiers[1],
                         batch_per_epoch, epoch, optimizers[1], classifier_optimizers[1],
                         models[0], classifiers[0], models, classifiers,
                         writer, num_classes, intra_domain_weight,
                         total_epochs=total_epochs,
                         batch_size=configs["TrainingConfig"]["batch_size"],
                         epoch_samples=configs["TrainingConfig"]["epoch_samples"])

def save_checkpoint(state, filename):
    filefolder = "{}/{}/parameter/train_time:{}".format(opt.base_path, configs["DataConfig"]["dataset"],
                                                        opt.train_time)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))

if __name__ == "__main__":
    main()
