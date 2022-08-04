import torch
import torch.nn as nn
import numpy as np
from lib.utils.federated_utils import *
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from lib.utils.loss import Icvar_p, loss_teaching

def source_train(logger, rate_schedule, train_dloader_list, test_dloaders_list
                 , args, model_list, classifier_list, batch_per_epoch, total_epochs, optimizer_list,
                 classifier_optimizer_list, epoch, writer,
                 num_classes, source_domains, batch_size):

    task_criterion = nn.CrossEntropyLoss().cuda()
    for model in model_list:
        model.train()
    for classifier in classifier_list:
        classifier.train()
    # If communication rounds <1,
    # then we perform parameter aggregation after (1/communication_rounds) epochs
    # If communication rounds >=1:
    # then we extend the training epochs and use fewer samples in each epoch.

    # Train model locally on source domains
    n = 0
    for train_dloader, model, classifier, optimizer, classifier_optimizer in zip(train_dloader_list[2:],
                                                                                 model_list[2:],
                                                                                 classifier_list[2:],
                                                                                 optimizer_list[2:],
                                                                                 classifier_optimizer_list[2:]):
        n += 1

        for i, (image_s, label_s, _) in enumerate(train_dloader):

            if i >= batch_per_epoch:
                break
            image_s = image_s.cuda()
            label_s = label_s.long().cuda()
            # reset grad
            optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            # each source domain do optimize
            feature_s = model(image_s)
            output_s = classifier(feature_s)

            task_loss_s = task_criterion(output_s, label_s)  # .requires_grad_()
            task_loss_s.backward()
            optimizer.step()
            classifier_optimizer.step()

    if (epoch + 1) % 40 == 0:
        torch.save(model.state_dict(), '%s/models/model_%s%d.pth' % (args.outf, args.target_domain, epoch))
        torch.save(classifier.state_dict(), '%s/models/classifier_%s%d.pth' % (args.outf, args.target_domain, epoch))

    return model_list, classifier_list

# function
def target_train(logger, args, target_domain, rate_schedule, train_dloader, test_dloader, Z_s,
                 maxaccA, model, classifier, batch_per_epoch, epoch, optimizer, classifier_optimizer,
                 model_ind, classifier_ind, model_list, classifier_list,
                 writer, num_classes, intra_domain_weight,
                 total_epochs, batch_size,epoch_samples):

    model.train()
    classifier.train()

    # If communication rounds <1,
    # then we perform parameter aggregation after (1/communication_rounds) epochs
    # If communication rounds >=1:
    # then we extend the training epochs and use fewer samples in each epoch.
    # Train model locally on source domains
    source_domain_num = len(model_list[2:])

    if epoch <= round(total_epochs/2):#total_epochs
        alpha = 0.0
    else:
        alpha = (epoch*2-total_epochs)/total_epochs

    totaldt = 0

    for i, (image_t, gray_label, index) in enumerate(train_dloader):

        if i >= batch_per_epoch:
            break
        image_t = image_t.cuda()
        output_source = torch.zeros([source_domain_num, image_t.size(0), num_classes], dtype=torch.float)

        with torch.no_grad():
            for source_idx in range(source_domain_num):
                output_source[source_idx] = intra_domain_weight[source_idx] * Z_s[source_idx, index, :]

            output_allsoft = torch.sum(output_source, 0)
            _, before_decorrect = torch.max(output_allsoft.data, 1)

            output_target = classifier(model(image_t))
            output_target = torch.softmax(output_target, dim=1)
            output_allsoft = (1 - alpha) * output_allsoft + alpha * output_target.cpu()

            output_conf, output_class0 = torch.max(output_allsoft.data, 1)
            _, output_target0 = torch.max(output_target.data, 1)

            output_class = output_class0.cuda()
            output_class = torch.zeros(output_class.size(0), num_classes).cuda()\
                .scatter_(1, output_class.view(-1, 1), 1)
            output_ind = classifier_ind(model_ind(image_t))

            totaldt += gray_label.size(0)

        output_t = classifier(model(image_t))
        ind_R_update, ind_update, _ = loss_teaching(output_t, output_ind, output_allsoft.cuda(), rate_schedule, alpha)

        with torch.no_grad():

            # mixmatch
            lam0 = np.random.beta(args.beta, args.beta)
            lam = max(lam0, 1 - lam0)

            batch = image_t[ind_R_update].size(0)
            index0 = torch.randperm(batch).cuda()
            mixed_image = lam * image_t[ind_R_update] + (1 - lam) * image_t[ind_R_update[index0], :]
            mixed_target = lam * output_class[ind_R_update] + (1 - lam) * output_class[ind_R_update[index0], :]

        # mixmatch
        output_mix = classifier(model(mixed_image))
        output_mix = torch.log_softmax(output_mix, dim=1)
        task_loss_mix = torch.mean(torch.sum(-1 * mixed_target.cuda()* output_mix.cuda(), dim=1))# .cuda().cuda()

        optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        loss0 = args.alpha_mix * task_loss_mix
        # each source domain do optimize#
        loss0.backward()
        optimizer.step()
        classifier_optimizer.step

def validate_target(logger, epoch, test_dloader, maxaccA, maxaccB, maxaccAB, args, model_list, classifier_list):
    correcttA = 0
    correcttB = 0
    correcttall = 0
    writer = SummaryWriter('curve%s_%s' % (args.dataset, args.target_domain))
    totaldt = 0

    for i, (image_t, label_t, _) in enumerate(test_dloader):

        image_t = torch.tensor(image_t)
        label_t = torch.Tensor(label_t.float())
        # each source domain do optimize .cuda()

        with torch.no_grad():
            output_tA = classifier_list[0](model_list[0](image_t))
            output_tB = classifier_list[1](model_list[1](image_t))

            out__soft_tA = torch.softmax(output_tA, dim=1)
            out__soft_tB = torch.softmax(output_tB, dim=1)
            out__soft = out__soft_tA + out__soft_tB

            _, predictedA = torch.max(out__soft_tA.data, 1)
            _, predictedB = torch.max(out__soft_tB.data, 1)
            _, predictedall = torch.max(out__soft.data, 1)

            correcttA += ((predictedA == label_t.cuda().long()).sum())
            correcttB += ((predictedB == label_t.cuda().long()).sum())
            correcttall += ((predictedall == label_t.cuda().long()).sum())

        totaldt += label_t.size(0)

    val_acctA = 100 * float(correcttA) / totaldt
    val_acctB = 100 * float(correcttB) / totaldt
    val_accAB = 100 * float(correcttall) / totaldt

    logger.info(' Epoch/%d, alpha_mix:%4f acctA: %4f acctB: %4f acctAB: %4f %%' % (
        epoch, args.alpha_mix, val_acctA, val_acctB, val_accAB))

    writer.add_scalars("Train/acctA", {'Acc4': val_acctA}, epoch)
    writer.add_scalars("Train/acctB", {'Acc4': val_acctB}, epoch)
    maxaccA = max(maxaccA, val_acctA)
    maxaccB = max(maxaccB, val_acctB)
    maxaccAB = max(maxaccAB, val_accAB)

    logger.info(' Epoch/%d, forgetrate:%4f alpha_mix:%4f maxacctA: %4f maxacctB: %4f maxacctAB: %4f %%' % (
        epoch, args.forget_rate, args.alpha_mix, maxaccA, maxaccB, maxaccAB))

    return maxaccA, maxaccB, maxaccAB

# Validation function
def visual_source(logger, epoch, train_dloader, args, batch_per_epoch, Z_s, models, classifiers, num_classes,
                  source_domains,batch_size):

    writer = SummaryWriter('curve%s_%s' % (args.dataset, args.target_domain))

    data_label = torch.LongTensor([])
    data_label = data_label.cuda()
    data_label = Variable(data_label)
    class_icvar = []
    out_put = []
    val_acct_all = []
    ind = []
    # Testing the model2
    with torch.no_grad():
        current_domain_index = 0
        n = 0
        source_num = len(models)

        if args.dataset == "AmazonReview":
            ntrain_b = len(train_dloader.dataset.labels)
        else:
            ntrain_b = len(train_dloader.dataset.data_labels)

        z_s = torch.zeros(source_num, ntrain_b, num_classes).float().cuda()  # temporal outputs
        outputs_s = torch.zeros(source_num, ntrain_b, num_classes).float().cuda()  # current outputs

        for model, classifier in zip(models, classifiers):

            totaldt = 0
            correctt = 0
            label = torch.zeros(batch_size, num_classes)
            out_soft = torch.zeros(batch_size, num_classes)

            for i, (image_t, label_t, index) in enumerate(train_dloader):
                #if i >= batch_per_epoch:
                    #break

                image_t = torch.tensor(image_t)
                label_t = torch.Tensor(label_t.float())
                index = torch.tensor(index)

                # each source domain do optimize .cuda()
                feature_t = model(image_t)
                output_t = classifier(feature_t)
                out__soft_t = torch.softmax(output_t, dim=1)
                _, predicted = torch.max(out__soft_t.data, 1)
                correctt += ((predicted == label_t.cuda().long()).sum())
                totaldt += label_t.size(0)

                if i == 0:
                    label = predicted.cuda()
                    out_soft = out__soft_t
                    ind = index

                else:
                    out_soft = torch.cat((out_soft.cuda(), out__soft_t), 0)
                    label = torch.cat((label, predicted), 0)
                    ind = torch.cat((ind, index), 0)

            out_put.append(out_soft)
            out_soft = out_soft[0:totaldt]
            out_soft = out_soft.view(totaldt, -1)
            outputs_s[n, ind, :] = out_soft  # current outputs

            # update temporal ensemble
            Z_s[n, :, :] = args.alpha_z * Z_s[n, :, :] + (1. - args.alpha_z) * outputs_s[n, :, :]
            z_s[n, :, :] = Z_s[n, :, :] * (1. / (1. - args.alpha_z ** (epoch + 1)))
            out_soft = z_s[n, ind, :]

            data_label.resize_(totaldt).copy_(label)
            outclass = torch.cuda.LongTensor(data_label.unsqueeze(1))
            class_one_hot_t = torch.zeros(totaldt, num_classes).scatter_(1, outclass.cpu(), 1).cuda()

            mul_soft_t = 0
            mul_hot_t = 0
            mul_pos_t = 0
            mul_neg_t = 0

            for i in range(totaldt):
                mul_soft_t += out_soft[i,]
                mul_pos_t += out_soft[i,] * class_one_hot_t[i,]
                mul_neg_t += out_soft[i,] * (1 - class_one_hot_t[i,])
                mul_hot_t += class_one_hot_t[i,]

            mul_soft_t = torch.div(mul_soft_t, totaldt)
            mul_pos_t = torch.div(mul_pos_t, mul_hot_t + 0.001)
            mul_neg_t = torch.div(mul_neg_t, (totaldt - mul_hot_t + 0.001))

            OTSU_all1 = torch.div(torch.sum((mul_hot_t * (mul_pos_t - mul_soft_t) * (mul_pos_t - mul_soft_t))), totaldt)
            OTSU_all0 = torch.div(
                torch.sum(((totaldt - mul_hot_t) * (mul_neg_t - mul_soft_t) * (mul_neg_t - mul_soft_t))), totaldt)

            OTSU = (OTSU_all1 + OTSU_all0)

            val_acct = 100 * float(correctt) / totaldt
            val_acct_all.append(val_acct)

            current_domain_index += 1
            class_icvar.append(OTSU)
            n += 1

        intra_domain_weight = Icvar_p(args, class_icvar)

        for i in range(len(source_domains)):
            writer.add_scalars("Train/{}".format(source_domains[i]),
                               {'Acc': val_acct_all[i], 'Weight': intra_domain_weight[i], 'Intra_Domain': class_icvar[i]}, epoch)

            logger.info(' Epoch/%d, source_%s val_acct: %4f icvar: %4f %%' % (
                epoch, source_domains[i], val_acct_all[i], class_icvar[i]))

    return class_icvar, Z_s, z_s

def visual_all(logger, epoch, train_dloader, args, num_classes, models, classifiers, domain_weight,
               batch_size):
    writer = SummaryWriter('curve%s_%s' % (args.dataset, args.target_domain))

    # Testing the model2
    with torch.no_grad():
        totaldt = 0
        correctt = 0
        correct_all = 0
        outsoft_mean = 0
        correct_mean_all = 0
        label = []
        soft_max_all = 0

        for i, (image_t, label_t, _) in enumerate(train_dloader):
            n = 0
            soft = 0
            soft_max = 0
            m = 0
            for model, classifier in zip(models, classifiers):

                image_t = torch.tensor(image_t)
                label_t = torch.Tensor(label_t.float())
                # each source domain do optimize .cuda()
                feature_t = model(image_t)
                output_t = classifier(feature_t)
                out__soft_t0 = torch.softmax(output_t, dim=1)
                _, predicted = torch.max(out__soft_t0.data, 1)
                out__soft_t1 = out__soft_t0 * domain_weight[n]
                n += 1
                out__soft_t = torch.unsqueeze(out__soft_t0, axis=2)

                if m == 0:
                    soft_max = out__soft_t
                    outsoft_mean = out__soft_t
                    soft = out__soft_t1
                else:
                    soft_max = torch.cat((soft_max, out__soft_t), dim=2)
                    outsoft_mean += out__soft_t
                    soft += out__soft_t1

                m += 1

            if i == 0:
                label = label_t
                soft_max_all = soft_max
            else:
                label = torch.cat((label, label_t), 0)
                soft_max_all = torch.cat((soft_max_all, soft_max), 0)

            _, predicted_all = torch.max(soft.data, 1)
            _, predicted_mean_all = torch.max(outsoft_mean.data, 1)

            correct_all += ((predicted_all == label_t.cuda().long()).sum())
            correct_mean_all += ((predicted_mean_all == label_t.cuda().long()).sum())

            correctt += ((predicted == label_t.cuda().long()).sum())
            totaldt += label_t.size(0)

        soft_max_all, _ = soft_max_all.max(axis=2)
        _, predicted_max = torch.max(soft_max_all.data, 1)
        correct_max_all = ((predicted_max == label.cuda().long()).sum())

        acc_all = 100 * float(correct_all) / totaldt
        acc_max_all = 100 * float(correct_max_all) / totaldt

        writer.add_scalars("Train/acc_all",{'Acc_all': acc_all, 'Acc_max_all': acc_max_all}, epoch)
        logger.info(' Epoch/%d,acc_all: %4f %%' % (epoch, acc_all))

def visual_test(logger, epoch, train_dloader, args, num_classes, models, classifiers, Z_s, domain_weight,
                batch_size):
    writer = SummaryWriter('curve%s_%s' % (args.dataset, args.target_domain))

    source_domain_num = len(models)

    # Testing the model2
    with torch.no_grad():
        totaldt = 0
        correct_all = 0

        for i, (image_t, label_t, index) in enumerate(train_dloader):
            output_source = torch.zeros([source_domain_num, image_t.size(0), num_classes], dtype=torch.float)

            for source_idx in range(source_domain_num):
                output_source[source_idx] = domain_weight[source_idx] * Z_s[source_idx, index, :]

            output_allsoft = torch.sum(output_source, 0)
            _, predicted_all = torch.max(output_allsoft.data, 1)
            correct_all += ((predicted_all == label_t.long()).sum())
            totaldt += label_t.size(0)

        acc_all = 100 * float(correct_all) / totaldt

        writer.add_scalars("Train/acc_all", {'Acc_test': acc_all}, epoch)
        logger.info(' Epoch/%d,acc_test: %4f %%' % (epoch, acc_all))

def test(args, target_domain, source_domains, train_dloaders_list,
         Z_s, epoch, intra_domain_weight, writer, num_classes,top_5_accuracy=True):

    writer = SummaryWriter('curve%s_%s' % (args.dataset, args.target_domain))

    source_domain_num = len(train_dloaders_list[1:]) - 1
    # calculate loss, accuracy for target domain
    tmp_score = []
    tmp_label = []
    train_dloader_t = train_dloaders_list[0]

    for _, (image_t, label_t, index) in enumerate(train_dloader_t):
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()
        output_source = torch.zeros([source_domain_num, image_t.size(0), num_classes], dtype=torch.float)

        with torch.no_grad():
            for source_idx in range(source_domain_num):
                output_source[source_idx] = intra_domain_weight[source_idx] * Z_s[source_idx, index, :]

            output_allsoft = torch.sum(output_source, 0)
            output_allsoft = torch.softmax(output_allsoft, dim=1)
            tmp_label.append(label_t)
            tmp_score.append(output_allsoft)
        # turn label into one-hot code

    tmp_score = torch.cat(tmp_score, dim=0).detach()
    tmp_label = torch.cat(tmp_label, dim=0).detach()
    tmp_label = tmp_label.view(-1, 1)

    _, y_pred1 = torch.topk(tmp_score, k=1, dim=-1)
    _, y_pred5 = torch.topk(tmp_score, k=5, dim=-1)
    _, y_pred3 = torch.topk(tmp_score, k=3, dim=-1)

    top_1_accuracy_t = 100 *float(torch.sum(tmp_label.cuda() == y_pred1.cuda()).item()) / tmp_label.size(0)
    writer.add_scalar("Train/Top_accuracy/top1",top_1_accuracy_t, epoch)

    top_3_accuracy_t = 100 *float(torch.sum(tmp_label.cuda() == y_pred3.cuda()).item()) / tmp_label.size(0)
    writer.add_scalar("Train/Top_accuracy/top3",top_3_accuracy_t, epoch)

    top_5_accuracy_t = 100 *float(torch.sum(tmp_label.cuda() == y_pred5.cuda()).item()) / tmp_label.size(0)
    writer.add_scalar("Train/Top_accuracy/top5",top_5_accuracy_t, epoch)

    print("Target Domain {} Accuracy Top1 :{:.3f} Top3 :{:.3f} Top5:{:.3f}".format(target_domain,
                         top_1_accuracy_t, top_3_accuracy_t, top_5_accuracy_t))

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()
