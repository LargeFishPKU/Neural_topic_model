import torch.nn as nn
from torch.nn import functional as F
# from pykp.masked_loss import masked_cross_entropy
# from utils.statistics import LossStatistics
from utils.time_log import time_since, convert_time2str
# from evaluate import evaluate_loss
import time
import math
import logging
import torch
import sys
import os

EPS = 1e-6

# functions abot fix/unfix model
def fix_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfix_model(model):
    for param in model.parameters():
        param.requires_grad = True

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def l1_penalty(para):
    return nn.L1Loss()(para, torch.zeros_like(para))


def check_sparsity(para, sparsity_threshold=1e-3):
    num_weights = para.shape[0] * para.shape[1]
    num_zero = (para.abs() < sparsity_threshold).sum().float()
    return num_zero / float(num_weights)


def update_l1(cur_l1, cur_sparsity, sparsity_target):
    diff = sparsity_target - cur_sparsity
    cur_l1.mul_(2.0 ** diff)


def train_ntm_one_epoch(model, dataloader, optimizer, opt, epoch):
    model.train()
    train_loss = 0
    for batch_idx, data_bow in enumerate(dataloader):
        data_bow = data_bow.to(opt.device)
        # normalize data
        data_bow_norm = F.normalize(data_bow)
        optimizer.zero_grad()
        _, _, recon_batch, mu, logvar = model(data_bow_norm)
        loss = loss_function(recon_batch, data_bow, mu, logvar)
        loss = loss + model.l1_strength * l1_penalty(model.fcd1.weight)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_bow), len(dataloader.dataset),
                       100. * batch_idx / len(dataloader),
                       loss.item() / len(data_bow)))

    logging.info('====>Train epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(dataloader.dataset)))
    sparsity = check_sparsity(model.fcd1.weight.data)
    logging.info("Overall sparsity = %.3f, l1 strength = %.5f" % (sparsity, model.l1_strength))
    logging.info("Target sparsity = %.3f" % opt.target_sparsity)
    update_l1(model.l1_strength, sparsity, opt.target_sparsity)
    return sparsity


def test_ntm_one_epoch(model, dataloader, opt, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data_bow in enumerate(dataloader):
            data_bow = data_bow.to(opt.device)
            data_bow_norm = F.normalize(data_bow)

            _, _, recon_batch, mu, logvar = model(data_bow_norm)
            test_loss += loss_function(recon_batch, data_bow, mu, logvar).item()

    avg_loss = test_loss / len(dataloader.dataset)
    logging.info('====> Test epoch: {} Average loss:  {:.4f}'.format(epoch, avg_loss))
    return avg_loss


def train_model(ntm_model, optimizer_ntm, bow_dictionary, train_bow_loader, valid_bow_loader, opt):
    logging.info('======================  Start Training  =========================')

    if opt.only_train_ntm or (opt.use_topic_represent and not opt.load_pretrain_ntm):
        print("\nWarming up ntm for %d epochs" % opt.ntm_warm_up_epochs)
        for epoch in range(1, opt.ntm_warm_up_epochs + 1):
            sparsity = train_ntm_one_epoch(ntm_model, train_bow_loader, optimizer_ntm, opt, epoch)
            val_loss = test_ntm_one_epoch(ntm_model, valid_bow_loader, opt, epoch)
            if epoch % 10 == 0:
                ntm_model.print_topic_words(bow_dictionary, os.path.join(opt.model_path, 'topwords_e%d.txt' % epoch))
                best_ntm_model_path = os.path.join(opt.model_path, 'e%d.val_loss=%.3f.sparsity=%.3f.ntm_model' %
                                                   (epoch, val_loss, sparsity))
                logging.info("\nSaving warm up ntm model into %s" % best_ntm_model_path)
                torch.save(ntm_model.state_dict(), open(best_ntm_model_path, 'wb'))
    elif opt.use_topic_represent:
        print("Loading ntm model from %s" % opt.check_pt_ntm_model_path)
        ntm_model.load_state_dict(torch.load(opt.check_pt_ntm_model_path))

    return
