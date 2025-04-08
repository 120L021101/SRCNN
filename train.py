import argparse
import os
import copy
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import torch.autograd as autograd
from tqdm import tqdm

from models import SRCNN, TransformerSRCNN, Discriminator, VGGLoss
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr


def gradient_penalty(D, real_data, fake_data, device):
    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    G = SRCNN(num_channels=1).to(device)
    #G = TransformerSRCNN(num_channels=1, dim=64, num_heads=8, num_blocks=6, ff_dim=256).to(device)
    D = Discriminator(num_channels=1, input_size=33).to(device)
    criterion_pix = nn.MSELoss()
    criterion_vgg = VGGLoss().to(device)
    # TODO: Implement WGAN
    criterion_gan = nn.BCELoss()
    optimizer_G = optim.Adam([
        {'params': G.conv1.parameters()},
        {'params': G.conv2.parameters()},
        {'params': G.conv3.parameters(), 'lr': args.lr*0.1}
    ], lr=args.lr)
    # optimizer_G = optim.Adam([
    #     {'params': G.patch_embed.parameters()},
    #     {'params': G.blocks.parameters()},
    #     {'params': G.final_conv.parameters(), 'lr': args.lr * 0.1}
    # ], lr=args.lr)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(G.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    dloss_list, gloss_list, psnr_list = [], [], []

    for epoch in range(args.num_epochs):
        #model.train()
        G.train()
        D.train()
        epoch_dloss = AverageMeter()
        epoch_gloss = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                #preds = model(inputs)
                # TODO: Maybe do label smoothing?
                real_labels = torch.ones((inputs.size(0), 1), device=device)
                #real_labels = torch.full((inputs.size(0), 1), 0.9).to(device)
                fake_labels = torch.zeros((inputs.size(0), 1), device=device)

                # loss = criterion(preds, labels)
                #
                # epoch_losses.update(loss.item(), len(inputs))
                #
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                #
                # t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                # t.update(len(inputs))

                lambda_gp = 10
                optimizer_D.zero_grad()
                real_output = D(labels)
                fake_imgs = G(inputs).detach()
                fake_outputs = D(fake_imgs)

                d_loss_real = criterion_gan(real_output, real_labels)
                d_loss_fake = criterion_gan(fake_outputs, fake_labels)
                d_loss = (d_loss_real + d_loss_fake) * 0.5

                # WGAN
                # gp = gradient_penalty(D, labels.data, fake_imgs.data, device)
                # d_loss = -torch.mean(real_output) + torch.mean(fake_outputs) + gp * lambda_gp

                d_loss.backward()
                optimizer_D.step()

                optimizer_G.zero_grad()
                fake_imgs = G(inputs)
                pred_outputs = D(fake_imgs)
                loss_g_gan = criterion_gan(pred_outputs, real_labels)
                # WGAN
                #loss_g_gan = -torch.mean(pred_outputs)
                loss_g_pix = criterion_pix(fake_imgs, labels)
                #loss_g_vgg = criterion_vgg(fake_imgs, labels)
                # TODO: Tune the weights !!!Direction may be counter-intuitive!!!
                g_loss = loss_g_pix*0.001 + loss_g_gan# + loss_g_vgg * 0.007
                g_loss.backward()
                optimizer_G.step()

                epoch_dloss.update(d_loss.item(), inputs.size(0))
                epoch_gloss.update(g_loss.item(), inputs.size(0))
                t.set_postfix(D_loss=f'{epoch_dloss.avg:.4f}', G_loss=f'{epoch_gloss.avg:.4f}')
                t.update(inputs.size(0))

        #torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        G.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = G(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        dloss_list.append(float(epoch_dloss.avg))
        gloss_list.append(float(epoch_gloss.avg))
        psnr_list.append(float(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(G.state_dict())

        torch.save(G.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))

    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax2 = ax1.twinx()
    ax1.plot(range(args.num_epochs), dloss_list, label='D Loss', marker='o', markevery=9)
    ax2.plot(range(args.num_epochs), gloss_list, label='G Loss', marker='x', markevery=9, color='red')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("D Loss")
    ax2.set_ylabel("G Loss")
    ax1.set_title("Adversarial Losses")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(range(args.num_epochs), psnr_list, marker='o', markevery=9, color='green')
    plt.title("Eval PSNR per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")

    plt.tight_layout()
    plot_path = os.path.join(args.outputs_dir, 'loss_psnr.png')
    plt.savefig(plot_path)
    plt.close()
