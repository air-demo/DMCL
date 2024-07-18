import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
import torch.nn.functional as F
from metric import inference_new
from metric import evaluate
import scipy.io


Dataname = 'cifar10'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=2.0)
parser.add_argument("--temperature_l", default=0.1)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=150)
parser.add_argument("--con_epochs", default=150)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--threshold", type=float, default=0.8)
parser.add_argument("--seed", type=int, default=15)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset, dims, view, data_size, class_num = load_data(args.dataset)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(args.seed)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

def make_pseudo_label(model, device):
    model.eval()
    scaler = MinMaxScaler()
    for step, (xs, y, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            hs, _, _, _ = model.forward(xs)
        for v in range(view):
            hs[v] = hs[v].cpu().detach().numpy()
            hs[v] = scaler.fit_transform(hs[v])

    kmeans = KMeans(n_clusters=class_num, n_init=100)
    new_pseudo_label = []
    for v in range(view):
        Pseudo_label = kmeans.fit_predict(hs[v])
        Pseudo_label = Pseudo_label.reshape(256, 1)
        Pseudo_label = torch.from_numpy(Pseudo_label)
        new_pseudo_label.append(Pseudo_label)

    return new_pseudo_label

def match(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    new_y = torch.from_numpy(new_y).long().to(device)
    new_y = new_y.view(new_y.size()[0])
    return new_y


#预训练函数
def pretrain(epoch):
    tot_loss=0.
    for batch_idx,(x,_,_)in enumerate(data_loader):
        for v in range(view):
            x[v]=x[v].to(device)
        optimizer.zero_grad()
        _,_,decoder_feature,_=model(x)
        loss_list=[]
        for v in range(view):
            loss_list.append(F.mse_loss(x[v],decoder_feature[v]))
        loss=sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss+=loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

def train(epoch, p_sample, adaptive_weight,new_pseudo_label,temperature_f):
    cross_entropy = torch.nn.CrossEntropyLoss()
    tot_loss=0.
    all_results = []
    for batch_idx,(x,_,_) in enumerate(data_loader):
        for v in range(view):
            x[v]=x[v].to(device)
        optimizer.zero_grad()

        #
        hs, qs, xrs, zs=model.forward(x)

        loss_list = []
       
    
        for v in range(view):
            for w in range(v+1, view):
                # similarity of the samples in any two views
                sim = torch.exp(torch.mm(hs[v], hs[w].t()))
                sim_probs = sim / sim.sum(1, keepdim=True)

                # pseudo matrix
                Q = torch.mm(qs[v], qs[w].t())
                Q.fill_diagonal_(1)
                pos_mask = (Q >= args.threshold).float()
                Q = Q * pos_mask
                Q = Q / Q.sum(1, keepdims=True)

                loss_contrast_local = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
                loss_contrast_local = loss_contrast_local.mean()

                loss_list.append(temperature_f*loss_contrast_local)
                loss_list.append(criterion.forward_label(qs[v], qs[w]))
            loss_list.append(F.mse_loss(x[v], xrs[v]))

        for v in range(view):
            p = new_pseudo_label[v].numpy().T[0]
            with torch.no_grad():
                q = qs[v].detach().cpu()
                q = torch.argmax(q, dim=1).numpy()
                p_hat = match(p, q)
            loss_list.append(cross_entropy(qs[v], p_hat))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
    print("len(data_loader)",len(data_loader))
    total_pred, pred_vectors, high_level_vectors, labels_vector, low_level_vectors  = inference_new(data_loader, model, device, view, len(data_loader)*256, p_sample,adaptive_weight)
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    nmi, ari, acc, pur = evaluate(labels_vector, total_pred)
    print('ACC = {:.4f} NMI = {:.4f} PUR={:.4f} ARI = {:.4f}'.format(acc, nmi, pur, ari))

    # Save results to a dictionary
    results = {
        'ACC': acc,
        'NMI': nmi,
        'PUR': pur,
        'ARI': ari,
        'tot_loss': tot_loss,
    }

    # Append results dictionary to the list
    all_results.append(results)

    # Return all_results if needed
    return all_results

#搭建模型
model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)


# 视图采样概率
p_sample = np.ones(view)
weight_history = []
p_sample = p_sample / sum(p_sample)
p_sample = torch.FloatTensor(p_sample).cuda()


# 初始化自适应权重
adaptive_weight = np.ones(view)
adaptive_weight = adaptive_weight / sum(adaptive_weight)
adaptive_weight = torch.FloatTensor(adaptive_weight).cuda()
adaptive_weight = adaptive_weight.unsqueeze(1)


# 开始训练
epoch = 1
#预训练阶段
while epoch <= args.mse_epochs:
    pretrain(epoch)
    epoch += 1
new_pseudo_label = make_pseudo_label(model, device)
#训练阶段
while epoch <= args.mse_epochs + args.con_epochs:
    all_results=train(epoch, p_sample,adaptive_weight,new_pseudo_label,args.temperature_f)
    filename = 'cifar100_results.mat'
    scipy.io.savemat(filename, {'results': all_results})

 
    #验证阶段
    if epoch == args.mse_epochs + args.con_epochs:
        valid(model, device, dataset, view, data_size, class_num, p_sample,adaptive_weight)
    epoch += 1

    