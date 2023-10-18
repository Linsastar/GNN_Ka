import os

root_path = os.path.join(os.path.join(os.getcwd(), '..'))
os.chdir(root_path)

import sys
import csv

sys.path.append(root_path)

import numpy as np
import torch.nn.functional as F
from model.layers import GAT_gate
from torch import nn
from data_processing.Prepare_Input import Prepare_Input
import torch
from data_processing.collate_fn import collate_fn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from data_processing.Extract_Interface import Extract_Interface
from rdkit.Chem.rdmolfiles import MolFromPDBFile
from data_processing.Feature_Processing import get_atom_feature
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from scipy.spatial import distance_matrix
from torch.utils.tensorboard import SummaryWriter

N_atom_features = 28


class Pka_Model(nn.Module):
    def __init__(self, n_graph_layer=4, d_graph_layer=256, n_FC_layer=4, d_FC_layer=128, dropout_rate=0.3):
        # --n_graph_layer：GNN层的数量，默认为4。
        # --d_graph_layer：GNN层的维度，默认为256。
        # --n_FC_layer：全连接层的数量，默认为4。
        # --d_FC_layer：全连接层的维度，默认为128。
        # --initial_mu：mu的初始值，默认为0.0。
        # --initial_dev：dev的初始值，默认为1.0。
        # --dropout_rate：dropout率，默认为0.3。

        super(Pka_Model, self).__init__()
        self.n_graph_layer = n_graph_layer
        self.d_graph_layer = d_graph_layer
        self.n_FC_layer = n_FC_layer
        self.d_FC_layer = d_FC_layer
        self.dropout_rate = dropout_rate

        self.layers1 = [d_graph_layer for i in range(n_graph_layer + 1)]
        self.gconv1 = nn.ModuleList \
            ([GAT_gate(self.layers1[i], self.layers1[i + 1]) for i in range(len(self.layers1) - 1)])

        self.FC = nn.ModuleList([nn.Linear(self.layers1[-1], d_FC_layer) if i == 0 else
                                 nn.Linear(d_FC_layer, 1) if i == n_FC_layer - 1 else
                                 nn.Linear(d_FC_layer, d_FC_layer) for i in range(n_FC_layer)])

        self.mu = nn.Parameter(torch.Tensor([0.0]).float())
        self.dev = nn.Parameter(torch.Tensor([1.0]).float())
        self.embede = nn.Linear(2 * N_atom_features, d_graph_layer, bias=False)


    def forward(self, x):
        x = self.embede(x)
        for layer in self.gconv1:
            x = layer(x)
        x = self.FC[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        for i in range(1, len(self.FC) - 1):
            x = self.FC[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.FC[-1](x)
        return x


def get_pka_from_csv(csv_file, pdb_id, chain, res_id):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['PDB ID'] == pdb_id and row['Chain'] == chain and row['Res ID'] == res_id:
                return float(row['model pKa'])  # 假设 'Cal pKa' 是你要提取的pKa列
        print(f"No pKa value found for PDB ID: {pdb_id}, Chain: {chain}, Res ID: {res_id}.")
        return None


class Single_Dataset(Dataset):

    def __init__(self, list_npz):
        self.listfiles = list_npz

    def __getitem__(self, idx):

        file_path = self.listfiles[idx]
        # print(file_path)
        file_name = os.path.basename(file_path)
        # print(file_name)

        protein_name = file_name.split('_')[0]
        aminoacid_name = file_name.split('_')[1]
        chain_name = file_name.split('_')[2]
        resid = file_name.split('_')[3]
        new_resid = resid.split('.')[0]

        # 确保文件路径存在
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
        # 尝试读取文件
        try:
            data = np.load(file_path)
            # print("File loaded successfully.")
        except FileNotFoundError:
            print(f"File not found at {file_path}.")
            return None, None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None
        # 假设你要处理的CSV文件名为 'pka_values.csv'
        csv_file = 'D:\GNN\GNN_DOVE\model\PHMD549.csv'
        pdb_id = protein_name
        chain = chain_name
        res_id = new_resid  # 用你想要查找的氨基酸残基ID替换
        pka_value = get_pka_from_csv(csv_file, pdb_id, chain, res_id)

        # if pka_value is not None:
        #     print(f"The pKa value for PDB ID: {pdb_id}, Chain: {chain}, Res ID: {res_id} is: {pka_value}")
        pka = pka_value

        return data, pka

    def __len__(self):
        return len(self.listfiles)


def prepare_data(dir_path, save_dir):
    file_namelist = os.listdir(dir_path)
    for file_name in file_namelist:
        save_name = file_name.split('.')[0] + '.npz'
        file_path = os.path.join(dir_path, file_name)
        save_path = os.path.join(save_dir, save_name)
        Prepare_Input(structure_path=file_path, save_path=save_path)
    # print(file_namelist)


def Prepare_Input(structure_path, save_path):
    root_path = os.path.split(structure_path)[0]
    receptor_path, ligand_path = Extract_Interface(structure_path)
    receptor_mol = MolFromPDBFile(receptor_path, sanitize=False)
    ligand_mol = MolFromPDBFile(ligand_path, sanitize=False)
    receptor_count = receptor_mol.GetNumAtoms()
    ligand_count = ligand_mol.GetNumAtoms()
    receptor_feature = get_atom_feature(receptor_mol, is_ligand=False)
    ligand_feature = get_atom_feature(ligand_mol, is_ligand=True)
    c1 = receptor_mol.GetConformers()[0]
    d1 = np.array(c1.GetPositions())
    adj1 = GetAdjacencyMatrix(receptor_mol) + np.eye(receptor_count)
    c2 = ligand_mol.GetConformers()[0]
    d2 = np.array(c2.GetPositions())
    adj2 = GetAdjacencyMatrix(ligand_mol) + np.eye(ligand_count)
    H = np.concatenate([receptor_feature, ligand_feature], 0)
    agg_adj1 = np.zeros((receptor_count + ligand_count, receptor_count + ligand_count))
    agg_adj1[:receptor_count, :receptor_count] = adj1
    agg_adj1[receptor_count:, receptor_count:] = adj2
    dm = distance_matrix(d1, d2)
    agg_adj2 = np.copy(agg_adj1)
    agg_adj2[:receptor_count, receptor_count:] = np.copy(dm)
    agg_adj2[receptor_count:, :receptor_count] = np.copy(np.transpose(dm))
    valid = np.zeros((receptor_count + ligand_count,))
    valid[:receptor_count] = 1
    np.savez(save_path, H=H, A1=agg_adj1, A2=agg_adj2, V=valid)


def collate_fn(batch):
    data = [x[0] for x in batch]
    pka = [x[1] for x in batch]

    # 在这之后继续进行您的处理
    max_natoms = max([len(item['H']) for item in data if item is not None])
    H = np.zeros((len(data), max_natoms, 56))
    A1 = np.zeros((len(data), max_natoms, max_natoms))
    A2 = np.zeros((len(data), max_natoms, max_natoms))
    V = np.zeros((len(data), max_natoms))
    # keys = []
    Atoms_Number = []
    for i in range(len(data)):
        natom = len(data[i]['H'])

        H[i, :natom] = data[i]['H']
        A1[i, :natom, :natom] = data[i]['A1']
        A2[i, :natom, :natom] = data[i]['A2']
        V[i, :natom] = data[i]['V']
        # keys.append(batch[i]['key'])
        Atoms_Number.append(natom)
    H = torch.from_numpy(H).float()
    A1 = torch.from_numpy(A1).float()
    A2 = torch.from_numpy(A2).float()
    V = torch.from_numpy(V).float()
    Atoms_Number = torch.Tensor(Atoms_Number)
    return H, A1, A2, V, Atoms_Number, pka


def train():

    # 数据集路径
    input_dir = "D:\GNN\GNN_DOVE\example\input"
    test_dir = "D:\GNN\GNN_DOVE\example\\test"
    # input_dir = 'D:\GNN\protein_data'
    # npz数据路径
    train_npzdir = 'D:\GNN\GNN_DOVE\\trainnpz_data'
    test_npzdir = 'D:\GNN\GNN_DOVE\\testnpz_data'
    if not os.path.exists(train_npzdir):
        os.mkdir(train_npzdir)
    if not os.path.exists(train_npzdir):
        os.mkdir(test_npzdir)

    prepare_data(input_dir, train_npzdir)
    prepare_data(test_dir, test_npzdir)
    train_npz = []
    test_npz = []

    npzfile_list = os.listdir(train_npzdir)
    for file in npzfile_list:
        train_npz.append(os.path.abspath(os.path.join(train_npzdir, file)))

    train_dataset = Single_Dataset(train_npz)

    test_dataset = Single_Dataset(test_npz)
    # print(train_dataset.__getitem__(0))

    # for temp in train_dataset:
    #     print('*' * 100)
    #     a, b = temp
    #     print(a.files)
    #     print(a, b, type(a), type(b))
    # exit(0)
    # length 长度
    train_data_size = len(train_dataset)
    test_data_size = len(test_dataset)

    # 如果train_data_size=10，则打印：训练数据集的长度为：10
    print("训练数据集的长度：{}".format(train_data_size))

    # 利用 Dataloader 来加载数据集
    train_dataloader = DataLoader(train_dataset, 1, shuffle=False,
                                  num_workers=4,
                                  drop_last=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False,
                                 num_workers=4,
                                 drop_last=False, collate_fn=collate_fn)
    # 创建网络模型
    pka_model = Pka_Model()

    if torch.cuda.is_available():
        pka_model = pka_model.cuda()  # 网络模型转移到cuda上

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵，fn 是 function 的缩写
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()  # 损失函数转移到cuda上

    # 优化器
    learning = 0.01  # 1e-2 就是 0.01 的意思
    optimizer = torch.optim.SGD(pka_model.parameters(), learning)  # 随机梯度下降优化器

    # 设置网络的一些参数
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0

    # 训练的轮次
    epoch = 5

    # 添加 tensorboard
    writer = SummaryWriter("logs")

    for i in range(epoch):
        print("-----第 {} 轮训练开始-----".format(i + 1))
        # 训练步骤开始
        pka_model.train()  # 当网络中有dropout层、batch norm层时，这些层能起作用

        for data in train_dataloader:

            H, A1, A2, V, Atoms_Number, pka = data

            if torch.cuda.is_available():
                protein = protein.cuda()  # 数据放到cuda上
                targets = targets.cuda()  # 数据放到cuda上
            outputs = pka_model(protein)
            loss = loss_fn(outputs, targets)  # 计算实际输出与目标输出的差距

            # 优化器对模型调优
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播，计算损失函数的梯度
            optimizer.step()  # 根据梯度，对网络的参数进行调优

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))  # 方式二：获得loss值
                writer.add_scalar("train_loss", loss.item(), total_train_step)

        # 测试步骤开始（每一轮训练后都查看在测试数据集上的loss情况）
        pka_model.eval()  # 当网络中有dropout层、batchnorm层时，这些层不能起作用
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():  # 没有梯度了
            for data in test_dataloader:  # 测试数据集提取数据
                protein, targets = data  # 数据放到cuda上
                if torch.cuda.is_available():
                    protein = protein.cuda()  # 数据放到cuda上
                    targets = targets.cuda()
                outputs = pka_model(protein)
                loss = loss_fn(outputs, targets)  # 仅data数据在网络模型上的损失
                total_test_loss = total_test_loss + loss.item()  # 所有loss
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy

        print("整体测试集上的Loss：{}".format(total_test_loss))
        print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
        total_test_step = total_test_step + 1

        torch.save(pka_model, "./model/tudui_{}.pth".format(i))  # 保存每一轮训练后的结果
        # torch.save(tudui.state_dict(),"tudui_{}.path".format(i)) # 保存方式二
        print("模型已保存")

    writer.close()


if __name__ == "__main__":
    train()
