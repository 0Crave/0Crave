import math
import torch
import sys
from tqdm.notebook import tqdm
from scipy.stats import entropy
use_gpu = torch.cuda.is_available()
import time
import FSLTask
import os
import cv2
import torch
import torch
import torchvision.transforms as transforms
from torchvision.models import wide_resnet50_2
from PIL import Image

#<---------------------------------引入库--------------------->


# ========================================
#      loading datas

#

def centerDatas(datas):
    datas= datas - datas.mean(1, keepdim=True) #零归一化
    datas = datas / torch.norm(datas, dim=2, keepdim= True) # 单位化

    return datas

#特征均值归一化
def scaleEachUnitaryDatas(datas):
  
    norms = datas.norm(dim=2, keepdim=True)

    return datas/norms


def QRreduction(datas):
    ndatas = torch.linalg.qr(datas.permute(0, 2, 1),'reduced').R
    ndatas = ndatas.permute(0, 2, 1)
    return ndatas

#奇异值分解 数据压缩降维
def SVDreduction(ndatas,K):
    # ndatas = torch.linear.qr(datas.permute(0, 2, 1),'reduced').R
    # ndatas = ndatas.permute(0, 2, 1)
    _,s,v = torch.svd(ndatas)
    ndatas = ndatas.matmul(v[:,:,:K])

    return ndatas


def predict(gamma, Z, labels):
    # 需要添加全局变量声明或作为参数传入
    global n_runs, n_lsamples, n_ways, n_shot, n_queries, n_usamples
    # #Certainty_scores = 1 + (Z*torch.log(Z)).sum(dim=2) / math.log(5)
    # Z[:,:n_lsamples].fill_(0)
    # Z[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
    Y = torch.zeros(n_runs,n_lsamples, n_ways,device='cuda')
    Y.scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
    #tZ_Z = torch.bmm(torch.transpose(Z,1,2), Z)
    delta = torch.sum(Z, 1)
    #L = tZ_Z - torch.bmm(tZ_Z, tZ_Z/delta.unsqueeze(1))
    iden = torch.eye(5,device='cuda')
    iden = iden.reshape((1, 5, 5))
    iden = iden.repeat(10000, 1, 1)
    W = torch.bmm(torch.transpose(Z,1,2), Z/delta.unsqueeze(1))
    #W = W/W.sum(1).unsqueeze(1)
    #isqrt_diag = 1. / torch.sqrt(1e-4 + torch.sum(W, dim=-1,keepdim=True))
    # checknan(laplacian=isqrt_diag)
    #W = W * isqrt_diag[:, None, :] * isqrt_diag[:, :, None]
    #W = W * isqrt_diag * torch.transpose(isqrt_diag,dim0=2,dim1=1)
    L = iden - W#(W + W.bmm(W))/2
    Z_l = Z[:,:n_lsamples]

    #A = np.dot(np.linalg.inv(torch.matmul(torch.transpose(Z_l,1,2), Z_l) + gamma * L), torch.bmm(torch.transpose(Z_l,1,2), Y))
    u = torch.linalg.cholesky(torch.bmm(torch.transpose(Z_l,1,2), Z_l) + gamma * L)# + 0.1*iden)
    A = torch.cholesky_solve(torch.bmm(torch.transpose(Z_l,1,2), Y), u)
    Pred = Z.bmm(A)
    normalizer = torch.sum(Pred,dim=1,keepdim=True)
    # #normalizer = Pred[:,:n_lsamples].max(dim=1)[0].unsqueeze(1)
    Pred = (n_shot+n_queries)*Pred/normalizer
    # normalizer = torch.sum(Pred, dim=2, keepdim=True)
    # Pred = Pred/normalizer
    # Pred[:, :n_lsamples].fill_(0)
    # Pred[:, :n_lsamples].scatter_(2, labels[:, :n_lsamples].unsqueeze(2), 1)
    # N = PredZ.shape[0]
    # K = PredZ.shape[1]
    # pred = np.zeros((N, K))
    #
    # for k in range(K):
    #     current_pred = np.dot(Z, A[:, k])

    return Pred#.clamp(0,1)

def predictW(gamma, Z, labels):
    # 需要添加全局变量声明或作为参数传入
    global n_runs, n_lsamples, n_ways, n_shot, n_queries, n_usamples
    # #Certainty_scores = 1 + (Z*torch.log(Z)).sum(dim=2) / math.log(5)
    # Z[:,:n_lsamples].fill_(0)
    # Z[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
    Y = torch.zeros(n_runs,n_lsamples, n_ways,device='cuda')
    Y.scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
    tZ_Z = torch.bmm(torch.transpose(Z,1,2), Z)
    delta = torch.sum(Z, 1)
    L = tZ_Z - torch.bmm(tZ_Z, tZ_Z/delta.unsqueeze(1))
    Z_l = Z[:,:n_lsamples]

    #A = np.dot(np.linalg.inv(torch.matmul(torch.transpose(Z_l,1,2), Z_l) + gamma * L), torch.bmm(torch.transpose(Z_l,1,2), Y))
    u = torch.linalg.cholesky(torch.bmm(torch.transpose(Z_l,1,2), Z_l) + gamma * L)# + 0.1*
    #u = torch.linalg.cholesky(gamma * L)
    A = torch.cholesky_solve(torch.bmm(torch.transpose(Z_l,1,2), Y), u)
    P = Z.bmm(A)
    _, n, m = P.shape
    r = torch.ones(n_runs, n_lsamples + n_usamples,device='cuda')
    c = torch.ones(n_runs, n_ways,device='cuda') * (n_shot + n_queries)
    u = torch.zeros(n_runs, n).cuda()
    maxiters = 1000
    iters = 1
    # normalize this matrix
    while torch.max(torch.abs(u - P.sum(2))) > 0.01:
        u = P.sum(2)
        P *= (r / u).view((n_runs, -1, 1))
        P *= (c / P.sum(1)).view((n_runs, 1, -1))
        P[:,:n_lsamples].fill_(0)
        P[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
        if iters == maxiters:
            break
        iters = iters + 1
    return P

class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways
              
# ---------  GaussianModel
class GaussianModel(Model):
    def __init__(self, n_ways, lam):
        super(GaussianModel, self).__init__(n_ways)
        self.mus = None         # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam
        
    def clone(self):
        other = GaussianModel(self.n_ways)
        other.mus = self.mus.clone()
        return self

    def cuda(self):
        self.mus = self.mus.cuda()
        
    def initFromLabelledDatas(self, ndatas, n_runs, n_shot, n_queries, n_ways, n_nfeat):
        self.mus_ori = ndatas.reshape(n_runs, n_shot+n_queries,n_ways, n_nfeat)[:,:n_shot,].mean(1)
        self.mus = self.mus_ori.clone()
        self.mus = self.mus / self.mus.norm(dim=2, keepdim=True)
        # self.mus_ori = torch.randn(n_runs, n_ways,n_nfeat,device='cuda')
        # self.mus_ori = self.mus_ori/self.mus_ori.norm(dim=2,keepdim=True)
        # self.mus = self.mus_ori.clone()

    def initFromCenter(self, mus):
        #self.mus_ori = ndatas.reshape(n_runs, n_shot+n_queries,n_ways, n_nfeat)[:,:1,].mean(1)
        self.mus = mus
        self.mus = self.mus / self.mus.norm(dim=2, keepdim=True)
        # self.mus_ori = torch.randn(n_runs, n_ways,n_nfeat,device='cuda')
        # self.mus_ori = self.mus_ori/self.mus_ori.norm(dim=2,keepdim=True)
        # self.mus = self.mus_ori.clone()

    def updateFromEstimate(self, estimate, alpha, l2 = False):

        diff = self.mus_ori - self.mus
        Dmus = estimate - self.mus
        if l2 == True:
            self.mus = self.mus + alpha * (Dmus) + 0.01 * diff
        else:
            self.mus = self.mus + alpha * (Dmus)
        #self.mus/=self.mus.norm(dim=2, keepdim=True)

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):
        
        r = r.cuda()
        c = c.cuda()
        n_runs, n, m = M.shape
        P = torch.exp(- self.lam * M)
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)
                                         
        u = torch.zeros(n_runs, n).cuda()
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M)
    
    def getProbas(self, ndatas, n_runs, n_ways, n_usamples, n_lsamples):
        # 需要访问 labels，应该添加全局声明或作为参数传入
        global labels
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        dist = (ndatas.unsqueeze(2)-self.mus.unsqueeze(1)).norm(dim=3).pow(2)
        device = ndatas.device  # 获取输入数据的设备
        p_xj = torch.zeros_like(dist).to(device)  # 确保p_xj在正确的设备上
        r = torch.ones(n_runs, n_usamples).to(device)
        c = torch.ones(n_runs, n_ways).to(device) * (n_queries)
       
        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-3)
        # _, y_pseudo = torch.max(p_xj_test, 2)
        # Certainty_scores = 1 + (p_xj_test*torch.log(p_xj_test)).sum(axis=2) / math.log(5)
        # Certainty_scores = Certainty_scores.unsqueeze(2)
        #p_xj = torch.where(p_xj > 0.9, torch.tensor(1.,device='cuda'), p_xj)
        # p_xj_test[alpha[0],alpha[1],:].fill_(0)
        # p_xj_test[alpha[0],alpha[1],:].scatter_(2, y_pseudo[alpha[0],alpha[1]], 1)
        #sup_alpha = np.where(Certainty_scores >= alpha)[0]
        p_xj[:, n_lsamples:] = p_xj_test
        p_xj[:,:n_lsamples].fill_(0)
        # 确保labels也在正确的设备上
        labels = labels.to(device)
        p_xj[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
        
        return p_xj

    def estimateFromMask(self, mask, ndatas):

        emus = mask.permute(0,2,1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))

        return emus

          
# =========================================
#    MAP
# =========================================

class MAP:
    def __init__(self, alpha=None):
        
        self.verbose = False
        self.progressBar = False
        self.alpha = alpha
        # 添加这些参数作为类属性
        self.n_runs = None
        self.n_lsamples = None
        self.n_ways = None
        self.n_shot = None
        self.n_queries = None
        self.n_usamples = None
    
    def getAccuracy(self, probas):
        # 需要访问 labels 和 n_lsamples
        global labels
        # 使用self.n_lsamples等类属性
        olabels = probas.argmax(dim=2)
        matches = labels.eq(olabels).float()
        acc_test = matches[:,self.n_lsamples:].mean(1)    

        m = acc_test.mean().item()
        pm = acc_test.std().item() *1.96 / math.sqrt(self.n_runs)
        return m, pm
    
    def performEpoch(self, model, ndatas, n_runs, n_ways, n_usamples, n_lsamples, epochInfo=None):
     
        p_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
        self.probas = p_xj
        
        if self.verbose:
            print("accuracy from filtered probas", self.getAccuracy(self.probas))

        m_estimates = model.estimateFromMask(self.probas,ndatas)
               
        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)
        #self.alpha -= 0.001
        if self.verbose:
            op_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
            acc = self.getAccuracy(op_xj)
            print("output model accuracy", acc)
        
    def loop(self, model, ndatas, n_runs, n_ways, n_usamples, n_lsamples, n_epochs=20):
        # 设置类属性
        self.n_runs = n_runs
        self.n_lsamples = n_lsamples
        self.n_ways = n_ways
        self.n_usamples = n_usamples
        self.verbose = False
        
        self.probas = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
        if self.verbose:
            print("initialisation model accuracy", self.getAccuracy(self.probas))

        if self.progressBar:
            if type(self.progressBar) == bool:
                pb = tqdm(total = n_epochs)
            else:
                pb = self.progressBar
           
        for epoch in range(1, n_epochs+1):
            if self.verbose:
                print("----- epoch[{:3d}]  lr_p: {:0.3f}".format(epoch, self.alpha))
            p_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
            self.probas = p_xj

            if self.verbose:
                print("accuracy from filtered probas", self.getAccuracy(self.probas))
            pesudo_L = predictW(0.05, self.probas, labels)
            if self.verbose:
                print("accuracy from AnchorGraph probas", self.getAccuracy(pesudo_L))
            #(pesudo_L + self.probas)
            beta = 0.7
            # p_xj[:,:n_lsamples].fill_(0)
            # p_xj[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
            # beta*pesudo_L + (1-beta)*self.probas
            #pesudo_L[:,n_lsamples:] = (beta * pesudo_L[:,n_lsamples:] + (1 - beta) * self.probas[:,n_lsamples:])
            m_estimates = model.estimateFromMask((beta*pesudo_L + (1-beta)*p_xj).clamp(0,1), ndatas)
            #m_estimates = model.estimateFromMask(pesudo_L.clamp(0, 1), ndatas)
            #m_estimates = model.estimateFromMask(p_xj, ndatas)

            # update centroids
            model.updateFromEstimate(m_estimates, self.alpha)
            # self.alpha -= 0.001
            if self.verbose:
                op_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
                acc = self.getAccuracy(op_xj)
                print("output model accuracy", acc)
            if (self.progressBar): pb.update()
        
        # get final accuracy and return it
        op_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
        acc = self.getAccuracy(op_xj)
        # print(op_xj)
        # print("\n")

        return acc,op_xj.argmax(dim = 2)
#<-------------------------------------------------------------->
def count_subdirectories(path):
    """
    统计给定路径下的子文件夹数量

    :param path: 文件夹的路径
    :return: 子文件夹的数量
    """
    try:
        # 获取给定路径下的所有条目
        entries = os.listdir(path)
        # 统计子文件夹的数量
        subdir_count = sum(1 for entry in entries if os.path.isdir(os.path.join(path, entry)))
        return subdir_count
    except FileNotFoundError:
        # print("指定的路径未找到。")
        print(f'{path} 路径未找到')
        return None
    except PermissionError:
        # print("没有权限访问该路径。")
        print(f'{path} 没有权限访问')
        return None




# 以下为逻辑函数, main函数的入参和最终的结果输出不修改
def main(to_pred_dir, result_save_path):
    import numpy as np

    model_dir = os.path.dirname(to_pred_dir) # 当前文件夹路径

# <---------------------------------------------------------------------------
    global labels,n_runs,n_shot,n_ways,n_queries,n_lsamples,n_usamples,n_samples
#<-----------------------------------------------------
    transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.485, 0.456, 0.406]),np.array([0.229, 0.224, 0.225]))
        ])
        
    ckpt = os.path.join(model_dir, 'wrn_pre_protonet_train700_center_10.pth')
    ckpt_weights = torch.load(ckpt,map_location='cuda:0')
    model = wide_resnet50_2(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 700)
    model.load_state_dict(ckpt_weights)
    model = torch.nn.Sequential(*list(model.children())[:-1]).cuda()
    model.eval()
    
    def extract_features(image_path):
            img = Image.open(image_path).convert('RGB')
            img = transform(img).unsqueeze(0).cuda() # 1,3,224,224
            with torch.no_grad():
                feature = model(img)
            
            feature = feature.reshape(-1) # 2048
            
            return feature
    def power_transform(ndatas, labels):
        """
        对输入数据进行特征变换
        """
        # 确保数据在GPU上
        ndatas = ndatas.cuda()
        labels = labels.cuda()
        
        # Power transform
        beta = 0.5
        ndatas = scaleEachUnitaryDatas(ndatas)  # 确保返回的tensor在GPU上
        ndatas = SVDreduction(ndatas, 40)
        n_nfeat = ndatas.size(2)
        
        ndatas = centerDatas(ndatas)
        print("size of the datas...", ndatas.size())
        print("ndatas device:", ndatas.device)
        print("labels device:", labels.device)

        # MAP
        lam = 10
        model = GaussianModel(n_ways, lam)
        n_runs = 1
        print("n_runs", n_runs)
        
        # 在初始化模型之前再次确认数据在GPU上
        ndatas = ndatas.cuda()
        labels = labels.cuda()
        model.initFromLabelledDatas(ndatas, n_runs, n_shot, n_queries, n_ways, n_nfeat)

        alpha = 0.2
        optim = MAP(alpha)
        optim.verbose = True
        optim.progressBar = True
        

        
        acc_test, pred_labels = optim.loop(model, ndatas, n_runs, n_ways, n_usamples, n_lsamples, n_epochs=50)
        result = pred_labels[:, n_lsamples:]
        print("result.shape",result.shape)
        return result


# ------------------------------------------------------------------------>

    dirpath = os.path.abspath(to_pred_dir)
# <------------------------------------------------------------------------
    filepath = os.path.join(dirpath, 'testA') # 测试集A文件夹路径
# ------------------------------------------------------------------------>
    task_list = os.listdir(filepath)
    print(task_list)
    task_list = [task for task in task_list if task.startswith('task')]
    task_list = sorted(task_list, key=lambda x: int(x.replace('task', '')))  # 按数值大小排序
    print(task_list)
    def process_tasks(task_name, filepath):
        task_features = []
        task_labels = []
        
    
        task_path = os.path.join(filepath, "")
        # print(f"Task path: {task_path}")
        
        # 处理support文件夹
        support_path = os.path.join(task_path, 'support')
        # print(f"Support path: {support_path}")
        if os.path.exists(support_path) and os.path.isdir(support_path):
            support_folders = os.listdir(support_path)
            # print(f"Support folders found: {support_folders}")
            
            for subfolder in support_folders:
                subfolder_path = os.path.join(support_path, subfolder)
                if os.path.isdir(subfolder_path):
                    files = os.listdir(subfolder_path)
                    # print(f"Files in {subfolder}: {files}")
                    
                    for filename in files:
                        if filename.endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(subfolder_path, filename)
                            feature = extract_features(image_path)
                            if feature is not None:
                                feature = feature.cpu()
                                feature = feature.tolist()
                                task_features.append(feature)
                                task_labels.append(int(subfolder))
                                # print(f"Added feature for {filename} with label {subfolder}")
        
        # 处理query文件夹
        query_path = os.path.join(task_path, 'query')
        print(f"Query path: {query_path}")
        if os.path.exists(query_path) and os.path.isdir(query_path):
            test_img_lst = sorted(
                [name for name in os.listdir(query_path) if name.endswith('.png')],
                key=lambda x: int(x.split('_')[1].split('.')[0])
            )
            # print(f"Query images found: {test_img_lst}")
            
            for filename in test_img_lst:
                image_path = os.path.join(query_path, filename)
                feature = extract_features(image_path)
                if feature is not None:
                    feature = feature.cpu()
                    feature = feature.tolist()
                    task_features.append(feature)
                    task_labels.append(-1)
                    # print(f"Added feature for query image {filename}")
        
        # print(f"Total features collected: {len(task_features)}")
        # print(f"Total labels collected: {len(task_labels)}")
    
        return task_features, task_labels

    res = ['img_name,label']  # 初始化结果文件，定义表头

    for task_idx, task_name in enumerate(task_list):
        print(f"Processing task: {task_name}")
        task_path = os.path.join(filepath, task_name)
        print("我的调试：",task_path)
        ndatas,labels = process_tasks(task_name, task_path)

         # 转换为tensor并添加维度
        ndatas = torch.tensor(ndatas).unsqueeze(0)  # shape: (1, n_samples, n_features)
        labels = torch.tensor(labels).unsqueeze(0)    # shape: (1, n_samples)
        n_runs = 1
        task_result = power_transform(ndatas,labels)
        query_path = os.path.join(filepath, task_name, 'query')  # 查询集路径（无标签，待预测图片）

                # 修改排序逻辑,从第5个字符开始提取数字进行排序
        test_img_lst = sorted([name for name in os.listdir(query_path) 
                          if name.endswith('.png')],
                          key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        for path_idx, pathi in enumerate(test_img_lst):
            name_img = os.path.join(query_path, pathi)
            
            # 从 result 中提取标签，task_idx 对应任务的索引，path_idx 对应当前图片的索引
            label = task_result[0,path_idx].item()  # 获取对应的标签
            
            # 将图片路径和标签加入到结果列表中
            res.append(f"{pathi},{label}")


    # 你可以在此打印或保存最终的结果
    # print(res)
    with open(result_save_path, 'w') as f:
        f.write('\n'.join(res))

n_shot = 5
n_ways = 10
n_queries = 2
n_lsamples = n_ways * n_shot
n_usamples = n_ways * n_queries
n_samples = n_lsamples + n_usamples

if __name__ == "__main__":
    # ！！！以下内容不允许修改，修改会导致评分出错

    to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    result_save_path = sys.argv[2]  # 预测结果保存文件路径，已指定格式为csv

    main(to_pred_dir, result_save_path)


