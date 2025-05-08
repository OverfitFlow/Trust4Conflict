import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math
import h5py
from sklearn.model_selection import train_test_split


def load_data(args):
    if args.data_path == "datasets/ICLR2021 Datasets":
        """_summary_
        Following logic is developed based on the inspection
        """
        
        dataname2filename = {
            "CUB": "cub_googlenet_doc2vec_c10.mat",  # file_name, split_seed
            "PIE": "PIE_face_10.mat",
            "Scene15": "scene15_mtv.mat",
            "Caltech101": "2view-caltech101-8677sample.mat",
            "Handwritten": "handwritten.mat",
        }
        if args.data_name in dataname2filename:
            data = sio.loadmat(
                os.path.join(args.data_path, dataname2filename[args.data_name]))
            # processing label items
            if 'gt' not in data:
                if 'Y' in data:
                    data['gt'] = data['Y']
                    data.pop('Y')
                elif 'y' in data:
                    data['gt'] = data['y']
                    data.pop('y')
                else:
                    raise NotImplementedError
            data['gt'] = data['gt'].astype(np.int64)
            if np.min(data['gt']) == 1:
                data['gt'] -= 1
            
            # processing X to have same format
            views_dims = []
            num_views = data['X'].shape[1]
            for v_num in range(num_views): 
                if data['X'][0, v_num].shape[0] !=  data['gt'].shape[0]:
                    data['X'][0, v_num] = data['X'][0, v_num].transpose()
                views_dims.append(data['X'][0, v_num].shape[1])

            # print(views_dims, args.data_name)
            data['n_views'] = num_views
            data['views_dims'] = views_dims            
            
            split_idx = {}
            split_idx['train_idx'] = np.load(os.path.join(args.data_path, f"{dataname2filename[args.data_name].strip('.mat')}_train_idx.npy"))
            split_idx['test_idx'] = np.load(os.path.join(args.data_path, f"{dataname2filename[args.data_name].strip('.mat')}_test_idx.npy"))
            
        elif args.data_name == "HMDB":
            f = h5py.File(os.path.join(args.data_path, 'HMDB51_HOG_MBH.mat'))
            data = {}
            data['gt'] = np.array(f['gt'], dtype=np.int64).transpose()
            data['gt'] -= 1
            data['X'] = np.empty((1, 2), dtype=object)
            data['X'][0, 0] = np.array(f['x1']).transpose()
            data['X'][0, 1] = np.array(f['x2']).transpose()
            
            data['n_views'] = 2
            data['views_dims'] = [data['X'][0, 0].shape[1], data['X'][0, 1].shape[1]]
            
            split_idx = {}
            split_idx['train_idx'] = np.load(os.path.join(args.data_path, f"{'HMDB51_HOG_MBH.mat'.strip('.mat')}_train_idx.npy"))
            split_idx['test_idx'] = np.load(os.path.join(args.data_path, f"{'HMDB51_HOG_MBH.mat'.strip('.mat')}_test_idx.npy"))
            
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    return data, split_idx


class Multi_view_data(Dataset):
    """
    load multi-view data
    """

    def __init__(self, args, data, train_idx, test_idx, is_train=False, used_views=-1):
        """
        :param root: data name and path
        :param train: load training set or test set
        """
        super(Multi_view_data, self).__init__()
        self.args = args
        self.data = data
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.is_train = is_train
        
        self.n_views = self.data['n_views']
        self.views_dims = self.data['views_dims']
        
        y = self.data['gt']
        if self.is_train:
            y = y[self.train_idx]
        else:
            y = y[self.test_idx]
        self.y = np.reshape(y, (-1, ))
        self.n_classes = np.max(y) + 1

        self.used_views = used_views
        self.preprocess_X(args.conflict_test, args.conflict_sigma, args.conflict_seed)
        
    def preprocess_X(self, is_conflict=False, conflict_sigma=0.0, conflict_seed=None):
        if is_conflict:
            rng = np.random.RandomState(conflict_seed)
            noise_vs = rng.choice(range(self.n_views), size=self.n_views//2, replace=False).tolist()
            
        if self.used_views == [-1]:
            used_views = range(self.n_views)
        else:
            used_views = self.used_views
            self.n_views = len(used_views)
            self.views_dims = [each for i, each in enumerate(self.views_dims) if i in used_views]
        
        self.X = dict()
        for i, v_num in enumerate(used_views):
            X_v = self.data['X'][0, v_num]
            if is_conflict and v_num in noise_vs:
                X_v = rng.normal(X_v, conflict_sigma)
            train_data_v, test_data_v = X_v[self.train_idx], X_v[self.test_idx]
            
            scaler_v = MinMaxScaler((0, 1))
            scaler_v.fit(X_v)
            if self.is_train:
                self.X[i] = scaler_v.transform(train_data_v)
            else:
                self.X[i] = scaler_v.transform(test_data_v)

    def __getitem__(self, idx):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][idx]).astype(np.float32)
        target = self.y[idx]
        return data, target

    def __len__(self):
        return self.X[0].shape[0]
