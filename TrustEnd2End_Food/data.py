import numpy as np
import torch
from torch.utils.data import Dataset
import json
from torchvision import transforms
from PIL import Image


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Food101Data(Dataset):
    """
    load multi-view data
    """

    def __init__(self, args):
        """
        :param root: data name and path
        :param train: load training set or test set
        """
        super().__init__()
        self.args = args
        
        with open(args.split_file) as fin:
            split_info = json.loads(fin.read())
        train_split = split_info["train"]
        val_split = split_info["val"]
        test_split = split_info["test"]
        
        with open(args.text_file) as fin:
            text = json.loads(fin.read())
        self.text = text
        
        with open(args.class_file) as fin:
            cls2int = json.loads(fin.read())
        self.cls2int = cls2int
        
        self.split = args.split
        if args.split == "train":
            self.the_split = [each for each in train_split if each in self.text] 
            self.transforms = transforms.Compose([
                transforms.Resize((256, 256)),        # Resize to a consistent size
                transforms.RandomResizedCrop(224),    # Random crop to fit ResNet input
                transforms.RandomHorizontalFlip(),    # Randomly flip images horizontally
                # transforms.RandomRotation(15),        # Randomly rotate images by 15 degrees
                transforms.ToTensor(),                # Convert to tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet means
                                     std=[0.229, 0.224, 0.225])   # Normalize with ImageNet stds
            ])
        elif args.split == "val":
            self.the_split = [each for each in val_split if each in self.text] 
            self.transforms = transforms.Compose([
                transforms.Resize((256, 256)),        # Resize to 256x256
                transforms.CenterCrop(224),           # Center crop to 224x224
                transforms.ToTensor(),                # Convert to tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet means
                                     std=[0.229, 0.224, 0.225])   # Normalize with ImageNet stds
            ])
        elif args.split == "test":
            self.the_split = [each for each in test_split if each in self.text] 
            self.transforms = transforms.Compose([
                transforms.Resize((256, 256)),        # Resize to 256x256
                transforms.CenterCrop(224),           # Center crop to 224x224
                transforms.ToTensor(),                # Convert to tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet means
                                     std=[0.229, 0.224, 0.225])   # Normalize with ImageNet stds
            ])
        else:
            raise NotImplementedError
        
        if args.debug:
            self.the_split = self.the_split[:20]
        
    def __len__(self):
        return len(self.the_split)
    
    def __getitem__(self, index):
        
        img_id = self.the_split[index]
        class_str = "_".join(img_id.split("_")[:-1])
        t = self.text[img_id]
        
        if self.split == "test":
            img_path = os.path.join(self.args.data_path, "images", "test", class_str, img_id)
        elif self.split in {"train", "val"}:
            img_path = os.path.join(self.args.data_path, "images", "train", class_str, img_id)
        else:
            raise NotImplementedError
        # TODO: apply transformer
        i = Image.open(img_path).convert('RGB')
        i_tensor = self.transforms(i)
        class_id  = self.cls2int[class_str]
        
        return i_tensor, t, class_id
    
    
def collate_fn(args, batch, tokenizer):
    
    batch_i = []
    batch_l = []
    batch_t = []
    for i, t, l in batch:
        batch_i.append(i)
        batch_l.append(l)
        
        # f_ = {}
        # input_ids = tokenizer(
        #     t,
        #     max_length=args.max_text_len, truncation=True, padding=False)['input_ids']
        # f_['input_ids'] = input_ids
        # batch_t.append(f_)
        batch_t.append(t)
    
    batch_i = torch.stack(batch_i, dim=0)
    batch_l = torch.as_tensor(batch_l).long()
    
    # texts_batch = tokenizer.pad(
    #         batch_t,
    #         padding=True,
    #         max_length=args.max_text_len,
    #         return_tensors='pt',
    #     )
    texts_batch = tokenizer(batch_t, padding="max_length", max_length=args.max_text_len, return_tensors='pt', truncation=True)
    texts_batch = {'input_ids': texts_batch['input_ids'], 'attention_mask': texts_batch['attention_mask']}
    
    return batch_i, texts_batch, batch_l

        
if __name__ == "__main__":
    
    import argparse
    from torch.utils.data import DataLoader
    
    from transformers import AutoTokenizer
    
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    args.split_file = "dataset/Food101/split.json"
    args.text_file = "dataset/Food101/text.json"
    args.class_file = "dataset/Food101/class_idx.json"
    args.data_path = "dataset/Food101/"
    args.lm_name = "bert-base-uncased"
    args.max_text_len = 128
    args.split = "train"
    
    
    tr_ds = Food101Data(args)
    # print(len(tr_ds))
    # print(tr_ds[1])
    
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    tr_dl = DataLoader(tr_ds, batch_size=16, shuffle=False, num_workers=4, collate_fn=lambda x: collate_fn(args, x, tokenizer))
    for batch in tr_dl:
        i, t, l = batch
        print(i.shape, l.shape, t['input_ids'].shape)
        print(type(t))
        break
        