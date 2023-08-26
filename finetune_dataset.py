import torch
import os
import re
import json
class PreDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, datapath, eos_token_id=2, pad_token_id=0, ignore_label_id=-100, max_len = 2048):
    #    self.alltext = ['我是大煞笔', '你是大聪明是吧哼嘿哈']
        # self.alltext = ['我是大疆助手，欢迎随时向我提问', '你是如何学习和记忆的']
        self.alltext = []

        self.get_text(datapath)
        # print(self.alltext)
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.ignore_label_id = ignore_label_id
        self.max_len = max_len
    def __len__(self):
        return len(self.alltext)
    
    def get_text(self, datapath):
        file_name_list = os.listdir(datapath)
        for file_name in file_name_list:
            file_path = os.path.join(datapath, file_name)
            with open(file_path) as f:
                # a = json.load(f)
                lines = f.readlines()
                for line in lines:
                    pattern = re.search(r"\"inside\":\"(.*?)\"", line)
                    if pattern is not None:
                        text = pattern.group(1)
                        self.alltext.append(text)

    def __getitem__(self, index):
        text = self.alltext[index]
        text_id = self.tokenizer.encode(text=text, add_special_tokens=False)
        text_id = text_id + [self.eos_token_id]
        return text_id

    def collate_fn(self, text_ids):
        batch_lens = [len(sample) for sample in text_ids]
        batch_len_max = min(max(batch_lens), self.max_len)
        pad_inputs = []
        pad_labels = []
        for batch_len, input in zip(batch_lens, text_ids):
            if batch_len < batch_len_max:
                pad_len = batch_len_max - batch_len
                pad_input = input + [self.pad_token_id]*pad_len
                pad_label = input + [self.ignore_label_id]*pad_len
            else:
                pad_input = input[:batch_len_max]
                pad_label = pad_input
            pad_inputs.append(torch.LongTensor(pad_input))
            pad_labels.append(torch.LongTensor(pad_label)) 
        pad_inputs = torch.stack(pad_inputs)
        pad_labels = torch.stack(pad_labels)
        return {'input_ids': pad_inputs, 'labels': pad_labels}

        