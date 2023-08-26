from finetune_dataset import *
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if __name__ == '__main__':
    model_path = 'remote_scripts/Baichuan-7B'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    trset = PreDataset(tokenizer=tokenizer, max_len=30)
    # dataloader = torch.utils.data.DataLoader(trset, batch_size=2,
    #                                              num_workers=8,
    #                                              collate_fn=trset.collate_fn)
    
    # for data in dataloader:
    #     print(data)
