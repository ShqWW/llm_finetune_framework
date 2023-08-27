# CUDA_VISIBLE_DEVICES=5 python train_qlora.py --cfg config/Config_Baichuan_7B_QLoRA.py
# CUDA_VISIBLE_DEVICES=4 nohup python -u train_qlora.py --cfg config/Config_Baichuan_7B_QLoRA.py >> out.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=23461 train_qlora.py --cfg config/Config_Baichuan_7B_QLoRA.py
# CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=23461 train_qlora.py --cfg config/Config_Qwen_7B_QLoRA.py

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=23461 train_qlora.py --cfg config/Config_Moss_QLoRA.py
# CUDA_VISIBLE_DEVICES=4,5 nohup torchrun --nproc_per_node=2 --master_port=23461 train_qlora.py --cfg config/Config_Moss_QLoRA.py >> out.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 python3 train_qlora.py --cfg config/Config_Moss_QLoRA.py
# CUDA_VISIBLE_DEVICES=4 python3 train_qlora.py --cfg config/Config_Chatglm2_6B_QLoRA.py
