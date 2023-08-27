# CUDA_VISIBLE_DEVICES=3,4 python my_inference.py
# CUDA_VISIBLE_DEVICES=5 python my_inference.py --cfg config/Config_Baichuan_7B_QLoRA.py
CUDA_VISIBLE_DEVICES=4 python my_inference.py --cfg Config_Qwen_7B_QLoRA.py
# CUDA_VISIBLE_DEVICES=5 python my_inference.py --cfg config/Config_Moss_QLoRA.py
# CUDA_VISIBLE_DEVICES=4 python my_inference.py --cfg config/Config_Chatglm2_6B_QLoRA.py 
