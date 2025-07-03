from transformers import BigBirdModel, BigBirdTokenizer

model_path = 'hub/google/bigbird-roberta-base'

try:
    model = BigBirdModel.from_pretrained(model_path)
    tokenizer = BigBirdTokenizer.from_pretrained(model_path)
    print("模型和tokenizer加载成功！")
except Exception as e:
    print(f"加载失败: {str(e)}")
