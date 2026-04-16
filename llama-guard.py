import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm

# 1. 環境設定(ここは2と3で切り替え)
model_id = "meta-llama/Meta-Llama-Guard-2-8B"
# model_id = "meta-llama/Llama-Guard-3-8B"
device="cuda"

# 4bitに量子化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 2. モデルとトークナイザーのロード
print("Loading model (Llama Guard 2 - 8B)...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda",
)

def moderate(chat):
    """
    Llama Guard 2/3 による有害性判定
    """
    # テンプレート適用 (BatchEncodingオブジェクトを取得)
    inputs = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)

    with torch.no_grad():
        # **inputs で辞書を展開して渡すことで、型エラーを回避
        output = model.generate(
            **inputs,
            max_new_tokens=20,
            pad_token_id=0,
            do_sample=False
        )

    # 入力プロンプトの長さを取得して、生成された判定部分のみを抽出
    prompt_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

# 3. データのロードと準備 (100-100抽出)
print("Loading ToxicChat dataset...")
dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")

# 有害100件、安全100件をランダムに抽出
toxic_samples = dataset.filter(lambda x: x['toxicity'] == 1).shuffle(seed=42).select(range(100))
safe_samples = dataset.filter(lambda x: x['toxicity'] == 0).shuffle(seed=42).select(range(100))
test_list = list(toxic_samples) + list(safe_samples)

# 4. 評価ループの実行
results = []
print(f"Starting Evaluation for {len(test_list)} samples...")

for data in tqdm(test_list):
    chat = [
        {"role": "user", "content": data["user_input"]},
        {"role": "assistant", "content": data["model_output"]}
    ]

    prediction = moderate(chat)

    # 正解ラベルの整理
    gt_label = "unsafe" if data["toxicity"] == 1 else "safe"

    # 判定結果の保存
    results.append({
        "conv_id": data["conv_id"],
        "user_input": data["user_input"],
        "ground_truth": gt_label,
        "prediction": prediction,
        "is_correct": (gt_label in prediction) # Llama Guardの出力に unsafe/safe が含まれるか
    })

# 5. 結果の集計とJSON出力
accuracy = sum(1 for r in results if r["is_correct"]) / len(results)
print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")

# JSONファイルに保存
with open("output_results2.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n--- output_results2.json に詳細を保存しました ---")

# 6. テーブル形式でプレビュー表示
df = pd.DataFrame(results)

# 有害カテゴリの集計を表示
print("\n### 検出された有害カテゴリ（S1〜S14）の内訳 ###")
print(df[df['prediction'].str.contains('unsafe')]['prediction'].value_counts())
