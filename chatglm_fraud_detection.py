import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ================== 配置区 ==================
MODEL_PATH = "./chatglm3-6b"
DATA_PATH = "dialogue_data.csv"          # 输入数据文件
OUTPUT_CSV = "chatglm_train_predictions.csv"  # 输出预测结果文件
MAX_LEN = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== Prompt ==================
def build_prompt(dialogue: str) -> str:
    return f"""
你是一个金融风控领域的专家，请判断下面的对话是否属于诈骗对话。

【诈骗】包括但不限于：
- 诱导转账、索要验证码
- 冒充客服、银行、公安
- 制造紧急情况或恐慌情绪

【非诈骗】包括：
- 正常咨询、正常业务沟通
- 不涉及资金、验证码或隐私诱导

请你只输出一个词：
诈骗
或
非诈骗

对话内容：
{dialogue}
"""

# ================== 输出解析 ==================
def parse_prediction(output: str) -> int:
    output = output.strip()
    if "非诈骗" in output:
        return 0
    if "诈骗" in output:
        return 1
    return 0  # 兜底

# ================== 主流程 ==================
def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print("Detected columns:", df.columns.tolist())

    TEXT_CANDIDATES = [
        "specific_dialogue_content",
        "dialogue",
        "content",
        "text",
        "utterance"
    ]
    LABEL_CANDIDATES = [
        "is_fraud",
        "label",
        "fraud"
    ]

    TEXT_COL = next(c for c in TEXT_CANDIDATES if c in df.columns)
    LABEL_COL = next(c for c in LABEL_CANDIDATES if c in df.columns)

    print(f"Using TEXT_COL = {TEXT_COL}")
    print(f"Using LABEL_COL = {LABEL_COL}")

    y_true = []
    y_pred = []
    skipped = 0

    print("Running ChatGLM inference...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        dialogue = str(row[TEXT_COL])
        label = row[LABEL_COL]

        # 跳过 NaN 标签
        if pd.isna(label):
            skipped += 1
            continue

        label = int(label)
        prompt = build_prompt(dialogue)

        with torch.no_grad():
            response, _ = model.chat(
                tokenizer,
                prompt,
                max_length=MAX_LEN
            )

        pred = parse_prediction(response)

        y_true.append(label)
        y_pred.append(pred)

    print(f"\nSkipped samples due to NaN labels: {skipped}")

    # ================== 评估 ==================
    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    acc = (y_true_tensor == y_pred_tensor).float().mean().item()
    fraud_mask = y_true_tensor == 1
    nonfraud_mask = y_true_tensor == 0

    fraud_acc = (y_true_tensor[fraud_mask] == y_pred_tensor[fraud_mask]).float().mean().item() \
        if fraud_mask.sum() > 0 else 0.0
    nonfraud_acc = (y_true_tensor[nonfraud_mask] == y_pred_tensor[nonfraud_mask]).float().mean().item() \
        if nonfraud_mask.sum() > 0 else 0.0

    print("\n===== Evaluation Results =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Accuracy (Fraud): {fraud_acc:.4f}")
    print(f"Accuracy (Non-Fraud): {nonfraud_acc:.4f}")

    # ================== 保存预测结果 ==================
    df_results = df.copy()
    df_results["chatglm_pred"] = None

    pred_idx = 0
    for idx, label in enumerate(df_results[LABEL_COL]):
        if pd.isna(label):
            continue
        df_results.at[idx, "chatglm_pred"] = y_pred[pred_idx]
        pred_idx += 1

    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPredictions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()