import re
import json
import requests
from tqdm import tqdm
import time
import os
import shutil
from typing import List, Optional

# === SiliconFlow API 配置 ===
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3"
API_KEY = ""  # TODO: 替换为你的密钥

# ====== 可配置项 ======
MAX_RETRIES = 10               # 每个样本最多重试次数（请求+解析）
RETRY_BACKOFF = 1.0           # 重试基础等待秒数（指数退避基数）
SLEEP_BETWEEN_REQUESTS = 0.5  # 每次请求后等待，防止速率过高
TEMP_INPUT_SUFFIX = ".tmp"    # 临时输入文件后缀

# ===============================
# 🔧 JSON 自动修复函数（稳健版）
# ===============================
def fix_json_string(text: str) -> str:
    """
    尽量保留原始合法 JSON，仅在格式错误时进行最小化修复。
    """
    if not isinstance(text, str):
        return text

    # 1. 去除首尾和 markdown 包裹
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # 2. 如果已经能解析，直接返回（优先保持原样）
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # 3. 尝试提取最外层 JSON 数组主体
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        text = m.group(0)

    # 4. 替换常见“智能引号”和特殊不可见字符
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    # 去掉零宽字符、行分隔符、BOM、非断空格等
    text = re.sub(r"[\u200b-\u200f\u2028\u2029\u00a0\uFEFF]", "", text)

    # 5. 修复少数键名没有引号的情况（谨慎处理）
    #    只在键和值附近存在冒号时添加引号（避免破坏正常字符串）
    text = re.sub(r'(?<=\{|\s)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'"\1":', text)

    # 6. 将单引号包裹的 value -> 双引号（谨慎）
    #    只替换形如 '...'
    text = re.sub(r"\'([^']*?)\'", r'"\1"', text)

    # 7. 删除多余的逗号，如 ,] 或 ,}
    text = re.sub(r",\s*(\]|\})", r"\1", text)

    # 8. 去掉不可见控制字符并紧缩多余空白（保留正常空格）
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 9. 最后尝试解析，若失败做最小兜底处理
    try:
        json.loads(text)
    except Exception:
        # 兜底：移除反斜杠转义字符并压缩空格
        text = text.replace("\\", "")
        text = re.sub(r"\s+", " ", text).strip()

    return text.strip()

# ===============================
# 🔧 调用大模型 API
# ===============================
def call_api(messages: List[dict]) -> Optional[str]:
    """调用大模型 API，返回文本内容或 None"""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "max_tokens": 512,
        "min_p": 0.05,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "stop": []
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    except Exception as e:
        print(f"⚠️ 网络请求异常: {e}")
        return None

    if response.status_code == 200:
        try:
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"⚠️ 解析 API 返回 JSON 失败: {e}")
            return None
    else:
        print(f"⚠️ API调用失败: {response.status_code} - {response.text}")
        return None

# ===============================
# 🔧 事件提取主函数（含重试）
# ===============================
def extract_events(dialogue: str, max_retries: int = MAX_RETRIES) -> List[dict]:
    """调用大模型生成 1-4 个关键事件，出现解析错误时重试。返回事件列表或空列表"""
    prompt = f"""Role:
You are an expert in dialogue event extraction.

Task:
From the following dialogue, extract ONLY the main actions as structured events. 
You must output **no more than four (≤4)** events. 
If the dialogue is very short, output fewer events (1–3). 
Never output more than four events under any circumstances.

Output requirements:
1. Output a **valid JSON array**.
2. Each event must be a dictionary of the form:
   {{"action": "...", "subject": "...", "object": "..."}}
3. If "object" is not applicable, omit it completely.
4. Do NOT invent or infer information not explicitly in the dialogue.
5. Do NOT include duplicates, emotions, or thoughts as separate events.
6. The JSON array must contain **between 1 and 4** events — never more.

Example dialogue:
Alice: I can't find my phone charger.
Bob: Did you check the living room?
Alice: Yes, but it's not there.

Example output:
[
  {{"action": "can't find", "subject": "Alice", "object": "phone charger"}},
  {{"action": "check", "subject": "Bob", "object": "living room"}}
]

Now extract 1–4 key events from this dialogue:
{dialogue}
"""
    messages = [{"role": "user", "content": prompt}]

    attempt = 0
    last_error_info = None

    while attempt < max_retries:
        attempt += 1
        if attempt > 1:
            wait = RETRY_BACKOFF * (2 ** (attempt - 2))
            print(f"⏳ 重试第 {attempt} 次，将在 {wait:.1f}s 后请求...")
            time.sleep(wait)

        result = call_api(messages)
        if result is None:
            last_error_info = "API returned None"
            continue

        # 小间隔，避免速率太高
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        # 自动修复 JSON 字符串
        fixed_result = fix_json_string(result)

        # 深度清理不可见字符并尝试解析
        try:
            cleaned = re.sub(r"[\u0000-\u001F\u007F-\u009F\u200B-\u200F\u2028-\u202F\uFEFF]", "", fixed_result)
            cleaned = cleaned.encode("utf-8", "ignore").decode("utf-8", "ignore")
            events = json.loads(cleaned)

            # 验证类型和长度
            if not isinstance(events, list):
                last_error_info = "Parsed JSON is not a list"
                # 尝试下一次重试
                continue

            # 限制事件数量 1–4
            events = events[:4]
            # 过滤掉不完整的事件（没有 action 或 subject）
            filtered = []
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                action = ev.get("action")
                subject = ev.get("subject")
                if action and subject:
                    filtered.append(ev)
            return filtered

        except Exception as e:
            last_error_info = str(e)
            # 记录本次失败响应方便调试
            with open("json_error_log.txt", "a", encoding="utf-8") as log:
                log.write(f"\n==== Attempt {attempt} Failed ====\n")
                log.write("Raw result:\n")
                log.write(result + "\n")
                log.write("Fixed result:\n")
                log.write(fixed_result + "\n")
                try:
                    log.write("Cleaned:\n")
                    log.write(cleaned + "\n")
                except Exception:
                    pass
                log.write(f"Error: {e}\n\n")
            # 尝试下一次重试
            continue

    # 超过重试次数仍失败，记录到失败日志
    with open("failed_samples.log", "a", encoding="utf-8") as flog:
        flog.write(f"Failed to extract after {max_retries} attempts. Last error: {last_error_info}\nDialogue:\n{dialogue}\n\n")
    return []

# ===============================
# 🔧 文件处理工具函数
# ===============================
def atomic_write_lines(filepath: str, lines: List[str]):
    """
    原子地写入行到文件（写入 temp 然后替换）。
    """
    tmp_path = filepath + TEMP_INPUT_SUFFIX
    with open(tmp_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln if ln.endswith("\n") else ln + "\n")
    # 在 Windows 上，os.replace 也可以原子地替换文件
    os.replace(tmp_path, filepath)

# ===============================
# 🔧 数据集批处理（主流程）
# ===============================
def process_dataset(input_path: str, output_path: str):
    """
    读取 JSONL 文件，为每条样本增加事件属性。每成功写入一条输出后，
    会将该样本从输入文件中删除（通过重写剩余行实现）。
    """
    # 读取所有行到内存（注意：如果文件很大，这一步会消耗内存）
    with open(input_path, "r", encoding="utf-8") as fi:
        lines = [ln.rstrip("\n") for ln in fi]

    total = len(lines)
    print(f"🔁 载入 {total} 条样本，开始处理...")

    # 逐条处理（在成功写入 output 后，从 lines 中移除并重写 input 文件）
    # 为了便于进度显示，我们使用 while 循环取第一条处理
    processed_count = 0
    idx = 0
    # tqdm 只用于外层展示剩余条数
    pbar = tqdm(total=total, desc="Processing dialogues")
    while lines:
        # 取队首
        raw_line = lines.pop(0)
        pbar.update(1)
        idx += 1

        if not raw_line.strip():
            # 空行跳过
            continue
        try:
            data = json.loads(raw_line)
        except Exception as e:
            print(f"⚠️ JSON 解析错误（输入文件）: {e}，跳过该行")
            # 记录并继续（不要删除原始行，因为我们已经从 lines pop 了）
            with open("failed_samples.log", "a", encoding="utf-8") as flog:
                flog.write(f"Input parse error: {e}\nLine: {raw_line}\n\n")
            # 因为此行已 pop，我们不想保留它，继续下一条
            # 如果你想保留原始错误行到输入文件，请改成 append 到另一个文件
            total -= 1
            continue

        dialogue = data.get("text", "")
        sample_id = data.get("id", "")
        if not dialogue:
            print(f"⚠️ 样本 {sample_id} 对话为空，跳过")
            # 不写入输出，输入文件中直接删除该行（即已 pop）
            continue

        # 提取事件（含重试）
        events = extract_events(dialogue, max_retries=MAX_RETRIES)
        if not events:
            print(f"⚠️ 样本 {sample_id} 事件提取失败，跳过（已记录）")
            # 事件提取失败：我们选择 **保留该行** 在输入文件中以便后续重试或人工检查。
            # 因为我们已经 pop 了该行，为了“保留”，我们把它追加到 lines 的末尾
            lines.append(raw_line)
            # 等待一会儿再继续，避免频繁再请求
            time.sleep(0.2)
            continue

        # 成功：将事件写入 data 并写入输出文件
        data["events"] = events
        try:
            with open(output_path, "a", encoding="utf-8") as fout:
                json.dump(data, fout, ensure_ascii=False)
                fout.write("\n")
        except Exception as e:
            print(f"⚠️ 写入输出文件失败: {e}，将样本重新放回输入队列")
            # 写失败则把样本放回队列以便稍后重试
            lines.append(raw_line)
            continue

        processed_count += 1
        print(f"✅ 样本 {sample_id} 处理成功并写入输出（已处理 {processed_count} 条）")

        # 成功写入后：**立即重写输入文件，去掉已处理的那条**
        # (此时 lines 列表已经不含已处理项：pop 后没 append)
        try:
            atomic_write_lines(input_path, lines)
        except Exception as e:
            # 如果重写输入文件失败，记录日志但继续处理（此错误不影响已经写入的输出）
            print(f"⚠️ 重写输入文件失败: {e}。请手动检查并删除已处理行。")
            with open("failed_samples.log", "a", encoding="utf-8") as flog:
                flog.write(f"Failed to rewrite input file after processing sample {sample_id}: {e}\n")

        # 稍微等一下，防止过快请求
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    pbar.close()
    print(f"\n✅ 处理完成：已成功写入 {processed_count} 条样本到 {output_path}")

# ===============================
# 🔧 主入口
# ===============================
if __name__ == "__main__":


    input_jsonl1 = "dialogsum_valid.jsonl"
    output_jsonl1 = "dialogsum_valid_events.jsonl"
    process_dataset(input_jsonl1, output_jsonl1)
    input_jsonl2 = "dialogsum_test.jsonl"
    output_jsonl2 = "dialogsum_test_events.jsonl"
    process_dataset(input_jsonl2, output_jsonl2)
