import re
import json
import requests
from tqdm import tqdm
import time
# === SiliconFlow API 配置 ===
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3"
API_KEY = "your-api-key"  # TODO: 替换为你的密钥

def call_api(messages):
    """调用大模型 API"""
    payload = {"model": MODEL_NAME,"messages": messages,"stream": False,"max_tokens": 512,"enable_thinking": True,"thinking_budget": 4096,"min_p": 0.05,"temperature": 0.7,"top_p": 0.7,"top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "stop": []
    }
    headers = {
        "Authorization": f"Bearer sk-xhigfntcupcfpcumberfhwbgxjtbbyugsszyvnknebtinvbg",
        "Content-Type": "application/json"
    }
    response = requests.post(API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        print(f"⚠️ API调用失败: {response.status_code} - {response.text}")
        return None

def generate_negative_summaries(dialogue, reference_summary):
    prompt = f"""You are given a dialogue and a factually correct summary. Your task is to rewrite the summary in three different ways, each introducing a specific type of factual error while keeping the text fluent, natural, and consistent with the dialogue's language and style.
Dialogue:
{dialogue}
Reference Summary:
{reference_summary}
Generate three flawed summaries with the following errors:
1. **Event Misordering**: Change the sequence of actions or events compared to the dialogue.
2. **Verb Misuse**: Replace one or more key verbs to subtly alter the meaning while remaining grammatical.
3. **Missing Detail**: Omit an important factual detail present in the original summary.
Output in JSON format, ensuring each summary is a single string (no newlines within summaries). Use the same language as the reference summary.
Your output (in JSON format):
```json
{{
  "Event Misordering Summary": "...",
  "Verb Misuse Summary": "...",
  "Missing Detail Summary": "..."
}}
```"""
    result = call_api([{"role": "user", "content": prompt}])
    if result is None:
        print("⚠️ API 返回 None，跳过生成负摘要")
        return "", "", ""
    try:
        # 提取 JSON 部分
        json_match = re.search(r"""```json\n([\s\S]*?)\n```""", result, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            em = data.get("Event Misordering Summary", "").strip()
            vm = data.get("Verb Misuse Summary", "").strip()
            md = data.get("Missing Detail Summary", "").strip()

            # 检查是否所有摘要都非空
            if not em or not vm or not md:
                print(f"⚠️ 部分摘要为空: EM={em}, VM={vm}, MD={md}")
                print(f"DEBUG: 原始 API 响应:\n{result}\n")
            return em, vm, md
        else:
            print(f"⚠️ 未找到 JSON 格式，原始响应:\n{result}\n")
            return "", "", ""
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 解析失败: {e}")
        print(f"DEBUG: 原始 API 响应:\n{result}\n")
        return "", "", ""
    except Exception as e:
        print(f"⚠️ 输出解析失败: {e}")
        print(f"DEBUG: 原始 API 响应:\n{result}\n")
        return "", "", ""
def process_dataset(input_path, output_path):
    """读取包含对话与参考摘要的JSONL文件，为每条样本生成三个负摘要，并逐个写入文件"""
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing dialogues"):
            try:
                # 解析输入行
                data = json.loads(line.strip())
                dialogue = data.get("source", "")
                ref_summary = data.get("summary", "")
                sample_id = data.get("id", "")
                if not dialogue or not ref_summary:
                    print(f"⚠️ 样本 {sample_id} 数据不完整: dialogue={dialogue}, summary={ref_summary}")
                    continue
                # 调用大模型生成负摘要
                em, vm, md = generate_negative_summaries(dialogue, ref_summary)
                if not em or not vm or not md:
                    print(f"⚠️ 负摘要生成失败，跳过样本 {sample_id}")
                    continue
                # 构造样本结果
                result = {
                    "id": sample_id,
                    "text": dialogue,
                    "summary": ref_summary,
                    "neg1": em,
                    "neg2": vm,
                    "neg3": md
                }
                # 追加写入到输出文件（JSONL 格式）
                try:
                    with open(output_path, 'a', encoding='utf-8') as f_out:
                        json.dump(result, f_out, ensure_ascii=False)
                        f_out.write('\n')  # 每行一个 JSON 对象
                except Exception as e:
                    print(f"⚠️ 写入样本 {sample_id} 到 {output_path} 失败: {e}")
                    continue
                time.sleep(0.5)  # 节制请求频率，避免限速
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON 解析错误: {e}, 跳过该行")
                continue
            except Exception as e:
                print(f"⚠️ 处理样本 {sample_id} 时出错: {e}")
                continue
    print(f"\n✅ 所有样本处理完成，结果保存至：{output_path}")

if __name__ == "__main__":
    input_jsonl = "test.jsonl"  # 格式: 每行含 "text" 和 "summary"
    output_json = "test_neg.jsonl"
    process_dataset(input_jsonl, output_json)

