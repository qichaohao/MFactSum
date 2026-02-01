import json
import os
import spacy
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

# 加载 SpaCy 英文模型
nlp = spacy.load("en_core_web_sm")

# 创建输出目录
os.makedirs("samsum", exist_ok=True)


def mask_entity(jsonl_file, output_file):
    """
    对摘要中的命名实体（PERSON, ORG, GPE）进行掩码处理，生成 JSONL 格式的输出。
    输出格式：
    {
        "source_article_sentences": [...],
        "original_summary_sentences": [...],
        "masked_sent": "...",
        "target": "...",
        "original_sent": "..."
    }
    """
    # 读取输入 JSONL 文件
    with open(jsonl_file, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    with open(output_file, "a", encoding='utf-8') as wf:
        for item in tqdm(lines, desc="Generating entity-masked samples"):
            source_text = item.get("source", "")
            summary_text = item.get("summary", "")
            source_sents = sent_tokenize(source_text)
            summary_sents = sent_tokenize(summary_text)

            # 处理每个摘要句子
            for sent in summary_sents:
                doc = nlp(sent)
                # 掩码命名实体（PERSON, ORG, GPE）
                for ent in doc.ents:
                    if ent.label_ in {"PERSON", "ORG", "GPE"}:
                        masked_sent = sent.replace(ent.text, "[MASK]", 1)
                        example = {
                            "source_article_sentences": source_sents,
                            "original_summary_sentences": summary_sents,
                            "masked_sent": masked_sent,
                            "target": ent.text,
                            "original_sent": sent
                        }
                        json.dump(example, wf, ensure_ascii=False)
                        wf.write("\n")


def mask_pronoun(jsonl_file, output_file):
    """
    对摘要中的代词进行掩码处理，生成 JSONL 格式的输出。
    输出格式：
    {
        "source_article_sentences": [...],
        "original_summary_sentences": [...],
        "masked_sent": "...",
        "target": "...",
        "original_sent": "..."
    }
    """
    # 读取输入 JSONL 文件
    with open(jsonl_file, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    with open(output_file, "a", encoding='utf-8') as wf:
        for item in tqdm(lines, desc="Generating pronoun-masked samples"):
            source_text = item.get("source", "")
            summary_text = item.get("summary", "")
            source_sents = sent_tokenize(source_text)
            summary_sents = sent_tokenize(summary_text)

            # 处理每个摘要句子
            for sent in summary_sents:
                doc = nlp(sent)
                # 掩码代词
                pronoun_whitelist = {
                    "I", "me", "you", "he", "him", "she", "her", "it",
                    "we", "us", "they", "them", "mine", "yours", "his",
                    "hers", "ours", "theirs", "this"
                }
                pronouns = [(token.text, token.idx) for token in doc if
                            token.pos_ == "PRON" and token.text.lower() in pronoun_whitelist]

                for pronoun, start_idx in pronouns:
                    end_idx = start_idx + len(pronoun)
                    masked_sent = sent[:start_idx] + "[MASK]" + sent[end_idx:]
                    example = {
                        "source_article_sentences": source_sents,
                        "original_summary_sentences": summary_sents,
                        "masked_sent": masked_sent,
                        "target": pronoun,
                        "original_sent": sent
                    }
                    json.dump(example, wf, ensure_ascii=False)
                    wf.write("\n")


if __name__ == "__main__":
    jsonl_file = "samsum/valid.jsonl"  # 改为你的路径
    output_file = "samsum/infill_test_valid.jsonl"

    # 清空输出文件（如果存在）
    if os.path.exists(output_file):
        os.remove(output_file)

    # 分别调用实体掩码和代词掩码
    mask_entity(jsonl_file, output_file)
    mask_pronoun(jsonl_file, output_file)