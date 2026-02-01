import json
import re
import random

from tqdm import tqdm
import os
import spacy
from nltk.tokenize import sent_tokenize

# 加载 spaCy 英文模型（可根据需求换成 en_core_web_trf）
nlp = spacy.load("en_core_web_sm")  # 加载spaCy的英文小模型，用于处理文本的NLP任务


def gen_pronoun_mask_spacy(text):
    """
    使用 spaCy 识别代词，并为每个代词的每次出现生成独立的遮盖样本。
    确保相同代词的不同位置被分别遮盖。
    """
    mask_samples = []  # 用于存储生成的遮盖样本
    # 按 \r\n 分割句子，并清理空句子
    sents = [s.strip() for s in text.split("\r\n") if s.strip()]
    # 遍历每个句子
    for sent in sents:
        # 使用 spaCy 解析每个句子
        sent_doc = nlp(sent)

        pronoun_whitelist = {"I", "me", "you", "he", "him", "she", "her", "it", "we", "us", "they", "them", "mine",
                             "yours", "his", "hers", "ours", "theirs", "this"}
        pronouns = [(token.text, token.idx) for token in sent_doc if
                    token.pos_ == "PRON" and token.text.lower() in pronoun_whitelist]

        # 如果句子中有代词
        if pronouns:
            # 遍历每个代词及其位置
            for pronoun, start_idx in pronouns:
                # 初始化原始句子
                masked_sent = sent

                # 计算代词的结束位置
                end_idx = start_idx + len(pronoun)

                # 直接基于位置替换代词为 [MASK]
                masked_sent = masked_sent[:start_idx] + "[MASK]" + masked_sent[end_idx:]

                # 上下文为其他句子
                ctx_list = [s for s in sents if s != sent]
                ctx = " ".join(ctx_list)  # 将上下文合并为一个字符串

                # 生成样本
                mask_samples.append({
                    "source": ctx,  # 上下文：不包含 [MASK] 的句子
                    "masked_sent": masked_sent,  # 包含 [MASK] 的句子
                    "target": pronoun,  # 被替换的代词
                    "mask_type": "pronoun"  # 标记为代词遮盖
                })

    return mask_samples  # 返回生成的遮盖样本


def gen_entity_mask(doc, mask_ratio=1):
    """
    使用 spaCy 的 NER 识别命名实体（如人名、组织、地点）并生成遮盖样本。
    按 \r\n 分割句子，尊重输入文本的预分割结构。
    """
    mask_sents = []
    sents = [s.strip() for s in doc.split("\r\n") if s.strip()]

    doc_nlp = nlp(doc)

    # 只遮盖命名实体（PERSON, ORG, GPE）
    ents = [ent for ent in doc_nlp.ents if ent.label_ in {"PERSON", "ORG", "GPE"}]
    # 随机选择部分实体（按 mask_ratio）
    ents = random.sample(ents, max(1, int(len(ents) * mask_ratio))) if ents else []

    for ent in ents:
        masked_doc = doc.replace(ent.text, '[MASK]', 1)
        masked_doc_sentlist = [s.strip() for s in masked_doc.split("\r\n") if s.strip()]

        ctx_list = []
        masked_sent = None
        for sent in masked_doc_sentlist:
            if "[MASK]" in sent:
                masked_sent = sent
            else:
                ctx_list.append(sent)

        if masked_sent:
            mask_sents.append({
                "source": " ".join(ctx_list),
                "target": ent.text,
                "masked_sent": masked_sent,
                "mask_type": "entity"
            })

    return mask_sents


def gen_num_mask(doc):
    """
    使用正则表达式识别数字并生成数字遮盖样本。
    按 \r\n 分割句子，尊重输入文本的预分割结构。
    """
    mask_sents = []
    # 按 \r\n 分割句子，并清理空句子
    sents = [s.strip() for s in doc.split("\r\n") if s.strip()]

    for sid, sent in enumerate(sents):
        # 上下文为其他句子
        ctx_list = [s for i, s in enumerate(sents) if i != sid]
        ctx = " ".join(ctx_list)

        # 使用正则表达式匹配数字
        re_number = re.finditer(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', sent)
        for match in re_number:
            start, end = match.start(), match.end()
            target = sent[start:end]
            masked_sent = sent[:start] + '[MASK]' + sent[end:]
            mask_sents.append({
                "source": ctx,
                "target": target,
                "masked_sent": masked_sent,
                "mask_type": "number"
            })

    return mask_sents

MASK_ENTITY = True
MASK_NUM = False
MASK_PRONOUN = True

if __name__ == '__main__':
    input_file = 'samsum/valid.jsonl'  # 定义输入文件路径，格式为JSON Lines，包含source字段的对话数据
    output_file = 'samsum/infill_valid.jsonl'  # 定义统一的输出文件路径
    os.makedirs("samsum", exist_ok=True)  # 创建输出目录，如果已存在则不报错

    train_data = []
    with open(input_file, encoding='utf-8') as rf:
        for line in rf:
            data = json.loads(line)
            train_data.append(data["source"])

    with open(output_file, 'w', encoding='utf-8') as wf:
        # 实体遮盖
        if MASK_ENTITY:
            for doc in tqdm(train_data, desc="Generating entity-masked samples"):
                mask_sents = gen_entity_mask(doc)
                if not doc.strip():
                    print(f"Warning: Empty source text found.")
                for mask_sent in mask_sents:
                    json.dump(mask_sent, wf, ensure_ascii=False)
                    wf.write("\n")

        # 数字遮盖
        if MASK_NUM:
            for doc in tqdm(train_data, desc="Generating number-masked samples"):
                mask_sents = gen_num_mask(doc)
                if not doc.strip():
                    print(f"Warning: Empty source text found.")
                for mask_sent in mask_sents:
                    json.dump(mask_sent, wf, ensure_ascii=False)
                    wf.write("\n")

        # ========= 指代遮盖（共指消解） =========
        if MASK_PRONOUN:
            for doc in tqdm(train_data, desc="Generating pronoun-masked samples"):
                mask_sents = gen_pronoun_mask_spacy(doc)
                if not doc.strip():
                    print(f"Warning: Empty source text found.")
                for mask_sent in mask_sents:
                    json.dump(mask_sent, wf, ensure_ascii=False)
                    wf.write("\n")
