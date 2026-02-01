#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并高质量摘要和低质量摘要
将 train_lowneg.jsonl 中的负摘要添加到 samsum_train_neg_events.json 对应样本中
"""

import json
import os
import random

def read_jsonl(file_path):
    """读取 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"警告: 跳过无效的JSON行: {e}")
                    continue
    return data

def save_jsonl(data, file_path):
    """保存为 JSONL 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def normalize_summary(sentences):
    """
    将句子列表合并为单个摘要字符串
    处理多余空格，统一格式
    """
    if isinstance(sentences, list):
        # 合并句子，并清理多余空格
        summary = ' '.join(sentences)
    else:
        summary = str(sentences)
    
    # 统一处理：去除首尾空格，将多个连续空格替换为单个空格
    summary = ' '.join(summary.split())
    return summary

def merge_negatives(
    lowneg_file='train_lowneg.jsonl',
    samsum_file='samsum_train_neg_events.json',
    output_file='samsum_train_neg_events_updated.json'
):
    """
    主函数：合并负摘要
    
    Args:
        lowneg_file: 包含低质量负摘要的 JSONL 文件路径
        samsum_file: 原始 samsum 数据文件路径
        output_file: 输出文件路径
    """
    print("="*70)
    print("开始合并负摘要...")
    print("="*70)
    
    # 检查文件是否存在
    if not os.path.exists(lowneg_file):
        print(f"错误: 文件 {lowneg_file} 不存在！")
        return
    if not os.path.exists(samsum_file):
        print(f"错误: 文件 {samsum_file} 不存在！")
        return
    
    # 读取数据（两个文件都是 JSONL 格式）
    print(f"\n1. 读取文件...")
    train_lowneg = read_jsonl(lowneg_file)
    samsum_data = read_jsonl(samsum_file)  # 也使用 read_jsonl
    print(f"   - {lowneg_file}: {len(train_lowneg)} 条记录")
    print(f"   - {samsum_file}: {len(samsum_data)} 条记录")
    
    # 提取负摘要，构建字典：original_summary -> [generated_summaries]
    print(f"\n2. 提取负摘要...")
    neg_summary_dict = {}
    
    for item in train_lowneg:
        if item.get('label') == 0:  # 只处理负样本
            # 合并 original_summary_sentences 为完整摘要
            original_summary = normalize_summary(item['original_summary_sentences'])
            generated_summary = normalize_summary(item['generated_summary_sentences'])
            
            if original_summary not in neg_summary_dict:
                neg_summary_dict[original_summary] = []
            neg_summary_dict[original_summary].append(generated_summary)
    
    print(f"   - 提取到 {len(neg_summary_dict)} 个不同的原始摘要对应的负样本")
    total_neg_samples = sum(len(v) for v in neg_summary_dict.values())
    print(f"   - 总共 {total_neg_samples} 个负摘要")
    
    # 遍历 samsum_data，为每个样本添加对应的负摘要
    print(f"\n3. 匹配并添加负摘要到 samsum 数据（每个样本随机选择3个负摘要）...")
    matched_count = 0
    unmatched_count = 0
    total_added_negs = 0
    
    # 设置随机种子以保证可重复性（可选）
    random.seed(42)
    
    for idx, sample in enumerate(samsum_data):
        # 获取当前样本的 summary 并标准化
        current_summary = normalize_summary(sample.get('summary', ''))
        
        # 尝试匹配
        if current_summary in neg_summary_dict:
            # 找到匹配的负摘要
            all_neg_summaries = neg_summary_dict[current_summary]
            
            # 随机选择3个负摘要
            if len(all_neg_summaries) >= 3:
                # 如果有3个或以上，随机选择3个（不重复）
                selected_negs = random.sample(all_neg_summaries, 3)
            else:
                # 如果不足3个，先全部选择，然后重复补足到3个
                selected_negs = all_neg_summaries.copy()
                while len(selected_negs) < 3:
                    # 随机选择一个已有的负摘要重复添加
                    selected_negs.append(random.choice(all_neg_summaries))
            
            # 从 neg4 开始添加（因为已经有 neg1, neg2, neg3）
            for i, neg_summary in enumerate(selected_negs):
                neg_key = f'neg{i + 4}'  # neg4, neg5, neg6
                sample[neg_key] = neg_summary
                total_added_negs += 1
            
            matched_count += 1
            if matched_count <= 3:  # 显示前3个匹配示例
                print(f"   - 样本 #{idx}: 从 {len(all_neg_summaries)} 个负摘要中随机选择了 3 个 (neg4, neg5, neg6)")
        else:
            unmatched_count += 1
    
    # 保存更新后的数据（保存为 JSONL 格式）
    print(f"\n4. 保存更新后的数据...")
    save_jsonl(samsum_data, output_file)
    
    # 打印统计信息
    print("\n" + "="*70)
    print("处理完成！")
    print("="*70)
    print(f"✓ 成功匹配的样本数: {matched_count}")
    print(f"✗ 未匹配的样本数: {unmatched_count}")
    print(f"✓ 总共添加的负摘要数: {total_added_negs}")
    print(f"✓ 平均每个匹配样本添加: {total_added_negs/matched_count:.2f} 个负摘要" if matched_count > 0 else "")
    print(f"\n更新后的文件已保存为: {output_file}")
    print("="*70)
    
    # 如果有大量未匹配，给出警告
    if unmatched_count > matched_count:
        print(f"\n⚠️ 警告: 未匹配样本数 ({unmatched_count}) 大于匹配样本数 ({matched_count})")
        print("   可能的原因:")
        print("   1. summary 字段格式不一致")
        print("   2. 句子拼接时的空格或标点符号差异")
        print("   3. 两个文件的数据集不匹配")
        
        # 显示一些未匹配的样本用于调试
        print("\n   未匹配样本示例（前3个）:")
        count = 0
        for sample in samsum_data:
            current_summary = normalize_summary(sample.get('summary', ''))
            if current_summary not in neg_summary_dict and count < 3:
                print(f"   - {current_summary[:100]}...")
                count += 1

if __name__ == '__main__':
    # 设置文件路径（相对于当前脚本的路径）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    lowneg_file = os.path.join(script_dir, 'train_lowneg.jsonl')
    samsum_file = os.path.join(script_dir, 'samsum_train_neg_events.json')
    output_file = os.path.join(script_dir, 'samsum_train_neg_events_updated.json')
    
    # 执行合并
    merge_negatives(lowneg_file, samsum_file, output_file)
