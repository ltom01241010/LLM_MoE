import torch
import numpy as np
import json
import copy
from transformers import OlmoeForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from collections import Counter
from tqdm import tqdm

# =========================
# Monkey-Patch OlmoeSparseMoeBlock.forward 方法，使其支持通过模块属性使用自定义 routing weights
# =========================

try:
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
except ImportError:
    raise ImportError("无法导入 OlmoeSparseMoeBlock，请检查 transformers 版本。")

def custom_moe_forward(self, hidden_states: torch.Tensor):
    """
    自定义 MoE 模块的 forward 方法。
    计算 routing weights 后，如果模块属性 self.custom_routing_weights 被设置，
    则将每个样本最后 token 的 routing weights 替换为自定义的概率分布。
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_flat = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states_flat)
    routing_weights = torch.nn.functional.softmax(router_logits, dim=1)
    
    # 如果设置了自定义 routing 权重，则覆盖每个样本最后一个 token 的 routing weights
    if self.custom_routing_weights is not None:
        # self.custom_routing_weights 应为字典，例如 {"expert_0": weight0, "expert_1": weight1, ...}
        custom_weights_list = [self.custom_routing_weights.get(f"expert_{i}", 1e-10) for i in range(self.num_experts)]
        custom_weights_tensor = torch.tensor(custom_weights_list, dtype=routing_weights.dtype, device=routing_weights.device)
        custom_weights_tensor = custom_weights_tensor / custom_weights_tensor.sum()
        # 替换每个样本最后一个 token 的 routing weights（计算 flat tensor 中的对应索引）
        indices = [i * sequence_length + (sequence_length - 1) for i in range(batch_size)]
        routing_weights[indices] = custom_weights_tensor.unsqueeze(0).expand(batch_size, -1)
    
    routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=1)
    if self.norm_topk_prob:
        routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=1, keepdim=True)
    routing_weights_topk = routing_weights_topk.to(hidden_states_flat.dtype)

    final_hidden_states = torch.zeros_like(hidden_states_flat)
    # 对选中的专家进行 one-hot 编码，维度为 [num_experts, tokens, 1]
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, token_indices = torch.where(expert_mask[expert_idx])
        if token_indices.numel() == 0:
            continue
        current_state = hidden_states_flat[token_indices]
        current_hidden = expert_layer(current_state) * routing_weights_topk[token_indices, idx].unsqueeze(-1)
        final_hidden_states.index_add_(0, token_indices, current_hidden)
    final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits

# 替换原有 forward 方法
OlmoeSparseMoeBlock.forward = custom_moe_forward

# =========================
# 辅助函数
# =========================

def extract_answer_option(text):
    """
    从文本中提取选项字母（例如 "A", "B", "C", "D"）。
    首先尝试查找 "Answer:" 后面的选项，如果没有，则查找文本中第一个出现的选项字母。
    如果无法提取，则返回空字符串。
    """
    idx = text.find("Answer:")
    if idx != -1:
        substring = text[idx+len("Answer:"):].strip()
        for token in substring.split():
            token = token.strip().upper()
            if token and token[0] in "ABCD":
                return token[0]
    for token in text.split():
        token = token.strip().upper()
        if token in ["A", "B", "C", "D"]:
            return token
        elif token.startswith(("A.", "B.", "C.", "D.")) and len(token) <= 3:
            return token[0]
        elif token.startswith(("A)", "B)", "C)", "D)")) and len(token) <= 3:
            return token[0]
    import re
    pattern = r'\b([A-D])[\.|\)]?\b'
    matches = re.findall(pattern, text.upper())
    if matches:
        return matches[0]
    return ""

def extract_routing_info(text, model, tokenizer, batch_size=512, max_length=64):
    """
    根据输入文本提取每个 token 的 routing 信息。
    这里只提取最后一个 token 的 routing 信息，并存放在 last_token_routing 字段中，
    同时生成输出文本（max_length 设为64）。
    """
    tokens = tokenizer(text, truncation=False)["input_ids"]
    expert_counter = Counter()
    last_token_routing = None

    with torch.no_grad():
        for i in tqdm(range(0, len(tokens), batch_size), desc="Processing tokens"):
            batch_tokens = tokens[i:min(i + batch_size, len(tokens))]
            input_ids = torch.tensor(batch_tokens).reshape(1, -1).to(model.device)
            outputs = model(input_ids=input_ids, output_router_logits=True)
            all_layers_logits = outputs["router_logits"]
            all_layers_probs = [
                torch.nn.functional.softmax(layer_logits.float(), dim=-1).cpu().numpy()
                for layer_logits in all_layers_logits
            ]
            for layer_probs in all_layers_probs:
                top_experts = np.argsort(-layer_probs, axis=-1)
                expert_counter.update(top_experts.flatten().tolist())

        last_token_idx = len(tokens) - 1
        layers_info = []
        n_layers = len(all_layers_probs)
        for layer_idx in range(n_layers):
            token_position = last_token_idx % batch_size
            distribution = all_layers_probs[layer_idx][token_position]
            sorted_experts = sorted(enumerate(distribution), key=lambda x: x[1], reverse=True)
            layer_info = {
                "layer": layer_idx,
                "routing_weights": {
                    f"expert_{expert_id}": float(weight)
                    for expert_id, weight in sorted_experts
                }
            }
            layers_info.append(layer_info)
        last_token_routing = {
            "token_id": tokens[last_token_idx],
            "token_text": tokenizer.decode([tokens[last_token_idx]]),
            "layers": layers_info
        }

    input_ids = torch.tensor(tokens).reshape(1, -1).to(model.device)
    generated_output = model.generate(input_ids=input_ids, max_length=max_length)
    generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

    return {
        'last_token_routing': last_token_routing,
        'expert_counts': {int(k): int(v) for k, v in expert_counter.items()},
        'tokens': tokens,
        'tokenizer': tokenizer.name_or_path,
        'generated_text': generated_text,
        'input_text': text
    }

def save_results_to_json(results, filename="all_results.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {filename}")

def print_analysis_results(results):
    total_selections = sum(results['expert_counts'].values())
    print("\nExpert utilization rates:")
    for expert_id, count in sorted(results['expert_counts'].items()):
        percentage = count / total_selections * 100
        print(f"Expert {expert_id}: {percentage:.2f}%")
    print("\nLast token routing weights for each layer:")
    last_routing = results.get("last_token_routing", {})
    if last_routing:
        print(f"\nToken ID: {last_routing['token_id']}, Text: '{last_routing['token_text']}'")
        for layer_info in last_routing.get("layers", []):
            print(f"  Layer {layer_info['layer']}:")
            for expert_id, weight in layer_info['routing_weights'].items():
                print(f"    {expert_id}: {weight:.4f}")
    print("\nGenerated Text:")
    print(results['generated_text'])

def load_multiple_reference_files(file_paths):
    """
    加载多个参考数据集文件并合并为一个参考集。
    """
    combined_references = {}
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
                print(f"从 {file_path} 加载了 {len(dataset)} 个参考案例")
                prefix = file_path.split("_")[1].split(".")[0]
                prefixed_dataset = {f"{prefix}_{key}": value for key, value in dataset.items()}
                combined_references.update(prefixed_dataset)
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
    print(f"总共加载了 {len(combined_references)} 个参考案例")
    return combined_references

def detect_custom_routing_usage(model):
    """
    检测模型本次生成中是否使用了自定义 routing weights（这里可根据实际需求自定义检测逻辑）。
    """
    # 示例：检查第一个 decoder layer 的 MoE 模块是否设置了 custom_routing_weights
    if hasattr(model.model.layers[0].mlp, 'custom_routing_weights') and model.model.layers[0].mlp.custom_routing_weights is not None:
        print("检测到模型使用了自定义 routing weights。")
        return True
    else:
        print("未检测到模型使用自定义 routing weights。")
        return False

def re_infer_case(case, reference_cases, embedder, model, tokenizer, max_length=64):
    """
    对传入的 case 重新更新 routing 权重（更新前 5 层），生成新文本，
    并输出问题、原始推理答案、正确答案、邻居案例信息以及优化后的推理结果等信息。
    """
    result = {}
    question = case.get("input_text", "")
    result["question"] = question
    result["correct_answer"] = case.get("correct_answer", "N/A").strip().upper()
    result["model_answer"] = case.get("model_answer", "N/A").strip().upper()
    result["is_correct"] = case.get("is_correct", None)
    result["original_inference"] = result["model_answer"]
    result["original_output_text"] = case.get("generated_text", "")

    # 获取当前问题 embedding
    case_embedding = embedder.encode(question, convert_to_tensor=True)

    # 邻居检索
    ref_questions = []
    ref_keys = []
    for key, one_case in reference_cases.items():
        q_text = one_case.get("input_text", "")
        ref_questions.append(q_text)
        ref_keys.append(key)
    ref_embeddings = embedder.encode(ref_questions, convert_to_tensor=True)
    cosine_scores = util.cos_sim(case_embedding, ref_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=3)
    top_indices = top_results[1].tolist()
    top_scores = top_results[0].tolist()

    neighbors = []
    print(f"\n处理 case 问题：\n  {question}")
    for idx, score in zip(top_indices, top_scores):
        similar_key = ref_keys[idx]
        similar_question = reference_cases[similar_key].get("input_text", "")
        neighbor_correct = reference_cases[similar_key].get("correct_answer", "N/A").strip().upper()
        neighbors.append({
            "case_id": similar_key,
            "question": similar_question,
            "correct_answer": neighbor_correct,
            "similarity": float(score)
        })
        print(f"  邻居 case {similar_key}: 相似度 {score:.4f}, 问题：{similar_question}  正确答案：{neighbor_correct}")
    result["neighbors"] = neighbors

    # 保存原始前 5 层 routing 权重
    routing_info = case.get("last_token_routing", None)
    if routing_info is None:
        print("当前 case 没有 last_token_routing，跳过。")
        return None
    original_routing = {"layers": copy.deepcopy(routing_info.get("layers", [])[:5])}
    result["original_routing_first5"] = original_routing["layers"]

    layers = routing_info.get("layers", [])
    if len(layers) < 5:
        print("当前 case 的层数不足 5 层，跳过。")
        return None

    # 按邻居相似度加权更新前 5 层 routing 权重
    updated_layers = []
    for layer in layers[:5]:
        layer_idx = layer["layer"]
        weighted_sum = None
        total_weight = 0.0
        for sim_score, idx in zip(top_scores, top_indices):
            neighbor_key = ref_keys[idx]
            neighbor_case = reference_cases[neighbor_key]
            neighbor_routing = neighbor_case.get("last_token_routing", None)
            if neighbor_routing is None:
                continue
            neighbor_layers = neighbor_routing.get("layers", [])
            neighbor_layer_info = next((l for l in neighbor_layers if l["layer"] == layer_idx), None)
            if neighbor_layer_info is None:
                continue
            weights = np.array([neighbor_layer_info["routing_weights"].get(f"expert_{i}", 0.0)
                                 for i in range(64)], dtype=np.float32)
            if weighted_sum is None:
                weighted_sum = sim_score * weights
            else:
                weighted_sum += sim_score * weights
            total_weight += sim_score
        if weighted_sum is not None and total_weight > 0:
            new_weights = weighted_sum / total_weight
            new_weights = new_weights / new_weights.sum()
            new_weights_dict = {f"expert_{i}": float(new_weights[i]) for i in range(64)}
            sorted_new_weights = dict(sorted(new_weights_dict.items(), key=lambda item: item[1], reverse=True))
            layer["routing_weights"] = sorted_new_weights
            print(f"更新当前 case 前 5 层中 layer {layer_idx} 的 routing weights。")
        else:
            print(f"无法更新当前 case 的 layer {layer_idx}（缺少对应数据）。")
        updated_layers.append(copy.deepcopy(layer))
    result["updated_routing_first5"] = updated_layers

    tokens = case.get("tokens", None)
    if tokens is None:
        input_text = case.get("input_text", "")
        tokens = tokenizer.encode(input_text, truncation=False)
        print("通过 input_text 重新生成 tokens。")
    result["tokens"] = tokens

    # 在 prompt 开头添加英文提示，要求直接输出答案选项
    english_prompt = "Answer with only a single letter (A, B, C, or D) representing the correct option. Do not explain your reasoning. Just output the letter of the answer. "
    modified_input_text = english_prompt + question
    input_ids = tokenizer(modified_input_text, return_tensors="pt").input_ids.to(model.device)

    # -----------------------------
    # 关键修改：设置自定义 routing 权重到对应 decoder 层的 MoE 模块
    # -----------------------------
    # updated_layers 是一个列表，每个元素包含 "layer" 和 "routing_weights" 字段
    for layer_info in updated_layers:
        layer_idx = layer_info["layer"]
        # 假设 MoE 模块位于每个 decoder layer 的 mlp 部分
        model.model.layers[layer_idx].mlp.custom_routing_weights = layer_info["routing_weights"]

    try:
        # 调用 generate 时不再传 custom_routing_info 参数
        new_generated_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
        )
        new_generated_text = tokenizer.decode(new_generated_output[0], skip_special_tokens=True)
        print("\n使用更新后的 routing weights 重新生成的完整文本：")
        print(new_generated_text)
    except Exception as e:
        print("\n生成时出错，执行普通生成。错误信息：", e)
        new_generated_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
        )
        new_generated_text = tokenizer.decode(new_generated_output[0], skip_special_tokens=True)
        print("\n普通生成的完整文本：")
        print(new_generated_text)
    
    routing_used = detect_custom_routing_usage(model)
    result["custom_routing_used"] = routing_used

    result["new_generated_text_full"] = new_generated_text
    optimized_inference = extract_answer_option(new_generated_text)
    result["optimized_inference"] = optimized_inference

    result["original_inference"] = case.get("model_answer", "").strip().upper()
    correct_ans = case.get("correct_answer", "").strip().upper()
    result["correct_inference"] = correct_ans

    result["optimized_is_correct"] = (optimized_inference == correct_ans)

    # 可选：生成后重置各层的 custom_routing_weights（如果不希望影响后续生成）
    for layer in model.model.layers:
        layer.mlp.custom_routing_weights = None

    return result

# =========================
# 主程序
# =========================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "allenai/OLMoE-1B-7B-0125-Instruct"
    model = OlmoeForCausalLM.from_pretrained(
        model_name,
        device_map={'': 0},
        torch_dtype=torch.float16
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 生成示例的 routing 信息（max_length 设为64）
    sample_text = "This is a test to analyze routing information in the OLMoE model."
    results = extract_routing_info(sample_text, model, tokenizer, max_length=64)
    save_results_to_json(results, filename="routing_results.json")
    print_analysis_results(results)

    with open("arc_challege_routing_results.json", "r", encoding="utf-8") as f:
        evaluation_cases = json.load(f)

    reference_files = [
        "instruct_openbookqa_correct_routing_results.json",
        "instruct_sciq_correct_routing_results.json"
    ]
    
    reference_cases = load_multiple_reference_files(reference_files)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    correct_evaluation = []
    incorrect_evaluation = []
    total = 0
    max_cases = 10000  # 根据需要调整处理案例数

    for idx, (case_id, one_case) in enumerate(evaluation_cases.items()):
        if idx >= max_cases:
            break
        print(f"\n=========== 处理 case {case_id} ===========")
        eval_result = re_infer_case(one_case, reference_cases, embedder, model, tokenizer, max_length=200)
        if eval_result is None:
            continue
        total += 1
        if one_case.get("is_correct", False):
            correct_evaluation.append({"case_id": case_id, **eval_result})
        else:
            incorrect_evaluation.append({"case_id": case_id, **eval_result})

    print(f"\n总共评估 {total} 个案例。")
    print(f"正确案例：{len(correct_evaluation)} 个；错误案例：{len(incorrect_evaluation)} 个。")

    originally_correct = 0
    originally_incorrect = 0
    improved = 0       
    worsened = 0       
    still_correct = 0  
    still_incorrect = 0  

    all_evaluation = correct_evaluation + incorrect_evaluation
    for case in all_evaluation:
        original_answer = case["original_inference"]
        optimized_answer = case["optimized_inference"]
        correct_answer = case["correct_inference"]
        
        original_is_correct = (original_answer == correct_answer)
        if original_is_correct:
            originally_correct += 1
        else:
            originally_incorrect += 1
            
        optimized_is_correct = (optimized_answer == correct_answer)
        
        if original_is_correct and optimized_is_correct:
            still_correct += 1
        elif original_is_correct and not optimized_is_correct:
            worsened += 1
        elif not original_is_correct and optimized_is_correct:
            improved += 1
        else:
            still_incorrect += 1

    print("\n========== 优化效果统计 ==========")
    print(f"总案例数: {total}")
    print(f"原始正确案例数: {originally_correct} ({originally_correct/total*100:.2f}%)")
    print(f"原始错误案例数: {originally_incorrect} ({originally_incorrect/total*100:.2f}%)")
    print(f"优化后正确案例数: {still_correct + improved} ({(still_correct + improved)/total*100:.2f}%)")
    print(f"优化后错误案例数: {still_incorrect + worsened} ({(still_incorrect + worsened)/total*100:.2f}%)")
    print("\n优化效果详情:")
    print(f"改进案例数 (原错→优化对): {improved} ({improved/total*100:.2f}%)")
    print(f"恶化案例数 (原对→优化错): {worsened} ({worsened/total*100:.2f}%)")
    print(f"保持正确案例数: {still_correct} ({still_correct/total*100:.2f}%)")
    print(f"保持错误案例数: {still_incorrect} ({still_incorrect/total*100:.2f}%)")

    net_improvement = improved - worsened
    print(f"\n优化净效果: {'+' if net_improvement >= 0 else ''}{net_improvement} 案例 ({net_improvement/total*100:.2f}%)")

    optimization_stats = {
        "total_cases": total,
        "originally_correct": originally_correct,
        "originally_incorrect": originally_incorrect,
        "optimized_correct": still_correct + improved,
        "optimized_incorrect": still_incorrect + worsened,
        "improved_cases": improved,
        "worsened_cases": worsened,
        "still_correct_cases": still_correct,
        "still_incorrect_cases": still_incorrect,
        "net_improvement": net_improvement,
        "original_accuracy": originally_correct/total,
        "optimized_accuracy": (still_correct + improved)/total
    }

    save_results_to_json(optimization_stats, filename="first5_optimization_statistics.json")
    save_results_to_json(correct_evaluation, filename="first5_combined_first5_correct_evaluation_results.json")
    save_results_to_json(incorrect_evaluation, filename="first5_combined_first5_incorrect_evaluation_results.json")
