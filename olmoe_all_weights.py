import torch
import numpy as np
import json
from transformers import OlmoeForCausalLM, AutoTokenizer
from collections import Counter
from tqdm import tqdm

def extract_routing_info(text, model_name="allenai/OLMoE-1B-7B-0924", batch_size=512, max_length=100):
    """
    Extract routing information using single GPU.

    Args:
        text (str): Input text to analyze.
        model_name (str): Name of the OLMoE model to use.
        batch_size (int): Batch size for processing tokens.
        max_length (int): Maximum length of the generated output.

    Returns:
        dict: Contains detailed routing weights, expert counts, tokens, tokenizer name, and model output.
    """
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device("cuda")

    # Load model and tokenizer
    model = OlmoeForCausalLM.from_pretrained(
        model_name,
        device_map={'': 0},  # Map to single GPU
        torch_dtype=torch.float16  # Use fp16 to reduce memory usage
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer(text, truncation=False)["input_ids"]

    expert_counter = Counter()
    routing_info = []

    # Process tokens in batches with progress bar
    with torch.no_grad():
        for i in tqdm(range(0, len(tokens), batch_size), desc="Processing tokens"):
            batch_tokens = tokens[i:min(i + batch_size, len(tokens))]
            input_ids = torch.tensor(batch_tokens).reshape(1, -1).to(device)

            outputs = model(input_ids=input_ids, output_router_logits=True)

            # Get routing information from all layers
            all_layers_logits = outputs["router_logits"]

            # Convert to float32 before softmax, then back to CPU
            all_layers_probs = [torch.nn.functional.softmax(layer_logits.float(), dim=-1).cpu().numpy()
                                for layer_logits in all_layers_logits]

            # Store detailed routing information
            for token_idx, token_id in enumerate(batch_tokens):
                token_info = {
                    "token_id": int(token_id),
                    "token_text": tokenizer.decode([token_id]),
                    "layers": []
                }
                for layer_idx, layer_probs in enumerate(all_layers_probs):
                    # Sort experts by weight in descending order
                    sorted_experts = sorted(
                        enumerate(layer_probs[token_idx]),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    layer_info = {
                        "layer": layer_idx,
                        "routing_weights": {
                            f"expert_{expert_id}": float(weight)
                            for expert_id, weight in sorted_experts
                        }
                    }
                    token_info["layers"].append(layer_info)
                routing_info.append(token_info)

            # Count top experts for each layer
            for layer_probs in all_layers_probs:
                top_experts = np.argsort(-layer_probs, axis=-1)
                expert_counter.update(top_experts.flatten().tolist())  # Convert to list of native ints

    # Generate model output
    input_ids = torch.tensor(tokens).reshape(1, -1).to(device)
    generated_output = model.generate(
        input_ids=input_ids,
        max_length=max_length
    )

    generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

    return {
        'routing_info': routing_info,  # Detailed routing information
        'expert_counts': {int(k): int(v) for k, v in expert_counter.items()},  # Convert keys and values to int
        'tokens': tokens,
        'tokenizer': tokenizer.name_or_path,  # Save tokenizer name instead of the object
        'generated_text': generated_text  # Model's generated output
    }

def save_results_to_json(results, filename="original_routing_results.json"):
    """
    Save results to a JSON file.

    Args:
        results (dict): Results from extract_routing_info.
        filename (str): Name of the output JSON file.
    """
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

def print_analysis_results(results):
    """
    Print analysis results in a readable format.

    Args:
        results (dict): Results from extract_routing_info.
    """
    # Print expert utilization rates
    total_selections = sum(results['expert_counts'].values())
    print("\nExpert utilization rates:")
    for expert_id, count in sorted(results['expert_counts'].items()):
        percentage = count / total_selections * 100
        print(f"Expert {expert_id}: {percentage:.2f}%")

    # Print routing weights for first 5 tokens for all layers
    print("\nRouting weights for first 5 tokens for all layers:")
    for token_info in results['routing_info'][:5]:  # First 5 tokens
        print(f"\nToken ID: {token_info['token_id']}, Text: '{token_info['token_text']}'")
        for layer_info in token_info['layers']:
            print(f"  Layer {layer_info['layer']}:")
            for expert_id, weight in layer_info['routing_weights'].items():
                print(f"    {expert_id}: {weight:.4f}")

    # Print generated text
    print("\nGenerated Text:")
    print(results['generated_text'])

if __name__ == "__main__":
    # Example usage
    sample_text = "Question: When food is reduced in the stomach\nChoices: A. the mind needs time to digest B. take a second to digest what I said C. nutrients are being deconstructed D. reader's digest is a body of works\nAnswer:"
    results = extract_routing_info(sample_text)

    # Save results to JSON
    save_results_to_json(results)

    # Print analysis results
    print_analysis_results(results)
