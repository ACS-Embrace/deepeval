from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.dataset.utils import convert_goldens_to_test_cases
from typing import List, Dict, Optional


def get_column_order(scores: Dict) -> List[str]:
    """
    Determine the order of columns for displaying scores.

    Args:
        scores (Dict): Dictionary containing scores.

    Returns:
        List[str]: List of column names in the desired order.
    """
    preferred = ["epoch", "step", "loss", "learning_rate"]
    order = [key for key in preferred if key in scores]
    order.extend([key for key in scores.keys() if key not in order])
    return order


def generate_test_cases(
    model,
    tokenizer,
    tokenizer_args: Optional[Dict],
    evaluation_dataset: EvaluationDataset,
    generator_args: Optional[Dict],
) -> List[LLMTestCase]:
    """
    Generate test cases based on a language model.

    Args:
        model: The language model to generate outputs.
        tokenizer: The tokenizer for processing prompts.
        tokenizer_args (Dict): Arguments for the tokenizer.
        evaluation_dataset (EvaluationDataset): The dataset containing Golden.

    Returns:
        List[LLMTestCase]: List of generated test cases.
    """

    tokenizer_args = tokenizer_args or {}
    generator_args = generator_args or {}

    goldens = evaluation_dataset.goldens
    for golden in goldens:
        context_prefix = (
            "Context: " + "; ".join(golden.context) + "\n\n"
            if golden.context
            else ""
        )
        user_message = context_prefix + golden.input

        # Use the tokenizer's chat template if available so the model sees
        # the exact format it was trained on (e.g. ChatML <|im_start|>/<|im_end|>).
        # Fall back to a plain prompt for base models without a chat template.
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            # Check whether this template uses <think> blocks (e.g. Qwen3, DeepSeek-R1).
            # If so, prefill the assistant turn with an empty think block so the model
            # skips chain-of-thought and generates only the answer.
            uses_think_tags = (
                tokenizer.chat_template is not None
                and "<think>" in tokenizer.chat_template
            )
            if uses_think_tags:
                messages = [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": "Answer:<think>\n\n</think>"},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
            else:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_message}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prompt = f"{user_message}\nAnswer:"

        tokenized = tokenizer(
            prompt,
            return_tensors="pt",
            **tokenizer_args,
        )

        tokenized_output = {k: v.to('cuda') for k, v in tokenized.items()}
        
        outputs = model.generate(**tokenized_output, **generator_args)
        decoded_output = tokenizer.decode(outputs[0][tokenized_output["input_ids"].shape[-1]:], skip_special_tokens=True)

        if not decoded_output or not decoded_output.strip():
            raw_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            print(
                f"\n[DeepEval] Warning: empty output generated.\n"
                f"  Prompt:   {repr(prompt)}\n"
                f"  Response: {repr(raw_response)}\n"
            )

        golden.actual_output = decoded_output
        del tokenized_output, outputs
    test_cases = convert_goldens_to_test_cases(
        goldens=evaluation_dataset.goldens,
        _alias=evaluation_dataset.alias,
    )
    return test_cases
