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
    order = ["epoch", "step", "loss", "learning_rate"]
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
        prompt = f"""{'CONTEXT: ' + str("; ".join(golden.context)) if golden.context else ''}
                QUESTION: {golden.input}
                ANSWER:"""

        tokenized = tokenizer(
            prompt,
            return_tensors="pt",
            **tokenizer_args,
        )

        tokenized_output = {k: v.to(model.device) for k, v in tokenized.items()}
        
        input_ids = tokenized_output["input_ids"]
        outputs = model.generate(input_ids, **generator_args)
        decoded_output = tokenizer.decode(outputs[0][tokenized_output["input_ids"].shape[-1]:], skip_special_tokens=True)
        golden.actual_output = decoded_output

    test_cases = convert_goldens_to_test_cases(
        goldens=evaluation_dataset.goldens,
        _alias=evaluation_dataset.alias,
    )
    return test_cases
