from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
from transformers import DataCollatorForLanguageModeling, logging

logger = logging.get_logger(__name__)


@dataclass
class TokenizationDataCollator:
    """
    Data collator for encoder-decoder training with token masking.

    This collator:
    - Pads sequences to the same length
    - Creates labels for causal language modeling
    - Masks tokens with specified probability for encoder input
    - Ensures proper shifting for decoder (labels shifted by 1)

    Args:
        tokenizer: The tokenizer to use for padding
        pad_to_multiple_of (int, optional): Pad to multiple of this value
    """

    tokenizer: Any
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate features into a batch with masking.

        Args:
            features: List of feature dictionaries with 'input_ids' and optionally 'attention_mask'

        Returns:
            Dictionary with:
                - input_ids: Original input IDs for decoder
                - input_ids_enc: Masked input IDs for encoder
                - attention_mask: Attention mask
                - labels: Labels for causal LM (input_ids shifted)
        """

        # Extract input_ids from features
        def tokenize_function_chat(examples):
            return self.tokenizer(
                [self.tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                ) for example in examples],
                max_length=self.pad_to_multiple_of,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )

        
        def tokenize_function_text(examples):
            return self.tokenizer(
                [example["text"] for example in examples],
                max_length=self.pad_to_multiple_of,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )

        if "input_ids" not in features[0]:
            features = tokenize_function_chat(features) if "messages" in features[0] else tokenize_function_text(features)
            input_ids = features["input_ids"]
        else:
            # Extract input_ids from features
            input_ids = [
                (
                    torch.tensor(f["input_ids"])
                    if not isinstance(f["input_ids"], torch.Tensor)
                    else f["input_ids"]
                )
                for f in features
            ]

        # Pad sequences
        batch_size = len(input_ids)
        max_length = max(len(ids) for ids in input_ids)

        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        # Create padded tensors
        padded_input_ids = torch.full(
            (batch_size, max_length), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)

        for i, ids in enumerate(input_ids):
            length = len(ids)
            padded_input_ids[i, :length] = ids
            attention_mask[i, :length] = 1

        # Create labels (same as input_ids, with -100 for padding)
        labels = padded_input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
