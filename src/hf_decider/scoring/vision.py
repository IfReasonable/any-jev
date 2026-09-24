"""Exact text-continuation scoring conditioned on image/text chat inputs."""

from collections.abc import Mapping, Sequence

import torch

from hf_decider.errors import CandidateBoundaryError, UnsupportedModelError
from hf_decider.scoring.outputs import CandidateScores


class VisionCandidateScorer:
    """Score assistant text continuations using a multimodal chat processor.

    Each candidate gets an independent processor call and model forward because
    vision tensor layouts and image metadata vary across model families.
    """

    def __init__(self, model: torch.nn.Module, processor: object) -> None:
        if not hasattr(processor, "apply_chat_template") or not hasattr(processor, "tokenizer"):
            raise UnsupportedModelError("VLM scoring requires a chat processor with a tokenizer")
        self.model = model.eval()
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def score_image_text(
        self, context: str, images: Sequence[object], candidates: Sequence[str]
    ) -> CandidateScores:
        """Score candidate assistant replies to a user message with images.

        ``images`` may contain PIL images, local paths, or URLs understood by
        the model's processor. At least one image is required.
        """
        if not isinstance(context, str):
            raise TypeError("context must be a string")
        if not images:
            raise ValueError("images must not be empty")
        content = []
        for image in images:
            if isinstance(image, str):
                content.append({"type": "image", "path": image})
            else:
                content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": context})
        return self.score_messages([{"role": "user", "content": content}], candidates)

    def score_messages(
        self, messages: Sequence[Mapping[str, object]], candidates: Sequence[str]
    ) -> CandidateScores:
        """Score assistant continuations after a multimodal chat history.

        Candidate tokenization is derived from a complete chat template for
        each candidate. A changed prompt prefix raises CandidateBoundaryError.
        """
        if not messages or messages[-1].get("role") == "assistant":
            raise ValueError("messages must end with a non-assistant message")
        if not candidates:
            raise ValueError("candidates must not be empty")
        if any(not isinstance(candidate, str) or not candidate for candidate in candidates):
            raise ValueError("each candidate must be a nonempty string")

        # ProcessorMixin may normalize content blocks in place; copy the structure.
        history = [
            {
                **message,
                "content": [dict(block) for block in message["content"]]
                if isinstance(message.get("content"), list)
                else message.get("content"),
            }
            for message in messages
        ]
        prefix_inputs = self.processor.apply_chat_template(
            history,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prefix_ids = prefix_inputs["input_ids"]
        if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1 or prefix_ids.shape[1] == 0:
            raise UnsupportedModelError("Processor must return one nonempty input_ids row")
        prefix = prefix_ids[0].tolist()
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        token_scores = []
        with torch.inference_mode():
            for candidate in candidates:
                full_messages = history + [
                    {"role": "assistant", "content": [{"type": "text", "text": candidate}]}
                ]
                model_inputs = self.processor.apply_chat_template(
                    full_messages,
                    continue_final_message=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                full_ids = model_inputs["input_ids"]
                if full_ids.ndim != 2 or full_ids.shape[0] != 1:
                    raise UnsupportedModelError("Processor must return one input_ids row")
                ids = full_ids[0].tolist()
                if ids[: len(prefix)] != prefix:
                    raise CandidateBoundaryError(
                        f"Token boundary changed for candidate {candidate!r}; try a leading "
                        "space or newline, or provide a different assistant continuation"
                    )
                continuation = ids[len(prefix) :]
                if not continuation:
                    raise ValueError(f"candidate {candidate!r} produced no continuation tokens")
                inputs = {
                    key: (
                        value.to(device=device, dtype=model_dtype)
                        if value.is_floating_point()
                        else value.to(device)
                    )
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in model_inputs.items()
                }
                logits = getattr(self.model(**inputs), "logits", None)
                if (
                    logits is None
                    or logits.ndim != 3
                    or logits.shape[1] != inputs["input_ids"].shape[1]
                ):
                    raise UnsupportedModelError(
                        "VLM forward must return logits aligned with input_ids positions"
                    )
                selected = logits[0, len(prefix) - 1 : len(prefix) + len(continuation) - 1]
                logprobs = torch.log_softmax(selected.float(), dim=-1)
                token_scores.append(
                    logprobs.gather(
                        -1, torch.tensor(continuation, dtype=torch.long, device=device)[:, None]
                    ).squeeze(-1)
                )
        return CandidateScores(
            tuple(candidates), token_scores, torch.stack([score.sum() for score in token_scores])
        )
