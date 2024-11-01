import os
import json
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from datetime import datetime
from typing import List, Dict, Any


def load_or_create_json_log(log_file: str) -> List[Dict[str, Any]]:
    """Create a new JSON log file if it doesn't exist, or load existing one"""
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            return json.load(f)
    return []


def append_to_json_log(log_file: str, entry: Dict[str, Any]) -> None:
    """Append a new entry to the JSON log file"""
    logs = load_or_create_json_log(log_file)
    logs.append(entry)
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)


def create_message_content(
    video_paths: List[str], prompts: List[str], max_pixels: int, fps: float
) -> List[Dict]:
    """
    Create message content for multiple videos and prompts
    Returns a list of alternating video and text content items
    """
    content = []

    # Add context videos and prompts first
    for vid_path, prompt in zip(video_paths[:-1], prompts[:-1]):
        content.extend(
            [
                {
                    "type": "video",
                    "video": vid_path,
                    "max_pixels": max_pixels,
                    "fps": fps,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ]
        )

    # Add the target video and prompt
    content.extend(
        [
            {
                "type": "video",
                "video": video_paths[-1],
                "max_pixels": max_pixels,
                "fps": fps,
            },
            {
                "type": "text",
                "text": prompts[-1],
            },
        ]
    )

    return content


# Model initialization with flash attention
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# Processor initialization with custom pixel ranges
min_pixels = 256 * 28 * 28
max_pixels = 720 * 28 * 28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
)

# Configuration
video_directory = "bad"
dtype = "bfloat16"
fps = 1.0
maxp = "720 * 1280"

# JSON log file
log_file = f"experiment_log.json"

# Example prompts for different purposes
context_prompt = "Given this video of a cricket player, rate the shot quality (Good/Bad). Shot Quality: {}"
target_prompt = (
    "Given this video of a cricket player, commend on his stance, was it good or bad"
)

# Get all video files
video_files = [f for f in os.listdir(video_directory) if f.endswith(".mp4")]

# Process videos in groups (for example, 2 context videos + 1 target video)
context_size = 0  # Number of context examples
for i in range(0, len(video_files), context_size + 1):
    batch_videos = video_files[i : i + context_size + 1]
    if len(batch_videos) < context_size + 1:
        break  # Skip if we don't have enough videos for the full context + target

    print(f"Processing batch starting with: {batch_videos[0]}")

    # Prepare video paths and prompts
    video_paths = [os.path.join(video_directory, v) for v in batch_videos]

    # Example of how to create prompts with different annotations for context
    prompts = [
        context_prompt.format("Good"),  # First context example
        context_prompt.format("Bad"),  # Second context example
        target_prompt,  # Target prompt
    ]

    # Create message content
    messages = [
        {
            "role": "user",
            "content": create_message_content(video_paths, prompts, eval(maxp), fps),
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(f"Output for batch: {output_text}")

    # Create log entry with context information
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "directory_name": video_directory,
        "maxp": maxp,
        "fps": fps,
        "dtype": dtype,
        "context_examples": [
            {
                "video_path": video_paths[i],
                "prompt": prompts[i],
            }
            for i in range(context_size)
        ],
        "target": {
            "video_path": video_paths[-1],
            "prompt": prompts[-1],
        },
        "llm_response": output_text,
    }

    # Append to JSON log
    append_to_json_log(log_file, log_entry)
