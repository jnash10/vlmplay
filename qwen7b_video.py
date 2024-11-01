from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# # default: Load the model on the available device(s)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="bfloat16", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
min_pixels = 256 * 28 * 28
max_pixels = 720 * 28 * 28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
)


# Messages containing a video and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "good/frontfoot1.mp4",
                "max_pixels": 640 * 480,
                "fps": 1.5,
            },
            # {
            #     "type": "text",
            #     "text": """"Based on the video, here is the assessment of the player's performance:\n\n
            #     1. **Stance**: The player's stance appears to be balanced and stable. He is standing with his feet shoulder-width apart, which is a good starting position for hitting a cricket shot.\n\n
            #     2. **Shot Quality**: The player seems to be executing a well-timed shot. The follow-through of the bat suggests that he has made good contact with the ball, indicating a high-quality shot.\n\n
            #     3. **Weight Transfer Through the Shot**: The player's weight transfer is evident as he moves from his back foot to his front foot during the shot.
            #     This is crucial for generating power and maintaining balance. The player's body weight appears to be well转移到 his front foot, which is a positive sign.\n\n
            #     Overall, the player's performance in the video shows good technique and control, indicating a well-executed cricket shot.""",
            # },
            {
                "type": "video",
                "video": "bad/drive2.mp4",
                "max_pixels": 640 * 480,
                "fps": 1.5,
            },
            {
                "type": "text",
                "text": "Rate the video compared to the previous one: 1)stance 2)shot quality 3)weight transfer through the shot",
            },
        ],
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
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
