import os
import json
import argparse
from glob import glob
from natsort import natsorted
import datasets
from PIL import Image
import ast

def string_to_list(string):
    return ast.literal_eval(string)

def get_image_index(image_path):
    filename = os.path.basename(image_path)
    return int(filename.split('_')[1].split('.')[0])

def convert_to_conversation(example, idx, image_map):
    if idx in image_map:
        conversation = [
            {
                "from": "user",
                "value": f"<img>{image_map[idx]}</img>\n{example['korean_question']}"
            },
            {
                "from": "user",
                "value": f"Choices: {example['korean_choices']}"
            },
            {
                "from": "assistant",
                "value": f"Answer: {example['answer_str']}"
            },
            {
                "from": "assistant",
                "value": f"Hint: {example['korean_hint']}"
            }
        ]
    else:
        conversation = [
            {
                "from": "user",
                "value": f"{example['korean_question']}"
            },
            {
                "from": "user",
                "value": f"Choices: {example['korean_choices']}"
            },
            {
                "from": "assistant",
                "value": f"Answer: {example['answer_str']}"
            },
            {
                "from": "assistant",
                "value": f"Hint: {example['korean_hint']}"
            }
        ]
    
    return {"conversations": conversation}

def process_dataset(output_path, num_samples):
    # Load datasets
    data1 = datasets.load_dataset("derek-thomas/ScienceQA")
    data2 = datasets.load_dataset("iamjoon/KorScienceQA")

    # Add Korean columns
    korean_columns = ['korean_question', 'korean_choices', 'korean_hint']
    korean_data = data2['train'].select_columns(korean_columns)

    new_data1 = data1['train']
    for column in korean_columns:
        new_data1 = new_data1.add_column(column, korean_data[column])

    # Convert Korean choices to list
    new_korean_choices = [string_to_list(choice) for choice in new_data1['korean_choices']]
    new_data1 = new_data1.remove_columns(['korean_choices'])
    new_data1 = new_data1.add_column('korean_choices', new_korean_choices)

    # Convert answer to choice string
    def convert_answer_to_choice(example):
        example['answer_str'] = example['korean_choices'][example['answer']]
        return example

    new_data1 = new_data1.map(convert_answer_to_choice)

    # Process images
    image_dir = "./image"
    os.makedirs(image_dir, exist_ok=True)
    image_files = natsorted(glob(os.path.join(image_dir, "*.jpeg")))
    image_map = {get_image_index(image_path): image_path for image_path in image_files}

    # Convert to conversation format
    converted_dataset = new_data1.select(range(num_samples)).map(
        lambda example, idx: convert_to_conversation(example, idx, image_map),
        with_indices=True
    )

    # Format for Qwen-vl-chat
    formatted_conversations = []
    for idx, conversation in enumerate(converted_dataset["conversations"]):
        conversation_entry = {
            "id": f"identity_{idx}",
            "conversations": conversation
        }
        formatted_conversations.append(conversation_entry)

    # Save to JSON file
    with open(os.path.join(output_path, 'formatted_conversations.json'), 'w', encoding='utf-8') as f:
        json.dump(formatted_conversations, f, indent=4, ensure_ascii=False)

    print(f"Data has been saved to {os.path.join(output_path, 'formatted_conversations.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and convert dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output directory")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to process")

    args = parser.parse_args()

    process_dataset(args.output_path, args.num_samples)