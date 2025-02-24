
import argparse
import json
import os

# I want to create the following json
# {idx: 0, "train": [{"input": [[color, material, shape]], "output": "on"}], "test": {"input": [color, material, shape], "output", "on", "type": "direct"}}

# In parsing, for each 6 example, we have 4 trials per input data
# For every 6 example per input, we use 1 trial. Hence we have 4 output data per input data

# Parse arguments
parser = argparse.ArgumentParser(description='Convert ACRE data split to symbolic or text jsonl inputs compatible with models.')
parser.add_argument('input_path')
parser.add_argument('output_path')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--symbolic', action='store_true')
group.add_argument('--text', action='store_true')

args = parser.parse_args()
input_file_path = args.input_path
output_file_path = args.output_path
symbolic = args.symbolic

print(f"Writing to {output_file_path} from {input_file_path} in {'symbolic' if symbolic else 'text'} format.")


# Read raw data
with open(input_file_path) as f:
    input_data = json.load(f)

print(f"Read {len(input_data)} samples.")
            

# Text translation functions
LIGHT_DICT_TEXT = ["off", "undetermined", "on"]
with open("object_dict.json", "r") as object_dict_file:
    OBJECT_DICT_TEXT = json.load(object_dict_file)
    
def get_instructions_text():
    return [
        {
            "role": "system",
            "content": "Objects of various color, shape, and texture are displayed. Some objects may contain a device to turn a light on if displayed. From the observations, deduce if the light is on, off, or if the state cannot be determined. Your answer must contain a single word:"
        },
        {
            "role": "system",
            "content": "on."
        },
        {
            "role": "system",
            "content": "off."
        },
        {
            "role": "system",
            "content": "undetermined."
        }
    ]

def get_example_text(objects, light_state):
    # object_text = ""
    input_examples = []
    for object in objects:
        obj_desc = OBJECT_DICT_TEXT[str(object)]
        color = obj_desc["color"]
        shape = obj_desc["shape"]
        material = obj_desc["material"]
        # object_text += f"A {c} {s} in {t} is visible. "
        # [color, material, shape]
        input_examples.append([color, material, shape])
    return {
            "input": input_examples,
            "output": light_state
        }

# Label has 0, 1, 2 integer which are indexed into the array ["off", "undetermined", "on"]
def get_trial_text(objects, label):
    # object_text = ""
    input_examples = []
    for object in objects:
        obj_desc = OBJECT_DICT_TEXT[str(object)]
        color = obj_desc["color"]
        shape = obj_desc["shape"]
        material = obj_desc["material"]
        # object_text += f"A {c} {s} in {t} is visible. "
        # [color, material, shape]
        input_examples.append([color, material, shape])
    
    # return [
    #             {
    #                 "role": "user",
    #                 "content": object_text
    #             }
    #         ], LIGHT_DICT_TEXT[label]
    return {
        "input": input_examples,
        "output": LIGHT_DICT_TEXT[label]
    }


# Select translation functions
get_example = get_example_text
get_trial = get_trial_text
get_answer_list = lambda: LIGHT_DICT_TEXT



# Generate output
# output_data = []
# for sample in input_data:
#     new_samples = []
#     input_sample = get_instructions()
#     for trial in sample:
#         if trial["light_state"] != "no": # example
#             input_sample += get_example(trial["objects"], trial["light_state"])
#         else: # test case
#             input_test = input_sample.copy()
#             trial_input, trial_ideal = get_trial(trial["objects"], trial["label"])
            
#             new_samples.append({
#                 "input": input_test + trial_input,
#                 "choice_strings": get_answer_list(),
#                 "ideal": trial_ideal
#             })
#     output_data += new_samples

output_data = []
count_idx = 0
for sample in input_data: # For each data in input (6 examples, 4 tests)
    new_samples = []
    # {idx: 0, "examples": [{"input": [[color, material, shape]], "output": "on"}], "question": {"input": [color, material, shape], "output", "on", "type": "direct"}}
    # examples, ideally is an array of dict(input, output)
    examples = []
    for entry in sample:
        if entry["light_state"] != "no": # example
            example = get_example(entry["objects"], entry["light_state"])
            examples.append(example)
        else: # test case
            # question is a dict of input and output
            question = get_trial(entry["objects"], entry["label"])
            
            output_data.append({
                "idx": count_idx,
                "examples": examples,
                "question": question
            })
            count_idx += 1

print(f"Generated {len(output_data)} samples.")

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Print to output file
with open(output_file_path, "w") as f:
    for sample in output_data:
        f.write(json.dumps(sample) + "\n")

print("Done.")