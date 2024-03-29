import os.path as osp
import json
import pickle
import os
import datetime
import argparse

test_qa_pkl=open(r'/data/Dataset/VQA/pvqa/qas/test/test_qa.pkl','rb')
test_qa_dic=pickle.load(test_qa_pkl)

import base64
import requests

# OpenAI API Key
api_key = API_KEY

parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int,
                    default=0, help='index for data')

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "/data/Dataset/VQA/pvqa/images/test"
answer_path = "/data/Dataset/VQA/pvqa/gpt_vqa/test"

args = parser.parse_args()
idx = args.idx
count = 0

for item_idx in range(len(test_qa_dic))[idx:]:

    if count == 3:
        print(count)
        break

    item = test_qa_dic[item_idx]
    item_str = "ITEM_IDX: "+str(item_idx)
    image_name = item['image']+'.jpg'
    image_ab_path = osp.join(image_path, image_name)
    image_text = item['question']
    image_answer = item['answer']

    image_text_1 = 'Input a medical image along with a question related to the image. Questions can fall into one of the two categories: closed-ended or open-ended. Closed-ended questions require the answer only among [“Yes”, “No”, “A”, “B”, “C”, “D”]. Open-ended questions require the answer of a word or a sentence, not in [“Yes”, “No”, “A”, “B”, “C”, “D”]. Questions can also fall into one of the seven categories: Anatomical Structures, Lesion & Abnormality Detection, Disease Diagnosis, Spatial Relationships, Contrast Agents and Staining, Microscopic Features, Pathophysiological Mechanisms & Manifestations. Based on the medical image and the corresponding question, determine whether the question is closed-ended or open-ended, determine which category the image belongs to in the given seven categories, directly provide an appropriate response without any additional information or explanations, and provide a number (?/10) to show how confident you are of the answer. question: '+image_text

    # Getting the base64 string
    base64_image = encode_image(image_ab_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": image_text_1
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 2000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    gpt_response = response.json()
    # item['gpt4_response'] = gpt_response['choices'][0]['message']['content']
    item['gpt4_full_response'] = gpt_response
    # print(item)
    if 'error' in list(gpt_response.keys()):
        print(count)
        break
    now_time = datetime.datetime.now()
    output_json_path = os.path.join(answer_path, item['image'] + '_'  + str(now_time)[-6:]  + '.json')
    with open(output_json_path, 'w') as json_file:
        json.dump(item, json_file, indent=4)
    count = count + 1
