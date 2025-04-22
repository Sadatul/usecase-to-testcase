from together import Together
import pandas as pd

import json
client = Together()

# Dummy project matching function (adjust as needed)
def isSameProject(usecase_list, usecase):
    response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    messages=[
        {"role": "system", "content": """
         We have a large dataset of usecases. We need to cluster the usecase
         that fall under the same project. Now you will be given a set of usecase and
         you will reply whether the usecase fall under the same project or not. The format
         of usecases is as follows:{
            "name": "usecase name",
            "description": "usecase description",
            "precondition": "usecase precondition",
         }

         output format should be as follows. Remember: Only output the JSON, with no explanation or extra text.:
         {
            "verdict": "True" or "False",
            "description": "description of the verdict"
         }
         """
        },
        {
            "role": "user",
            "content": f"usecase_list: {usecase_list}, usecase: {usecase}. Do they fall under the same project?"
        }
        ]
    )
    return eval(json.loads(response.choices[0].message.content).get("verdict"))

def generateUserStoryFromUsecaseList(usecase_list):
    response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    messages=[
        {
         "role": "system", "content": """
         You will be given a set of usecase and you will generate a user story for the usecase list. The format
         of usecases is as follows:{
            "name": "usecase name",
            "scenario": "usecase description",
            "actors": "usecase actors",
            "preconditions": "usecase preconditions",
            "steps": "usecase steps",
         }

         output format should be as follows. Remember: Only output the JSON, with no explanation or extra text.:
         {
            "story": "description of the user story",
         }
         """
        },
        {
            "role": "user",
            "content": f"usecase_list: {usecase_list}. Generate a user story based on the usecase list."
        }
        ]
    )
    return json.loads(response.choices[0].message.content).get("story")

# Load CSV file
df = pd.read_csv('usecase2brd-dataset/testset.csv')

# Convert each row into a JSON-like dict
df = df.drop(columns=["result"])
usecases = df.to_dict(orient='records')

# Initialize the usecase_list with the first usecase
usecase_list = [{
    "name": usecases[0]["name"],
    "description": usecases[0]["scenario"],
    "precondition": usecases[0]["preconditions"]
}]

row_list = [
    usecases[0]
]

results = []
# Process the remaining usecases
for idx, row in enumerate(usecases[1:], start=1):
    usecase = {
        "name": row["name"],
        "description": row["scenario"],
        "precondition": row["preconditions"]
    }
    res = isSameProject(usecase_list, usecase)
    # print(usecase_list)
    # print(usecase)
    print(f"res = {res}")
    if res:
        usecase_list.append(usecase)
        row_list.append(row)
    else:
        description = generateUserStoryFromUsecaseList(row_list)
        results.append({
            "usecases": row_list,
            "user_story": description
        })
        usecase_list = [usecase]
        row_list = [row]
        print(f"Description: {description}")
        print("-" * 50)

print("Total number of projects: ", len(results))
with open("test_usecase2brd_dataset.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)