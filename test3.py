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

# Load CSV file
df = pd.read_csv('usecase2brd-dataset/testset.csv')

# Convert each row into a JSON-like dict
usecases = df.to_dict(orient='records')

# Initialize the usecase_list with the first usecase
usecase_list = [{
    "name": usecases[0]["name"],
    "description": usecases[0]["scenario"],
    "precondition": usecases[0]["preconditions"]
}]

count = 0
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
    prev_result = usecases[idx - 1]["result"]
    print(f"res = {res}, prev_result = {prev_result}")
    if res == prev_result:
        count += 1
        print("Offending usecase:")
        print(usecase)
        print("-" * 50)
    if res:
        usecase_list.append(usecase)
    else:
        # Print the new project's starting usecase
        # print("New project starts with usecase:")
        # print(usecase)
        # print("-" * 50)
        # Start a new group
        usecase_list = [usecase]
        print("Current Mistatch Count:", count)

print("Final Mismatch Count:", count)
