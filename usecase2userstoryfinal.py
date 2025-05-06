from together import Together
import pandas as pd
import json
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import sys

# Load environment variables
load_dotenv()

LLAMA_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
GPT4_MODEL = "gpt-4o"

openAIClient = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
togetherAIClient = Together(api_key=os.getenv("TOGETHER_API_KEY"))

class StoryRes(BaseModel):
    story: str

sample_usecase_1 = {
    "usecases": [
      {
        "name": "Blood Test Report Extraction (ai-enabled-blood-test-report-analysis project)",
        "scenario": "Automated extraction of required fields from the lab report for a specific type of blood test.",
        "actors": "['RPA Robot', 'Lab Clinician', 'Doctor']",
        "preconditions": "Lab reports are available in a format readable by the RPA robot and the UiPath Document Understanding is able to classify the reports.",
        "steps": "1: RPA robot fetches the lab report for a specific type of blood test. 2: RPA robot uses UiPath Document Understanding to classify the lab report for the type of test. 3: RPA robot extracts the required fields from the report. 4: RPA robot saves the extracted data in a global database without any Personally Identifiable Information (PII)."
      },
      {
        "name": "Blood Test Report Analysis using ML Models (ai-enabled-blood-test-report-analysis project)",
        "scenario": "Analysis of blood test reports using ML models hosted in the UiPath AI Center to derive prescriptive observations.",
        "actors": "['RPA Robot', 'ML Model', 'Lab Clinician', 'Doctor']",
        "preconditions": "Automatically extracted data from the lab report is available in the global database, and ML models for different types of blood tests are available in the UiPath AI Center.",
        "steps": "1: RPA robot sends the report parameters to the ML models hosted in the UiPath AI Center. 2: ML models analyze the report parameters and derive prescriptive observations. 3: Prescriptive observations are attached to the test report and sent to the lab clinician."
      },
      {
        "name": "Highlight Similar Cases (ai-enabled-blood-test-report-analysis project)",
        "scenario": "Highlighting similar cases of a test report found in the database for the specific hospital or laboratory to the lab clinician.",
        "actors": "['RPA Robot', 'Lab Clinician', 'Doctor']",
        "preconditions": "Data on similar cases is available in the database and is accessible by the RPA robot.",
        "steps": "1: RPA robot identifies similar cases for the specific hospital or laboratory based on the test report. 2: Lab clinician or doctor receives information on similar cases found in the database."
      },
      {
        "name": "Review Past Blood Test Reports (ai-enabled-blood-test-report-analysis project)",
        "scenario": "Provide additional recommendations based on a test report on changes observed over time by reviewing past blood test reports of the same patient.",
        "actors": "['RPA Robot', 'Lab Clinician', 'Doctor']",
        "preconditions": "Past blood test reports of the same patient are available in the database and accessible by the RPA robot.",
        "steps": "1: RPA robot retrieves past blood test reports of the same patient. 2: Lab clinician or doctor receives additional recommendations based on changes observed over time."
      }
    ],
    "user_story": "As a lab clinician, I want to use the RPA Robot so that necessary details from blood test reports can be automatically extracted and saved in a global database, without any personally identifiable information. In addition, the system should use ML models to analyze these reports giving us prescriptive observations. Adding to this feature, the system should also highlight any similar cases related to a specific test report from our hospital or laboratory database. The system should also provide the facility to retrieve past blood test reports of the same patient, providing additional recommendations based on changes observed over time in a patient's test reports"
}

# Dummy project matching function (still uses LLaMA)
def isSameProject(usecase_list, usecase):
    response = togetherAIClient.chat.completions.create(
        model=LLAMA_MODEL,
        messages=[
            {"role": "system", "content": """
            We have a large dataset of usecases. We need to cluster the usecase
            that fall under the same project. Now you will be given a set of usecase and
            you will reply whether the usecase fall under the same project or not. The format
            of usecases is as follows:{
                "name": "usecase name",
                "scenario": "usecase description",
                "actors": "usecase actors",
                "preconditions": "usecase preconditions",
                "steps": "usecase steps"
            }

            output format should be as follows. Remember: Only output the JSON, with no explanation or extra text.:
            {
                "verdict": "True" or "False",
                "description": "description of the verdict"
            }
            """},
            {"role": "user", "content": f"usecase_list: {usecase_list}, usecase: {usecase}. Do they fall under the same project?"}
        ]
    )
    return eval(json.loads(response.choices[0].message.content).get("verdict"))

def generateUserStoryFromUsecaseGPT(usecase_list):
    response = openAIClient.responses.parse(
        model=GPT4_MODEL,
        input=[
            {"role": "system", "content": """
            You are a helpful assistant that generates a user story from a list of structured use cases. The format
            of usecases is as follows:{
                "name": "usecase name",
                "scenario": "usecase description",
                "actors": "usecase actors",
                "preconditions": "usecase preconditions",
                "steps": "usecase steps"
            }
            Use the style and structure of the provided examples to generate a coherent, high-level user story summarizing the goals and interactions in a single paragraph
            """ + "\n" + f"Example: \n{sample_usecase_1}" + "\n"},
            {"role": "user", 
             "content": f"usecase_list: {usecase_list}. Generate a user story based on the usecase list."
            }
        ],
        text_format=StoryRes
    )
    return response.output_parsed.story

def generateUserStoryFromUsecaseLLAMA(usecase_list):
    response = togetherAIClient.chat.completions.create(
        model=LLAMA_MODEL,
        messages=[
            {"role": "system", "content": """
            You are a helpful assistant that generates a user story from a list of structured use cases. The format
            of usecases is as follows:{
                "name": "usecase name",
                "scenario": "usecase description",
                "actors": "usecase actors",
                "preconditions": "usecase preconditions",
                "steps": "usecase steps"
            }
            Use the style and structure of the provided examples to generate a coherent, high-level user story summarizing the goals and interactions in a single paragraph
            """ + "\n" + f"Example: \n{sample_usecase_1}" + "\n"},
            {"role": "user", 
             "content": f"usecase_list: {usecase_list}. Generate a user story based on the usecase list."
            }
        ],
        response_format={
            "type": "json_object",
            "schema": StoryRes.model_json_schema()
        }
    )
    return json.loads(response.choices[0].message.content).get("story")

def main():
    # Load CSV file
    df = pd.read_csv('usecase2brd-dataset/cleaned_usecases.csv')
    usecases = df.to_dict(orient='records')

    # Initialize the usecase_list with the first usecase
    usecase_list = [{
        "name": usecases[0]["name"],
        "scenario": usecases[0]["scenario"],
        "precondition": usecases[0]["preconditions"],
        "actors": usecases[0]["actors"],
        "steps": usecases[0]["steps"]
    }]

    results = []

    # Process the remaining usecases
    for idx, row in enumerate(usecases[1:], start=1):
        usecase = {
            "name": row["name"],
            "scenario": row["scenario"],
            "precondition": row["preconditions"],
            "actors": row["actors"],
            "steps": row["steps"]
        }
        res = isSameProject(usecase_list, usecase)
        print(f"res = {res}")
        if res:
            usecase_list.append(usecase)
        else:
            descriptionGPT = generateUserStoryFromUsecaseGPT(usecase_list)
            descriptionLLAMA = generateUserStoryFromUsecaseLLAMA(usecase_list)
            results.append({
                "usecases": usecase_list,
                "user_story_gpt": descriptionGPT,
                "user_story_llama": descriptionLLAMA,
                "length": len(usecase_list)
            })
            usecase_list = [usecase]
            print(f"Story generated: {len(results)}")
            print("-" * 50)

    # Final flush
    if usecase_list:
        descriptionGPT = generateUserStoryFromUsecaseGPT(usecase_list)
        descriptionLLAMA = generateUserStoryFromUsecaseLLAMA(usecase_list)
        results.append({
            "usecases": usecase_list,
            "user_story_gpt": descriptionGPT,
            "user_story_llama": descriptionLLAMA,
            "length": len(usecase_list)
        })

    print("Total number of projects: ", len(results))
    file_name = f"userstory2usecase.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()