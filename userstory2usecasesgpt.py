import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import sys
import json
import time
import random

# Load environment variables
load_dotenv()

GPT4_MODEL = "gpt-4o"

openAIClient = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Usecase(BaseModel):
    name: str
    scenario: str
    actors: str
    preconditions: str
    steps: str

class UsecaseRes(BaseModel):
    usecases: list[Usecase]

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

def generateUsecasesFromUserStoryGPT(user_story, max_retries=5):
    """Generate usecases from a user story with rate limit handling"""
    retry_count = 0
    base_delay = 5  # Start with a 5-second delay
    
    while retry_count < max_retries:
        try:
            response = openAIClient.responses.parse(
                model=GPT4_MODEL,
                input=[
                    {"role": "system", "content": """
                    You are a helpful assistant that generates usecases list from a user story. The format
                    of usecases should be as follows:{
                        "name": "usecase name",
                        "scenario": "usecase description",
                        "actors": "usecase actors",
                        "preconditions": "usecase preconditions",
                        "steps": "usecase steps"
                    }
                    Use the style and structure of the provided examples to generate a coherent, high-level usecases list
                    """ + "\n" + f"Example: \n{sample_usecase_1}" + "\n"},
                    {"role": "user", 
                     "content": f"user_story: {user_story}. Generate a usecases list based on the user story."
                    }
                ],
                text_format=UsecaseRes
            )
            return response.output_parsed.usecases
            
        except openai.RateLimitError as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"\nMax retries ({max_retries}) reached. Giving up.")
                raise e
            
            # Extract wait time from error message if available
            wait_time = None
            error_msg = str(e)
            if "Please try again in" in error_msg and "s" in error_msg.split("Please try again in")[1]:
                try:
                    wait_time_str = error_msg.split("Please try again in")[1].split("s")[0].strip()
                    wait_time = int(wait_time_str)
                except (ValueError, IndexError):
                    pass
            
            # Calculate delay with exponential backoff and jitter
            if wait_time:
                delay = wait_time + random.uniform(1, 5)  # Add some jitter
            else:
                delay = base_delay * (2 ** (retry_count - 1)) + random.uniform(1, 5)
            
            print(f"\nRate limit hit. Retrying in {delay:.1f} seconds... (Attempt {retry_count}/{max_retries})")
            time.sleep(delay)
        
        except Exception as e:
            # For other errors, don't retry
            print(f"\nError: {str(e)}")
            raise e


def save_to_json(dataset, filename="userstory2usecase_dataset.json"):
    """Save the dataset to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved to {filename}")

def load_from_json(filename="userstory2usecase_dataset.json"):
    """Load the dataset from a JSON file if it exists"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def main():
    # Set output filename
    output_file = "userstory2usecase_dataset_new.json"
    
    # Load the CSV data
    df = pd.read_csv("userstory2usecase-dataset/RealWorldUsecaseDataset - Sheet3.csv")
    df = df.iloc[:, 1:]
    
    # Load existing dataset if it exists
    dataset = load_from_json(output_file)
    
    # Determine starting index (for resume functionality)
    start_idx = len(dataset)
    
    if start_idx > 0:
        print(f"Resuming from index {start_idx} (already processed {start_idx} stories)")
    
    # Process each user story
    try:
        for idx in range(start_idx, len(df)):
            user_story = df.iloc[idx]["UserStory"]
            print(f"\nProcessing story {idx+1}/{len(df)}: {user_story[:100]}...")
            
            # Generate usecases for the current story
            usecases = generateUsecasesFromUserStoryGPT(user_story)
            
            # Convert Pydantic models to dictionaries
            usecase_dicts = [usecase.dict() for usecase in usecases]
            
            # Create a new entry
            entry = {
                "user_story": user_story,
                "usecases": usecase_dicts
            }
            
            # Add to dataset
            dataset.append(entry)
            
            # Save after each successful processing
            save_to_json(dataset, output_file)
            
            print(f"Saved story {idx+1} with {len(usecases)} usecases")
            
            # Add a delay between API calls to avoid rate limits
            # Even with retry logic, it's better to avoid hitting limits in the first place
            if idx < len(df) - 1:  # Don't delay after the last item
                delay = random.uniform(10, 15)  # Random delay between 10-15 seconds
                print(f"Waiting {delay:.1f} seconds before processing next story...")
                time.sleep(delay)
            
    except Exception as e:
        print(f"\nError processing story at index {start_idx + len(dataset)}: {str(e)}")
        print(f"You can resume from this point by running the script again.")
        # Save what we have so far
        save_to_json(dataset, output_file)
        raise e
    
    print(f"\nCompleted processing all {len(df)} user stories.")
    print(f"Final dataset saved to {output_file}")

if __name__ == "__main__":
    main()