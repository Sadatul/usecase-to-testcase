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
    "user_story": "A registered patient may use the HealthHub system to schedule, reschedule, or cancel appointments with available doctors. Each patient must be verified through a secure login and an active patient ID. The system shall allow the patient to search for available doctors based on specialization, location, or availability. Once a doctor is selected, the system shall display the earliest available slots. If the patient chooses a slot that is already booked due to concurrent requests, the system shall offer alternative nearby slots. For first-time appointments, the patient must complete a medical history form before confirmation. The patient shall receive a confirmation message upon successful booking, including the appointment date, time, and location. The system must also allow patients to upload insurance details and verify them in real time. For video consultations, the system shall verify the availability of both doctor and patient internet connections and device compatibility at least 15 minutes prior to the appointment start.",
    "usecases": [
      {
        "name": "Patient Login",
        "scenario": "A registered patient logs into the HealthHub system using secure credentials.",
        "actors": "['Patient']",
        "preconditions": "Patient is registered and has valid login credentials.",
        "steps": "1: Patient navigates to the HealthHub login page. 2: Patient enters username and password. 3: System verifies credentials and grants access upon successful verification."
      },
      {
        "name": "Search Available Doctors",
        "scenario": "A patient searches for doctors based on specialization, location, or availability.",
        "actors": "['Patient']",
        "preconditions": "Patient is logged into the HealthHub system.",
        "steps": "1: Patient accesses the search function in the HealthHub system. 2: Patient selects search criteria such as specialization, location, or availability. 3: System displays a list of available doctors matching the criteria."
      },
      {
        "name": "Select Doctor and View Slots",
        "scenario": "A patient selects a doctor and views their earliest available slots.",
        "actors": "['Patient']",
        "preconditions": "Patient has searched for available doctors and a list is displayed.",
        "steps": "1: Patient selects a doctor from the search results. 2: System displays the earliest available appointment slots for the selected doctor."
      },
      {
        "name": "Book Appointment Slot",
        "scenario": "A patient books an available appointment slot with a doctor.",
        "actors": "['Patient']",
        "preconditions": "Patient has selected a doctor and viewed available slots.",
        "steps": "1: Patient selects an available appointment slot. 2: System checks slot availability and confirms booking. 3: System sends a confirmation message with appointment details."
      },
      {
        "name": "Reschedule Appointment",
        "scenario": "A patient reschedules an existing appointment with a doctor.",
        "actors": "['Patient']",
        "preconditions": "Patient has an existing confirmed appointment.",
        "steps": "1: Patient navigates to the 'My Appointments' section. 2: Patient selects the appointment to reschedule. 3: System displays available new slots for the same or a different doctor. 4: Patient selects a new slot. 5: System confirms the new appointment and cancels the previous one. 6: Patient receives updated appointment confirmation."
      },
      {
        "name": "Cancel Appointment",
        "scenario": "A patient cancels a previously booked appointment.",
        "actors": "['Patient']",
        "preconditions": "Patient has a confirmed appointment.",
        "steps": "1: Patient navigates to the 'My Appointments' section. 2: Patient selects the appointment to cancel. 3: System prompts for confirmation. 4: Patient confirms cancellation. 5: System cancels the appointment and sends a cancellation confirmation message."
      },
      {
        "name": "Handle Appointment Slot Conflict",
        "scenario": "System offers alternative slots when a chosen slot is already booked.",
        "actors": "['System']",
        "preconditions": "Patient has selected an already booked appointment slot.",
        "steps": "1: System detects that the selected slot is already booked. 2: System provides alternative nearby appointment slots. 3: Patient selects an alternative slot and confirms booking."
      },
      {
        "name": "Complete Medical History Form",
        "scenario": "Patient completes a medical history form for first-time appointments.",
        "actors": "['Patient']",
        "preconditions": "Patient is booking a first-time appointment and an available slot is selected.",
        "steps": "1: System prompts patient to complete a medical history form. 2: Patient fills out the form with required medical history details. 3: Patient submits the form to complete the appointment booking process."
      },
      {
        "name": "Upload and Verify Insurance Details",
        "scenario": "Patient uploads and verifies insurance details in real-time.",
        "actors": "['Patient', 'System']",
        "preconditions": "Patient is logged into the HealthHub system.",
        "steps": "1: Patient navigates to the insurance details section. 2: Patient uploads insurance documents. 3: System verifies the uploaded insurance details in real-time."
      },
      {
        "name": "Verify Video Consultation Readiness",
        "scenario": "System verifies connectivity and compatibility for video consultations.",
        "actors": "['System']",
        "preconditions": "Patient has booked a video consultation.",
        "steps": "1: System checks internet connectivity for both doctor and patient 15 minutes before the appointment. 2: System verifies device compatibility for video consultation. 3: System notifies both parties if any issues are detected."
      }
    ]
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

                    Try to make the usecases atomic and do not combine multiple usecases into one via steps field. Each action should be a separate usecase. 
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
    output_file = "testset.json"
    
    # Load the CSV data
    df = pd.read_csv("TestStories.csv")
    # df = df.iloc[:, 1:]
    
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