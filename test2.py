# 5c77cafee1258a0813bcf8da976349ac5128ebd8bdd9d13fe84bc6664fa7af14

from together import Together
import json
client = Together()

usecase_list = [
            {
    "name":"User Enters Patient Notes",
    "scenario":"A user enters patient notes in an Automated AI-based Clinical Coding App for medical coding.",
    "preconditions":"User has access to the Clinical Coding App and patient notes are available for coding.",
    },
    {
    "name":"Patient Notes Added in Data Service",
    "scenario":"The patient notes entered by the user in an Automated AI-based Clinical Coding App for medical coding are added in the Data Service for further processing.",
    "preconditions":"Patient notes are successfully entered in the App.",
},
  {
    "name":"Assignment of ICD Codes to patient notes using Language Classification Model for an Automated AI-based Clinical Coding App",
    "scenario":"ICD codes are assigned to the patient notes using a language classification model from the AI center.",
    "preconditions":"Patient notes are available in the Data Service for processing.",
},
  {
    "name":"Review of Assigned ICD Codes to Patient Notes in An Automated AI-based Clinical Coding App by Medical Coder",
    "scenario":"The assigned ICD codes are reviewed by a medical coder in the UiPath Action Center.",
    "preconditions":"ICD codes have been assigned to the patient notes and are available for review in the Action Center of the app."},
    {
    "name":"Data Entry in Legacy Application by Robot",
    "scenario":"A user enters patient notes in an Automated AI-based Clinical Coding App for medical coding. After it is assgined ICD code by AI, it is approved or rejected by the medical coder. If the assigned codes are approved by the medical coder, the data is entered in the legacy application by a robot.",
    "preconditions":"The assigned ICD codes have been approved by the medical coder in the Action Center.",
  }
]

usecase = {
    "name":"Blood Test Report Extraction (ai-enabled-blood-test-report-analysis project)",
    "scenario":"Automated extraction of required fields from the lab report for a specific type of blood test.",
    "preconditions":"Lab reports are available in a format readable by the RPA robot and the UiPath Document Understanding is able to classify the reports.",
}
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
            "description": "description of the verdict",
         }
         """
        },
        {
            "role": "user",
            "content": f"usecase_list: {usecase_list}, usecase: {usecase}. Do they fall under the same project?"
        }
        ],
)

print(json.loads(response.choices[0].message.content).get("verdict"))

# {
#     "name":"Blood Test Report Extraction (ai-enabled-blood-test-report-analysis project)",
#     "scenario":"Automated extraction of required fields from the lab report for a specific type of blood test.",
#     "preconditions":"Lab reports are available in a format readable by the RPA robot and the UiPath Document Understanding is able to classify the reports.",
#     }