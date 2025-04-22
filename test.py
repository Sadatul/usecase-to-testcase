import pandas as pd
from sentence_transformers import SentenceTransformer
def compute_similarity(text1, text2, model):
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    similarity = model.similarity(embedding1, embedding2)
    return float(similarity[0][0])

def compare_scenario_to_next_precondition(csv_path, num_rows=None, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # Load model
    model = SentenceTransformer(model_name, trust_remote_code=True)
    
    # Load data
    df = pd.read_csv(csv_path)

    # Limit rows if specified
    if num_rows:
        df = df.head(num_rows)

    results = []
    count = 0
    ls = []
    # Compare scenario of row[i] with preconditions of row[i+1]
    for i in range(len(df) - 1):
        scenario = df.loc[i, "scenario"]
        precondition = df.loc[i + 1, "preconditions"]
        similarity = compute_similarity(scenario, precondition, model)
        res = similarity < 0.2
        results.append({
            "Scenario": scenario,
            "Next Precondition": precondition,
            "Similarity": round(similarity, 4),
            "Result": res
        })

        if df.loc[i, "result"] != res:
            count += 1
            ls.append(i)

    result_df = pd.DataFrame(results)
    print(f"Number of rows not matched: {count}, {ls}")
    return result_df

# Example usage:
csv_file_path = "usecase2brd-dataset/testset.csv"  # Replace with your actual CSV path
result = compare_scenario_to_next_precondition(csv_file_path, num_rows=107, model_name="sentence-transformers/all-MiniLM-L6-v2")
print(result)
result.to_csv("scenario_similarity_results.csv", index=False)

# import pandas as pd
# csv_file_path = "usecase2brd-dataset/cleaned_usecases.csv"  # Replace with your actual CSV path

# df = pd.read_csv(csv_file_path)
# df = df.head(107)
# df["result"] = False
# df.to_csv("usecase2brd-dataset/testset.csv", index=False)

