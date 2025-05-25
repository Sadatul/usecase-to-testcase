import json
import sys
import os

def main():
    # Check if filename is provided as command line argument
    if len(sys.argv) != 2:
        print("Usage: python stats.py <filename.json>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    
    try:
        # Load the JSON file
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Count total usecases
        total_usecases = sum(len(item['usecases']) for item in data)
        
        # Print results
        print(f"Analyzing: {filename}")
        print(f"Total user stories: {len(data)}")
        print(f"Total usecases: {total_usecases}")
        print(f"Average usecases per story: {total_usecases/len(data):.2f}")
        
        # Optional: Show distribution of usecases per story
        print("\nUsecases per story distribution:")
        for i, item in enumerate(data, 1):
            print(f"Story {i}: {len(item['usecases'])} usecases")
            
    except json.JSONDecodeError:
        print(f"Error: '{filename}' is not a valid JSON file")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()