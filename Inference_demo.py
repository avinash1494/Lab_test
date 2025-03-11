import requests
import time


# Configuration
PREDICTION_API_URL = "http://34.100.239.84:5000/prediction_API"
RESULTS_API_URL = "http://34.100.239.84:5000/prediction_results/"
ACCESS_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI1MTc2ZWE1Zi02Y2E3LTQyYTQtYjgyZS1iZDIzYjE2M2I0MmUiLCJleHAiOjE4OTgxNTY2NDd9.bsSt-zFF4cwpBlK4eVj-dCZ2UyxSqsX-B1xBhCYLbOs"

HEADERS = {
    "x-access-token": ACCESS_TOKEN,
    "Content-Type": "application/json"
}

# Data for prediction
data = {
    "workflowId": "69419ed9-b00a-4e12-8e51-b8b89992ac7b",
    "data": {
        "id_1": "Explain Spiral model ?"
    }
}  

def make_prediction():
    """Send data to the prediction API and retrieve the task ID."""
    try:
        print("Sending prediction request...")
        response = requests.post(PREDICTION_API_URL, headers=HEADERS, json=data)
        print("Response Status Code:", response.status_code)
        print("Response Text:", response.text)
        return response.text
    except requests.exceptions.RequestException as e:
        print("Error while sending prediction request:", e)
        return None

def track_results(task_id):
    """Track the results of the prediction using the task ID."""
    if not task_id:
        print("Invalid Task ID. Exiting.")
        return

    results_url = f"{RESULTS_API_URL}{task_id}"
    print("Tracking request URL:", results_url)
    try:
        while True:
            response = requests.get(results_url, headers={"x-access-token": ACCESS_TOKEN})
            print("Tracking Response Status Code:", response.status_code)
            print("Tracking Response Text:", response.text)
            response.raise_for_status()
            
            result = response.json()
            if result.get("status") == "completed":
                print("Prediction Completed. Results:", result)
                break
            else:
                print("Task is still in progress. Retrying in 5 seconds...")
                time.sleep(3)
    except requests.exceptions.RequestException as e:
        print("Error while tracking prediction results:", e)

if __name__ == "__main__":
    task_id = make_prediction()
    print("Task ID:", task_id)
    track_results(task_id)
