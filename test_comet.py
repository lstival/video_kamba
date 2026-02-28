import comet_ml
import os

# Set API key manually for this test
api_key = "SNv4ks5JjUxZ1X0FhARDGt4SY"

try:
    print("Attempting to initialize Comet ML...")
    experiment = comet_ml.Experiment(
        api_key=api_key,
        project_name="video_mamba_test",
        auto_output_logging="simple"
    )
    print("Success! Experiment URL:", experiment.url)
    experiment.end()
except Exception as e:
    print("Failed to initialize Comet ML!")
    print("Error:", str(e))
