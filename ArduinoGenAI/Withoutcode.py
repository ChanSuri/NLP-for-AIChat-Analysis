from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize intent classification model
intent_model_name = "bert-base-uncased"  # Replace with your fine-tuned model path or name
tokenizer = BertTokenizer.from_pretrained(intent_model_name)
model = BertForSequenceClassification.from_pretrained(intent_model_name)

# Define confidence threshold
CONFIDENCE_THRESHOLD = 0.7  # Decision threshold

# Example context
user_context = {"last_action": "loaded_sketch"}

def classify_user_intent(user_request: str, context: dict = None):
    """
    Classify user intent using rules, context, and the model.
    """
    # Rule-based matching
    keywords_for_code = ["modify", "debug", "variable", "in the code", "function"]
    keywords_for_independent = ["generate", "create", "new program", "independent"]

    if any(kw in user_request for kw in keywords_for_code):
        return 0, 1.0  # Clearly requires loading the sketch
    if any(kw in user_request for kw in keywords_for_independent):
        return 1, 1.0  # Clearly does not require loading the sketch

    # Context-based decision
    if context and "modify" in user_request and context.get("last_action") == "loaded_sketch":
        return 0, 0.9  # Context suggests loading the sketch

    # Model-based prediction
    inputs = tokenizer(user_request, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    confidence, intent_id = torch.max(probabilities, dim=1)  # Get the highest confidence and intent ID

    return intent_id.item(), confidence.item()

def handle_user_request(user_request: str, context: dict):
    """
    Main flow: Use classification results, confidence, and thresholds to decide whether to load the sketch.
    """
    intent_id, confidence = classify_user_intent(user_request, context)

    if confidence < CONFIDENCE_THRESHOLD:
        # Prompt user for clarification if confidence is low
        return f"I'm not sure if the current sketch needs to be loaded. Please clarify your request.\nConfidence: {confidence:.2f}"

    # Output results based on intent
    if intent_id == 0:
        return f"The current sketch needs to be loaded to process the request.\nConfidence: {confidence:.2f}"
    else:
        return f"The current sketch does not need to be loaded; the request can be handled independently.\nConfidence: {confidence:.2f}"

# Example user requests
user_requests = [
    "Modify the code to adjust the frequency.",
    "Generate a new LED control program.",
    "What global variables are present in the current code?",
    "Help debug the sensor output data."
]

# Test the main flow
for request in user_requests:
    response = handle_user_request(request, user_context)
    print(f"User request: \"{request}\"\nResponse: {response}\n")
