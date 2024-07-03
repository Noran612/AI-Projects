import json

data = [
    {
        "prompt": "What is the status of my order?",
        "completion": "Can you please provide your order ID?"
    },
    {
        "prompt": "The order ID is 123456.",
        "completion": "Your order with ID 123456 is currently being processed and will be shipped within 3 days."
    },
    {
        "prompt": "What is the return policy for items purchased at our store?",
        "completion": "You can return most items within 30 days of purchase for a full refund or exchange. Items must be in their original condition, with all tags and packaging intact. Please bring your receipt or proof of purchase when returning items."
    },
    {
        "prompt": "Are there any items that cannot be returned under this policy?",
        "completion": "Yes, certain items such as clearance merchandise, perishable goods, and personal care items are non-returnable. Please check the product description or ask a store associate for more details."
    },
    {
        "prompt": "How will I receive my refund?",
        "completion": "Refunds will be issued to the original form of payment. If you paid by credit card, the refund will be credited to your card. If you paid by cash or check, you will receive a cash refund."
    },
    {
        "prompt": "I would like to speak to a human representative.",
        "completion": "Sure, I can help with that. Can you please provide your full name, email, and phone number?"
    }
]

# Write the JSONL file
with open('dataset.jsonl', 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')

print("JSONL file created successfully.")
