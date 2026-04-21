"""Default follow-up prompts used when dynamic suggestions are unavailable."""

HEART_DEFAULT_FOLLOW_UPS = [
    "What kind of model is used?",
    "What is the dataset?",
    "What are the patient IDs?",
    "Does patient 2 have heart disease?",
    "Why did the model predict that this patient has heart disease?",
    "Show the patient's data",
    "If the patient's blood pressure was lower by 15, would the risk of heart disease decrease?",
    "How would it be possible to change this prediction?",
    "If this patient were 10 years younger, how would that affect the prediction?",
    "Which patients did the model misclassify most often?",
    "What is the false positive rate of the model?",
    "What is the precision and recall of the model?",
    "How well does the model generalize across different age groups?",
    "What is the AUC-ROC score of the model?",
    "How does the model generally decide whether a patient has heart disease?",
    "How do different risk factors interact with each other in the model's decision-making process?",
]
