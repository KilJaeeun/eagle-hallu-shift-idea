"""
Create sample training data for Phase 1/2 testing.
In production, replace with actual ShareGPT data download.
"""
import json
import os

os.makedirs('data', exist_ok=True)

sample_data = [
    {
        'conversations': [
            {'from': 'human', 'value': 'What is machine learning?'},
            {'from': 'gpt', 'value': 'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.'}
        ]
    },
    {
        'conversations': [
            {'from': 'human', 'value': 'Explain neural networks.'},
            {'from': 'gpt', 'value': 'Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information using connectionist approaches.'}
        ]
    },
    {
        'conversations': [
            {'from': 'human', 'value': 'What is deep learning?'},
            {'from': 'gpt', 'value': 'Deep learning is a subset of machine learning that uses neural networks with many layers to learn representations of data with multiple levels of abstraction.'}
        ]
    },
    {
        'conversations': [
            {'from': 'human', 'value': 'How does gradient descent work?'},
            {'from': 'gpt', 'value': 'Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function by moving in the direction of steepest descent.'}
        ]
    },
    {
        'conversations': [
            {'from': 'human', 'value': 'What is a transformer model?'},
            {'from': 'gpt', 'value': 'A transformer is a deep learning architecture that uses self-attention mechanisms to process sequential data, enabling parallel processing and capturing long-range dependencies.'}
        ]
    },
] * 40  # Create 200 samples for testing

with open('data/sharegpt_train.json', 'w') as f:
    json.dump(sample_data, f, indent=2)

print(f'Created sample training data with {len(sample_data)} samples')
