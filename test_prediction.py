from NexText.components.model_prediction import ModelPrediction


predictor = ModelPrediction()


text = """
I recently purchased a new smartphone, and I'm really impressed with its features. The camera quality is amazing, and the battery life lasts for almost two days on a single charge. The processor is also very fast, allowing me to multitask without any lag. Overall, it's a great device for the price.
"""




summary = predictor.predict(text)


print("\n" + "=" * 60)
print("INPUT:")
print(text.strip())

print("\nSUMMARY:")
print(summary)

print("=" * 60)