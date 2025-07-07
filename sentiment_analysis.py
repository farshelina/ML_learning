from transformers import pipeline

def load_model():
    print("正在加载情感分析模型...")
    return pipeline("text-classification", 
                   model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentence(classifier, text):
    result = classifier(text)[0]
    return f"情感倾向: {result['label']} (置信度: {result['score']:.2%})"

if __name__ == "__main__":
    classifier = load_model()
    print("模型加载完成！输入'quit'退出程序")
    
    while True:
        user_input = input("\n请输入待分析的句子: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
        if not user_input:
            continue
            
        print(analyze_sentence(classifier, user_input))
