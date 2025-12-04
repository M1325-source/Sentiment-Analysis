from sentiment_utils import predict_from_texts, plot_sentiment_bar, wordcloud_for_sentiment
import matplotlib.pyplot as plt

def main():
    texts = [
        "I love this product, it is amazing and works perfectly!",
        "This is the worst experience I have ever had.",
        "The movie was fantastic, I enjoyed every moment of it.",
        "I am very disappointed with the service.",
        "What a wonderful day, feeling great and happy!",
        "The food was terrible and I will never come back.",
        "Customer support was helpful and resolved my issue quickly.",
        "The app keeps crashing, totally useless.",
        "Really impressed by the quality and performance.",
        "I hate how slow this website is."
    ]

    df_pred = predict_from_texts(texts)
    print(df_pred[["text", "label", "prob_pos"]])

    fig_bar = plot_sentiment_bar(df_pred)
    fig_pos_wc = wordcloud_for_sentiment(df_pred, 1)
    fig_neg_wc = wordcloud_for_sentiment(df_pred, 0)

    plt.show()

if __name__ == "__main__":
    main()
