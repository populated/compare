from compare.core import TextComparer

if __name__ == "__main__":
    texts = [
        "I like cats",
        "I love dogs",
        "I enjoy birds",
        "I adore cats"
    ]
    comparer = TextComparer(texts)

    results = comparer.compare_texts("I heart cats")
    # -> {'closest': 'I like cats', 'score': 0.6191302964899972, 'similar': ['I adore cats']}
    print(results)
