"""Tiny sentiment dataset: 60 hand-written + 120 template-generated sentences. Real projects load thousands of rows from a CSV/DB."""

POSITIVE = [
    "I love this phone, the battery lasts all day",
    "Great service and very friendly staff",
    "The food was delicious and fresh",
    "Absolutely fantastic movie, I enjoyed every minute",
    "Excellent quality, worth every penny",
    "The delivery was fast and the packaging was perfect",
    "Wonderful experience, I will definitely come back",
    "This laptop is fast and the screen is beautiful",
    "Amazing value for the price",
    "Super helpful support team solved my issue quickly",
    "The hotel room was clean and comfortable",
    "I am very happy with my purchase",
    "Brilliant book, I could not put it down",
    "The app is smooth and easy to use",
    "Best coffee in town, highly recommend",
    "Very pleased, it works perfectly",
    "Lovely atmosphere and great music",
    "The teacher was inspiring and kind",
    "Great product, exceeded my expectations",
    "Smooth checkout and quick refund, impressive",
    "The concert was awesome and the crowd was fun",
    "Really good headphones with clear sound",
    "Fantastic customer care, thank you",
    "The pizza tasted amazing",
    "Perfect fit and very comfortable shoes",
    "I enjoyed the course, it was clear and useful",
    "Terrific update, everything feels faster",
    "The staff were polite and helpful",
    "Good value and reliable performance",
    "Beautiful design and sturdy build",
]

NEGATIVE = [
    "I hate this phone, the battery dies in an hour",
    "Terrible service and rude staff",
    "The food was cold and tasteless",
    "Awful movie, a complete waste of time",
    "Poor quality, a waste of money",
    "The delivery was late and the box was damaged",
    "Horrible experience, I will never come back",
    "This laptop is slow and the screen is dim",
    "Overpriced and disappointing",
    "Useless support team, they ignored my emails",
    "The hotel room was dirty and noisy",
    "I am very unhappy with my purchase",
    "Boring book, I could not finish it",
    "The app keeps crashing and is hard to use",
    "Worst coffee ever, do not recommend",
    "Very disappointed, it stopped working in a week",
    "Awful atmosphere and terrible music",
    "The teacher was boring and unhelpful",
    "Bad product, far below my expectations",
    "Checkout failed twice and the refund never came",
    "The concert was awful and the sound was bad",
    "Really bad headphones with a crackling sound",
    "Dreadful customer care, nobody answered",
    "The pizza tasted burnt",
    "Terrible fit and very uncomfortable shoes",
    "I disliked the course, it was confusing and useless",
    "Horrible update, everything feels slower",
    "The staff were impolite and unhelpful",
    "Bad value and unreliable performance",
    "Ugly design and flimsy build",
]


# Template-generated sentences: a toy way to get MORE data that shares sentiment words with the hand-written
# ones.  (Real projects collect real data; synthetic data like this is only for learning the workflow.)
NOUNS = ["food", "service", "movie", "phone", "hotel", "app", "book", "staff", "delivery", "product",
         "laptop", "course", "coffee", "concert", "update"]
POS_WORDS = ["great", "excellent", "amazing", "wonderful", "fantastic", "good", "lovely", "perfect", "brilliant", "fast"]
NEG_WORDS = ["terrible", "awful", "horrible", "bad", "poor", "dreadful", "boring", "useless", "slow", "disappointing"]
TEMPLATES = ["The {noun} was {word}", "I think the {noun} is {word}", "{word} {noun}, really", "Honestly the {noun} felt {word}"]


def _generate(words, n, seed):
    import random
    rng = random.Random(seed)                      # seeded -> the same dataset on every run
    out = set()                                    # set removes duplicate sentences
    while len(out) < n:
        out.add(rng.choice(TEMPLATES).format(noun=rng.choice(NOUNS), word=rng.choice(words)))
    return sorted(out)


def load_dataset():
    """Return (texts, labels) where labels are the strings 'positive' / 'negative'. 60 hand-written + 120 generated."""
    pos = POSITIVE + _generate(POS_WORDS, 60, seed=1)
    neg = NEGATIVE + _generate(NEG_WORDS, 60, seed=2)
    return pos + neg, ["positive"] * len(pos) + ["negative"] * len(neg)
