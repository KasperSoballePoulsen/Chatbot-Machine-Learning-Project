import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

df = pd.read_csv("heart.csv")
##Cleaning Data
#print(df.describe().to_string())

# Removing features that a chatbot user can't answer
df.drop(columns=["RestingBP", "Cholesterol", "MaxHR", "FastingBS", "RestingECG", "ST_Slope"], inplace=True)


#Check for null values
#print(df.isnull().sum())

#Replacing male with 1 and female with 0
df['Sex'] = df['Sex'].replace(['F'], 0)
df['Sex'] = df['Sex'].replace(['M'], 1)

#Replacing Y with 1 and N with 0 in ExerciseAngina
df['ExerciseAngina'] = df['ExerciseAngina'].replace(['N'], 0)
df['ExerciseAngina'] = df['ExerciseAngina'].replace(['Y'], 1)


df = pd.get_dummies(df, columns=["ChestPainType"])

# Converting bool to int
df = df.astype({col: int for col in df.columns if df[col].dtype == bool})


#print(df.head().to_string())







X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

##Creating model
from sklearn.model_selection import train_test_split

# Split the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

#print("Accuracy:", accuracy_score(y_test, y_pred))
#print("\nClassification report:\n", classification_report(y_test, y_pred))
#print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

from transformers import pipeline


# Global user data
user_data = {
    "Age": None,
    "Sex": None,
    "ChestPainType_ATA": None,
    "ChestPainType_NAP": None,
    "ChestPainType_TA": None,
    "ChestPainType_ASY": None,
    "ExerciseAngina": None,
    "Oldpeak": None
}

question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

def start_chatbot():
    print("Chatbot: Tell me about yourself and I will predict if you have a heart disease.")
    while True:
        user_input = input("User: ").strip()
        if user_input:
            conversation_log.append(user_input)
            sentiment = sentiment_analyzer(user_input)[0]
            sentiment_scores.append(sentiment)
        else:
            print("Chatbot: I didn't catch that. Could you try describing yourself or your symptoms?")
            continue
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        answer = generate_answer(user_input)
        print(answer)
    get_summary()


def generate_answer(user_input):

    context = user_input

    # Age
    age_result = question_answerer(question="How old are you?", context=context)
    if age_result['score'] > 0.3 and user_data['Age'] is None:
        digits = [int(s) for s in age_result["answer"].split() if s.isdigit()]
        if digits:
            user_data['Age'] = digits[0]

    # Sex
    sex_result = question_answerer(question="What is your gender?", context=context)
    if sex_result['score'] > 0.3 and user_data['Sex'] is None:
        answer = sex_result["answer"].lower()
        if "man" in answer or "male" in answer:
            user_data['Sex'] = 1
        elif "woman" in answer or "female" in answer:
            user_data['Sex'] = 0

    # Symptom → ChestPain + ExerciseAngina
    symptom_result = question_answerer(question="What symptoms are you experiencing?", context=context)
    if symptom_result['score'] > 0.3:
        if user_data['ChestPainType_TA'] is None:
            chestpain_dict = classify_chest_pain(symptom_result["answer"])
            # Only assign values if one of the 4 chest pain types is matched
            if any(chestpain_dict.values()):  # ASY included
                user_data['ChestPainType_TA'] = chestpain_dict["ChestPainType_TA"]
                user_data['ChestPainType_ATA'] = chestpain_dict["ChestPainType_ATA"]
                user_data['ChestPainType_NAP'] = chestpain_dict["ChestPainType_NAP"]
                user_data['ChestPainType_ASY'] = chestpain_dict["ChestPainType_ASY"]
        if user_data['ExerciseAngina'] is None:
            user_data['ExerciseAngina'] = int("exercise" in symptom_result["answer"].lower() or "effort" in symptom_result["answer"].lower())
    return get_next_response()



def get_next_response():
    response = ""
    if user_data['Age'] is None:
        response = "Chatbot: hmm okay, how old are you?"
    elif user_data['Sex'] is None:
        response = "Chatbot: please tell me if you are a man or a woman"
    elif user_data['ChestPainType_ATA'] is None:
        response = "Chatbot: Alright, are you experiencing any symptoms that could be related to heart disease?"
    elif user_data['ExerciseAngina'] is None:
        return "Chatbot: hmm describe how you feel when you exercise"
    elif user_data['Oldpeak'] is None:
        print("Chatbot: Can you describe how you feel after physical activity? Any chest discomfort, tiredness, etc?")
        followup = input("User: ").strip()
        if followup:
            user_data['Oldpeak'] = estimate_oldpeak(followup)
            return get_next_response()
        else:
            return "Chatbot: I didn't get that. Try describing how you feel after exercise."
    else:
        user_df = pd.DataFrame([user_data])
        user_df = user_df.reindex(columns=X.columns, fill_value=0)


        proba = model.predict_proba(user_df)[0][1]
        response = f"Chatbot: Based on your input, there's a {round(proba * 100, 1)}% chance you have heart disease.\n\tYou can enter information about another person, or type 'exit' to end the chat."

        reset_user_data()
    return response

def reset_user_data():
    user_data.update({
        "Age": None,
        "Sex": None,
        "ChestPainType_ATA": None,
        "ChestPainType_NAP": None,
        "ChestPainType_TA": None,
        "ChestPainType_ASY": None,
        "ExerciseAngina": None,
        "Oldpeak": None
    })






def classify_chest_pain(symptom_text):
    lemmas = lemmatize_text(symptom_text)


    non_anginal_keywords = {
        "sharp", "stab", "burn", "brief", "localize", "poke", "tender", "touch", "second",
        "ache", "sting", "irritate"
    }
    atypical_keywords = {
        "discomfort", "weird", "unclear", "vague", "dull", "mild", "come", "go", "not", "always", "effort",
        "sometimes", "occasionally", "uneasy"
    }
    typical_keywords = {
        "pressure", "tight", "squeeze", "heavy", "radiate", "jaw", "arm", "chest", "exercise", "stress",
        "hurt", "pain", "crush", "strain"
    }
    asymptomatic_keywords = {
        "no", "symptom", "fine", "nothing", "normal", "healthy", "okay", "well", "great"
    }


    chestpain_ta = any(word in lemmas for word in typical_keywords)
    chestpain_ata = any(word in lemmas for word in atypical_keywords) and not chestpain_ta
    chestpain_nap = any(word in lemmas for word in non_anginal_keywords) and not (chestpain_ta or chestpain_ata)
    chestpain_asy = any(word in lemmas for word in asymptomatic_keywords) and not (chestpain_ta or chestpain_ata or chestpain_nap)

    return {
        "ChestPainType_TA": int(chestpain_ta),
        "ChestPainType_ATA": int(chestpain_ata),
        "ChestPainType_NAP": int(chestpain_nap),
        "ChestPainType_ASY": int(chestpain_asy)
    }

def estimate_oldpeak(symptom_text):
    doc = nlp(symptom_text.lower())
    lemmas = [token.lemma_ for token in doc]

    high_depression = {"severe", "pain", "strong", "collapse", "breathe", "hurt", "terrible", "can't"}
    moderate_depression = {"shortness", "breath", "discomfort", "tightness", "heavy", "chest", "slight"}
    no_depression = {"no", "pain", "fine", "mild", "nothing", "normal", "tired"}

    if any(word in lemmas for word in high_depression):
        return 2.5
    elif any(word in lemmas for word in moderate_depression):
        return 1.5
    elif any(word in lemmas for word in no_depression):
        return 0.0
    else:
        return 1.0


test_data = {
    "Age": 22,
    "Sex": 1,
    "ChestPainType_ATA": 0,
    "ChestPainType_NAP": 0,
    "ChestPainType_TA": 0,
    "ChestPainType_ASY": 1,
    "ExerciseAngina": 0,
    "Oldpeak": 1.0
}






import spacy
nlp = spacy.load("en_core_web_sm")

def lemmatize_text(text):
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]


def get_summary():
    print("\n--- Conversation Summary ---")

    # Word frequency excluding stopwords/punctuation
    all_text = " ".join(conversation_log).lower()
    words = [word.strip(string.punctuation) for word in all_text.split()]
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and w]
    freq = Counter(words).most_common(10)
    print("\nTop keywords:")
    for word, count in freq:
        print(f"  {word}: {count} times")

    # Sentiment analysis result
    positive = sum(1 for s in sentiment_scores if s['label'] == 'POSITIVE')
    negative = sum(1 for s in sentiment_scores if s['label'] == 'NEGATIVE')

    print(f"\nSentiment: {positive} positive, {negative} negative messages")

    if positive > negative:
        print("You seemed generally positive during our chat 😊")
    elif negative > positive:
        print("You seemed mostly negative during the conversation 😟")
    else:
        print("Your sentiment was balanced.")





from collections import Counter
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load models
sentiment_analyzer = pipeline("sentiment-analysis")
conversation_log = []  # stores all user messages
sentiment_scores = []  # stores sentiment per message



start_chatbot()
