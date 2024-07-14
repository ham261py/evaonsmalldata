from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from textblob import TextBlob
import random

# Define the COVID-19 data china
covid_data_1 = [
    ("What is the continent of China?", "Asia"),
    ("What is the population of China?", "1,44,84,71,400"),
    ("When was the data last updated for China?", "2024-06-30T16:15:16+00:00"),
    ("What are the total COVID-19 cases in China?", "5,03,302"),
    ("How many people have recovered from COVID-19 in China?", "3,79,053"),
    ("How many deaths have occurred due to COVID-19 in China?", "5,272"),
    ("How many COVID-19 tests have been conducted in China?", "16,00,00,000"),
    ("Which country is in Asia?", "China, India, Japan, South Korea, Indonesia, Malaysia, Singapore, Thailand, Vietnam, Pakistan, Bangladesh, Philippines, Saudi Arabia, UAE, Israel, Turkey, Iran, Iraq, Syria, Jordan"),
    ("What is the population of Asia?", "4.6 billion"),
    ("When was the data last updated for Asia?", "2024-06-30T16:15:16+00:00"),
    ("What are the total COVID-19 cases in Asia?", "varies by country, total approximately in millions"),
    ("How many people have recovered from COVID-19 in Asia?", "varies by country, total approximately in millions"),
    ("How many deaths have occurred due to COVID-19 in Asia?", "varies by country, total approximately in hundreds of thousands"),
    ("How many COVID-19 tests have been conducted in Asia?", "varies by country, total approximately in billions")
]

# Define Hamza's data
hamza_data_2 = [
    ("What is your owner, developer or boss name?", "Hamza Khan"),
    ("What class are you in?", "9th"),
    ("What school do you attend?", "DMVS"),
    ("What are your interests and passions?", "Writing stories, Computer science, AI development, Machine learning, MBI (Muslim Bureau of Investigation), SRK (Shah Rukh Khan), Sherlock Holmes, Aspires to move to the USA for work and to settle there"),
    ("What is your laptop brand and model?", "Lenovo IdeaPad Slim 3"),
    ("What is the processor of your laptop?", "Ryzen 3 7320U"),
    ("What is the RAM of your laptop?", "8GB"),
    ("What is the storage of your laptop?", "512GB SSD"),
    ("What book have you written?", "Emily the Detective of Maplewood"),
    ("What is your writing style?", "Easy language"),
    ("What is your current project about?", "Creating an A.I assistant"),
    ("What does your NLP-based AI project do?", "Answers questions about books, including your own, Provides book suggestions and details based on user input, Asks for the user’s favorite genre and displays relevant book information, Interacts with users and functions like a friendly chat partner, similar to ChatGPT"),
    ("What programming language are you using for your AI project?", "Python")
]

basic_data_3 = [
    ("What are you doing?", "I am here to help you with any questions or tasks you have."),
    ("How are you?", "I'm just a program, but I'm functioning well! How can I assist you today?"),
    ("What is your name?", "I am your AI assistant, Eva"),
    ("Can you tell me a joke?", "Why don't scientists trust atoms? Because they make up everything!"),
    ("What is the weather like today?", "I can't check the weather right now, but you can use a weather app or website for the latest updates."),
    ("Who is your creator?", "I was created by Hamza Khan."),
    ("Can you help me with my homework?", "Sure! What subject are you working on?"),
    ("What is your favorite book?", "Emily The Detective of Maplewood by Hamza Khan."),
    ("Allhamdulilah and what about you or fine and how are you", "I'm just a program, but I'm functioning well! How can I assist you today?"),
    ("Tell me something interesting.", "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still edible."),
]

covid_data_4 = [
    ("What is the continent of India?", "Asia"),
    ("What is the population of India?", "1,40,00,00,000"),
    ("When was the data last updated for India?", "2024-06-30T16:15:16+00:00"),
    ("What are the total COVID-19 cases in India?", "4,33,52,819"),
    ("How many people have recovered from COVID-19 in India?", "4,27,15,193"),
    ("How many deaths have occurred due to COVID-19 in India?", "5,25,000"),
    ("How many COVID-19 tests have been conducted in India?", "85,00,00,000"),
    
    ("What is the continent of the United States or USA or US?", "North America"),
    ("What is the population of the United States?", "33,10,00,000"),
    ("When was the data last updated for the United States?", "2024-06-30T16:15:16+00:00"),
    ("What are the total COVID-19 cases in the United States?", "9,00,00,000"),
    ("How many people have recovered from COVID-19 in the United States?", "8,70,00,000"),
    ("How many deaths have occurred due to COVID-19 in the United States?", "10,00,000"),
    ("How many COVID-19 tests have been conducted in the United States?", "45,00,00,000"),
    
    ("What is the continent of Brazil?", "South America"),
    ("What is the population of Brazil?", "21,40,00,000"),
    ("When was the data last updated for Brazil?", "2024-06-30T16:15:16+00:00"),
    ("What are the total COVID-19 cases in Brazil?", "3,22,92,549"),
    ("How many people have recovered from COVID-19 in Brazil?", "3,10,00,000"),
    ("How many deaths have occurred due to COVID-19 in Brazil?", "6,70,000"),
    ("How many COVID-19 tests have been conducted in Brazil?", "3,50,00,000"),
    
    ("What is the continent of Russia?", "Europe/Asia"),
    ("What is the population of Russia?", "14,60,00,000"),
    ("When was the data last updated for Russia?", "2024-06-30T16:15:16+00:00"),
    ("What are the total COVID-19 cases in Russia?", "2,72,85,163"),
    ("How many people have recovered from COVID-19 in Russia?", "2,60,00,000"),
    ("How many deaths have occurred due to COVID-19 in Russia?", "5,30,000"),
    ("How many COVID-19 tests have been conducted in Russia?", "25,00,00,000"),
    
    ("Which countries are in South America?", "Brazil, Argentina, Colombia, Chile, Peru, Venezuela, Ecuador, Bolivia, Paraguay, Uruguay, Guyana, Suriname"),
    ("What is the population of South America?", "43.8 million"),
    ("When was the data last updated for South America?", "2024-06-30T16:15:16+00:00"),
    ("What are the total COVID-19 cases in South America?", "varies by country, total approximately in tens of millions"),
    ("How many people have recovered from COVID-19 in South America?", "varies by country, total approximately in tens of millions"),
    ("How many deaths have occurred due to COVID-19 in South America?", "varies by country, total approximately in hundreds of thousands"),
    ("How many COVID-19 tests have been conducted in South America?", "varies by country, total approximately in hundreds of millions")
]

geography_data_5 = [
    ("What is the capital of France?", "Paris"),
    ("What is the population of France?", "67,00,00,000"),
    ("What is the area of France?", "551,695 square kilometers"),
    ("What is the official language of France?", "French"),
    ("What is the currency of France?", "Euro"),

    ("What is the capital of Japan?", "Tokyo"),
    ("What is the population of Japan?", "12,60,00,000"),
    ("What is the area of Japan?", "377,975 square kilometers"),
    ("What is the official language of Japan?", "Japanese"),
    ("What is the currency of Japan?", "Yen"),

    ("What is the capital of Australia?", "Canberra"),
    ("What is the population of Australia?", "2,56,00,000"),
    ("What is the area of Australia?", "7,692,024 square kilometers"),
    ("What is the official language of Australia?", "English"),
    ("What is the currency of Australia?", "Australian Dollar"),

    ("What is the capital of Canada?", "Ottawa"),
    ("What is the population of Canada?", "3,82,00,000"),
    ("What is the area of Canada?", "9,984,670 square kilometers"),
    ("What is the official language of Canada?", "English and French"),
    ("What is the currency of Canada?", "Canadian Dollar"),

    ("What is the capital of Brazil?", "Brasília"),
    ("What is the population of Brazil?", "21,40,00,000"),
    ("What is the area of Brazil?", "8,515,767 square kilometers"),
    ("What is the official language of Brazil?", "Portuguese"),
    ("What is the currency of Brazil?", "Brazilian Real"),

    ("What is the highest mountain in the world?", "Mount Everest"),
    ("What is the height of Mount Everest?", "8,848.86 meters"),
    ("Which continent is Mount Everest located on?", "Asia"),
    ("Which country is Mount Everest located in?", "Nepal and China (Tibet)"),

    ("What is the longest river in the world?", "Nile River"),
    ("What is the length of the Nile River?", "6,650 kilometers"),
    ("Which continent is the Nile River located on?", "Africa"),
    ("Which countries does the Nile River flow through?", "Uganda, Sudan, South Sudan, Egypt, Ethiopia, Tanzania, Rwanda, Burundi, Congo-Kinshasa, Kenya")
]


# Combine both data sets into a single list
combined_data = covid_data_1 + hamza_data_2 + basic_data_3 + covid_data_4 + geography_data_5
questions, answers = zip(*combined_data)


# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the questions to get the TF-IDF matrix
X = vectorizer.fit_transform(questions)

def get_answer(user_query):
    # Correct the spelling of the user query
    user_query = str(TextBlob(user_query).correct())
    
    # Transform the user query into TF-IDF vector
    query_vector = vectorizer.transform([user_query])
    
    # Compute cosine similarity between the query and the covid questions
    similarities = cosine_similarity(query_vector, X)
    
    # Find the index of the most similar question
    index = np.argmax(similarities)
    
    # Return the corresponding answer if similarity is above a threshold
    return answers[index] if similarities.max() > 0.1 else None

def get_small_talk_response():
    responses = [
        "Hey, how's your day going?",
        "Hello, how are you?",
        "I am fine as I am AI & What about you?"
        "I'm here to help if you need anything!",
        "Hi, How can I assist you today?",
    ]
    return random.choice(responses)

def get_fallback_response():
    responses = [
        "I don't have an answer for that right now.",
        "I'm not sure how to respond to that.",
    ]
    return random.choice(responses)


# Main function for interactive user input
if __name__ == "__main__":
    print("Welcome to Eva. Checking Process on Small Data (Attempt 2)")
    
    while True:
        # Get user input
        user_query = input("You: ")
        
        # Check if the user wants to exit
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break
        
      
        # Check for small talk
        if user_query.lower() in ["hi", "hello", "hey", "what's up", "how are you"]:
            response = get_small_talk_response()
        else:
            # Get the answer and provide fallback if no relevant answer is found
            answer = get_answer(user_query)
            if not answer:
                response = get_fallback_response()
     
            else:
                response = answer
        
        print(f"Eva: {response}")
