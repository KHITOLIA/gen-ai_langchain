import warnings
warnings.filterwarnings("ignore")

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, Optional,Literal
from dotenv import load_dotenv
load_dotenv()

# Initialize the chat model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",  # Hugging Face model repo
    task = "text-generation",
    temperature = 0.1,
    max_new_tokens= 1000,
)
model = ChatHuggingFace(llm = llm)

# Schema 

class Review(TypedDict):
    key_themes: Annotated[str, "write down all the key themes from the review in a list"]
    summary: Annotated[str, "Give me brief Summary of the given review"]
    sentiment: Annotated[Literal["pos", "neg", "neu"], "just return me the sentiment of the given review"]
    pros: Annotated[Optional[list[str]], "Write all the pros inside a list if you found , if not found just say no pros given by the user"]
    cons: Annotated[Optional[list[str]], "Write all the cons inside a list if you found , if not found just say no pros given by the user"]
    rating: Annotated[str, "give the rating of the product mentioned if present otherwise return nothing."]

structured_model = model.with_structured_output(Review)

review = '''I was confused. I wanted to try the OnePlus 15 but settled for OnePlus Nord 5 for many reasons.

Here's a flat comparison

Pros & Cons

OnePlus Nord 5 5G 📱
Excellent performance. Outstanding selfie camera - one of the best till date by any phone, big display, great gaming modes and support.

OnePlus 15 Pro 📱
Amazing Overall Performance: Great display, battery is outstanding, more compact.

BATTERY

OnePlus Nord 5 5G 📱
6800 mAh (typical) battery is good enough unless you want to go for that slightly extra life.

OnePlus 15 Pro 📱
At 7300 mAh battery, you do get a 500 mAh difference. It it matters, go for it.

DISPLAY

OnePlus Nord 5 5G 📱
The 6.83″ AMOLED, 1272×2800 px is far better than the OnePlus pro 15. 144 Hz refresh rate is good enough even for rash gaming use.

OnePlus 15 Pro 📱
6.78″ LTPO AMOLED, 1272×2772 px is not as impressive as the Nord 5 5G but the 15 pro gets a better refresh rate at 165 Hz.

GAMING

OnePlus Nord 5 5G 📱
This phone is one of the best when it comes to mobile gaming. Qualcomm Snapdragon 8s Gen 3 (4 nm) processer is is good enough although not as good as the OnePlus 15 Pro. No heating for 40 long minutes in a high end graphics game. I did not try it beyond that.

OnePlus 15 Pro 📱
Qualcomm Snapdragon 8 Elite Gen 5 (3 nm) processors is one of the best in the market today. However there are heat buildup challenges that I experienced during extended heavy use or long gaming sessions.

CAMERA

OnePlus Nord 5 5G 📱
Although modest, the OnePlus Nord 5 main camera and the front selfie camera perform far better than the ones provided in OnePlus 15 pro. 50 MP selfie camera and the 50 MP main camera in Nord 5 are some of the best in the industry till date (except for Samsungs main camera and iPhones video capabilities).

OnePlus 15 Pro 📱
Triple — 50 MP main + 50 MP + 50 MP setup (main + ultra-wide + telephoto) are all outstanding features. Sadly these cameras fail to deliver. The images vary from great to very poor. The 32 MP front camera is another disappointment at this price point, as the images are not consistent. Not that its any better than its competitors or even OnePlus Nord 5 5G.

SPEAKER
Both OnePlus Nord 5 5G and OnePlus 15 Pro, speakers are loud and clear. I don't hear much difference there.'''

result = structured_model.invoke(review)
for i in result.items():
    print(i[0], ":", i[1])
    print(" ")