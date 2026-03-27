import warnings
warnings.filterwarnings("ignore")
from typing import TypedDict,Annotated, Optional, Literal
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-Llama/Llama-3.2-3B-Instruct",
    task = "text-generation",
    temperature = 0,
    max_new_tokens = 1000
    )

model = ChatHuggingFace(llm = llm)

class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief Summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative or positive"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list if present do not hallucinate otherwise"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list if present do not hallucinate otherwise"]
    name: Annotated[str, "Write the name of reviewer"]

structured_output = model.with_structured_output(Review)
result = structured_output.invoke("""I bought it in the pre-booking sale on 27th Nov and luckily got the 1-day delivery on 28th Nov. I have been an Iqoo smartphone user since 2000 and I totally rely on this brand. Iqoo never let you down be it performance, looks, services, or camera. This time iqoo 15 is a beast. Camera is way better than oneplus. Camera has so many features like the Bokeh effect in video recording which usually comes with iphone. Performance I tested with games is remarkable. 144 fps smooth running of COD. No lag.
It looks premium in hand. 2k display is top notch, you will love watching movies or videos on this. Heating issues are not yet found. Slightly heats up during fast charging otherwise during usuage there is no heating issue. Back RGB Halo light around the camera looks elegant. Overall I feel it was worth spending this much amount on iqoo 15.
Note - With the pre-booking I got this phone 16gb variant in 68k (including all card offer
- Tushar khitoliya
""")
for points in result.items():
    print(points)

