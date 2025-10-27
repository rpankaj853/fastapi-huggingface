from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from ..services.model_config import load_text_generation_model

llm = load_text_generation_model()

# Step 2: Create a prompt
template = "Write a motivational quote about {topic}."
prompt = PromptTemplate(input_variables=["topic"], template=template)

# Step 3: Create chain
chain = LLMChain(llm=llm, prompt=prompt)


# Step 4: Invoke chain
def generate_llm_chain_quote(topic: str):
    print(f"Generating quote about: {topic}")
    response = chain.invoke({"topic": topic})
    return response
