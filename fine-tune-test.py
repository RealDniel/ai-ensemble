from transformers import AutoModelForCausalLM
import os
from dotenv import load_dotenv

#Load environment variables and get keys
load_dotenv('.env')




base_model = AutoModelForCausalLM.from_pretrained("** PUT MODEL HERE **", dtype="auto", device_map="auto")

