from transformers import AutoModelForCausalLM
import os
from dotenv import load_dotenv

#Load environment variables and get keys
load_dotenv('.env')

#Decided on mistral 7b instruct for writing agent
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", dtype="auto", device_map="auto")
