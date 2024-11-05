import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM

def merge_models(base_model_name, adapter_model_path, merged_model_path):
    # Load the base model
    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # Load the PEFT model
    print(f"Loading PEFT model: {adapter_model_path}")
    model = PeftModel.from_pretrained(model, adapter_model_path)

    # Merge the models
    print("Merging models...")
    model = model.merge_and_unload()

    # Save the merged model
    print(f"Saving merged model to: {merged_model_path}")
    model.save_pretrained(merged_model_path)

    print("Merge completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a base model with a PEFT model")
    parser.add_argument("base_model_name", type=str, help="Name or path of the base model")
    parser.add_argument("adapter_model_path", type=str, help="Path to the PEFT adapter model")
    parser.add_argument("merged_model_path", type=str, help="Path to save the merged model")

    args = parser.parse_args()

    merge_models(args.base_model_name, args.adapter_model_path, args.merged_model_path)