# main.py
import json
import subprocess
import sys
import re
from typing import Dict, List

# This is the exact extraction function from your working test script.
def extract_material_and_specs(prompt: str) -> Dict[str, str]:
    """
    Extract a dictionary from the prompt where the key is the material 
    and the value is other specifications.
    """
    prompt_lower = prompt.lower()
    
    # Find material as first noun phrase after keywords
    material_patterns = [
        r'(?:need|looking for|searching for|want|require|find) ([a-z ]+?)(?: with| that| which|,|$)',
        r'(?:buy|purchase|get|order) ([a-z ]+?)(?: with| that| which|,|$)',
        r'^([a-z ]+?)(?: with| that| which|,)'
    ]
    
    material = None
    for pattern in material_patterns:
        material_match = re.search(pattern, prompt_lower)
        if material_match:
            material = material_match.group(1).strip()
            break
    
    # Fallback: take first 2-3 words if no pattern matches
    if not material:
        words = prompt_lower.split()
        material = ' '.join(words[:min(3, len(words))])
    
    # Extract specs as everything after material
    specs_start = prompt_lower.find(material)
    if specs_start != -1:
        specs_start += len(material)
        specs = prompt_lower[specs_start:].strip(' ,.with')
    else:
        specs = ""
    
    # Clean up material name
    material = material.strip('a an the ')
    
    # Build dictionary
    result = {material: specs if specs else None}
    return result

def main():
    """
    Main function to extract materials from prompts and search for vendors.
    """
    print("=== SupplyScout Material Vendor Search ===\n")
    
    test_prompts = [
        "I need a quantum sensor with wavelength range 400-700nm, price under $5000",
        "Looking for silicon wafers, 300mm diameter, high purity",
        "Buy laser diodes that have 650nm wavelength and 5mW power",
        "Need optical fibers with single mode, 9/125 specifications",
        "invalid prompt to test error handling" # Example of a prompt that might fail
    ]

    # This will store the final aggregated results.
    final_results = {
        "total_searches": len(test_prompts),
        "successful_searches": 0,
        "total_vendors": 0,
        "all_vendors": [],
        "errors": []
    }

    # --- THE FIXED FOR LOOP ---
    # This loop is now simple and directly inside the main function.
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Processing Search {i}/{len(test_prompts)} ---")
        print(f"Prompt: '{prompt}'")
        
        try:
            # Step 1: Extract the dictionary from the prompt.
            material_dict = extract_material_and_specs(prompt)
            material_name = list(material_dict.keys())[0]
            print(f"  -> Extracted Material: '{material_name}'")

            # Step 2: Call your Modal script to search for vendors.
            print(f"  -> Calling vendor search for '{material_name}'...")
            result = subprocess.run([
                "modal", "run", "vendor_search_modal.py",
                "--material", material_name
            ], capture_output=True, text=True, check=True) # Using check=True for easier error handling

            # Step 3: Read the results from the generated file.
            output_filename = f"vendors_{material_name.replace(' ', '_')}.json"
            with open(output_filename, 'r') as f:
                vendor_data = json.load(f)
            
            print(f"  -> Success! Found {vendor_data.get('total_vendors', 0)} vendors.")

            # Step 4: Aggregate the results.
            final_results["successful_searches"] += 1
            final_results["total_vendors"] += vendor_data.get("total_vendors", 0)
            
            # Add material info to each vendor for tracking
            for vendor in vendor_data.get("vendors", []):
                vendor["searched_material"] = material_name
                final_results["all_vendors"].append(vendor)

        except subprocess.CalledProcessError as e:
            # This catches errors if the modal script fails.
            error_message = f"Modal script failed for material '{material_name}'. Error: {e.stderr}"
            print(f"  -> Error: {error_message}")
            final_results["errors"].append({"prompt": prompt, "error": error_message})
        except FileNotFoundError:
            # This catches errors if the JSON file isn't created.
            error_message = f"Results file not found for material '{material_name}'."
            print(f"  -> Error: {error_message}")
            final_results["errors"].append({"prompt": prompt, "error": error_message})
        except Exception as e:
            # This catches any other unexpected errors.
            error_message = f"An unexpected error occurred for prompt '{prompt}'. Error: {str(e)}"
            print(f"  -> Error: {error_message}")
            final_results["errors"].append({"prompt": prompt, "error": error_message})

    # Save the final aggregated results to a file
    output_file = "supplyscout_aggregated_results.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print the final summary
    print("\n\n=== FINAL SUMMARY ===")
    print(f"Successfully processed {final_results['successful_searches']} out of {final_results['total_searches']} searches.")
    print(f"Found a total of {final_results['total_vendors']} unique vendors.")
    print(f"Aggregated results saved to: {output_file}")
    if final_results['errors']:
        print(f"Encountered {len(final_results['errors'])} errors. Check the JSON file for details.")


if __name__ == "__main__":
    main()
