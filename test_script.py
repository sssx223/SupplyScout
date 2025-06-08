import re
import json

def extract_material_and_specs(prompt):
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

# Test with different prompts
test_prompts = [
    "I need a quantum sensor with wavelength range 400-700nm, price under $5000",
    "Looking for silicon wafers, 300mm diameter, high purity",
    "Buy laser diodes that have 650nm wavelength and 5mW power",
    "Need optical fibers with single mode, 9/125 specifications",
    "quantum sensor wavelength 400-700nm price under $5000"
]

print("Testing Material Extraction:")
print("=" * 50)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\nTest {i}:")
    print(f"Input: '{prompt}'")
    result = extract_material_and_specs(prompt)
    print(f"Output: {result}")
    print(f"JSON: {json.dumps(result)}")
