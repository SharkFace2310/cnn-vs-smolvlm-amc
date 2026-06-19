import json
import os

def create_vlm_jsonl(input_json, output_jsonl):
    if not os.path.exists(input_json):
        print(f"Σφάλμα: Το αρχείο {input_json} δεν βρέθηκε!")
        return
        
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in data:
            vqa_pair = {
                "image": entry['file'],
                "conversations": [
                    {
                        "role": "user",
                        "content": "Analyze this signal constellation. What is the modulation and the SNR level?"
                    },
                    {
                        "role": "assistant",
                        "content": f"The signal uses {entry['modulation']} modulation and the SNR level is {entry['snr_level']}."
                    }
                ]
            }
            f.write(json.dumps(vqa_pair) + '\n')
    print(f"Επιτυχία! Το αρχείο {output_jsonl} δημιουργήθηκε.")

if __name__ == "__main__":
    input_path = 'constellation_dataset/labels_train.json' 
    create_vlm_jsonl(input_path, 'vlm_dataset.jsonl')