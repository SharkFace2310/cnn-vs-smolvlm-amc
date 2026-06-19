import os
import json
import numpy as np
from generators import *
from impairments import *
from normalization import *
from visualizer import iq_to_constellation_image

# 1. Ορισμός Φακέλων Αποθήκευσης
DATASET_DIR = "constellation_dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# 2. Ορισμός Παραμέτρων Εκπαίδευσης & Γενίκευσης (Σύμφωνα με την εκφώνηση)
SNR_TRAIN = {"low": 0, "medium": 10, "high": 20}
SNR_GEN = {"inter_low": 5, "inter_high": 15}

IQ_LEVELS = {
    "none":   {"g": 0.0, "p": 0.0},
    "low":    {"g": 0.05, "p": 2.0},  # 5% gain, 2 deg phase
    "medium": {"g": 0.15, "p": 7.0},  # 15% gain, 7 deg phase
    "high":   {"g": 0.25, "p": 12.0}  # 25% gain, 12 deg phase
}

def generate_and_save_dataset(samples_per_mod=50):
    train_metadata = []
    gen_metadata = []
    sample_id = 0

    mod_list = [
        '4-ASK', '8-ASK', 
        'BPSK', 'QPSK', 
        '4-HQAM', '16-HQAM', '64-HQAM', 
        '16-QAM', '32-QAM', '64-QAM', '128-QAM', '256-QAM',
        '16-APSK', '32-APSK', '64-APSK', '128-APSK'
    ]

    print(f"Starting dataset generation... Total classes: {len(mod_list)}")

    for mod in mod_list:
        # --- TRAINING SAMPLES ---
        for snr_cat, snr_val in SNR_TRAIN.items():
            for iq_cat, iq_vals in IQ_LEVELS.items():
                for _ in range(samples_per_mod // 4): # Μοιράζουμε τα samples
                    label = create_sample(sample_id, mod, snr_val, snr_cat, iq_vals, iq_cat)
                    train_metadata.append(label)
                    sample_id += 1

        # --- GENERALIZATION SAMPLES (Ενδιάμεσες συνθήκες) ---
        for snr_cat, snr_val in SNR_GEN.items():
            # Επιλέγουμε ένα "μεσαίο" IQ imbalance για το generalization test
            iq_vals = {"g": 0.10, "p": 4.5} 
            iq_cat = "intermediate"
            for _ in range(samples_per_mod // 10): 
                label = create_sample(sample_id, mod, snr_val, snr_cat, iq_vals, iq_cat)
                gen_metadata.append(label)
                sample_id += 1

    # Αποθήκευση σε δύο ξεχωριστά αρχεία
    with open(os.path.join(DATASET_DIR, 'labels_train.json'), 'w') as f:
        json.dump(train_metadata, f, indent=4)
    with open(os.path.join(DATASET_DIR, 'labels_generalization.json'), 'w') as f:
        json.dump(gen_metadata, f, indent=4)

    print(f"Done! Created {len(train_metadata)} training samples and {len(gen_metadata)} generalization samples.")

def create_sample(sample_id, mod, snr, snr_cat, iq_vals, iq_cat):
    # 1. Generate symbols
    if 'ASK' in mod: symbols = generate_ask(1000, M=int(mod.split('-')[0]))
    elif 'BPSK' == mod: symbols = generate_psk(1000, M=2)
    elif 'QPSK' == mod: symbols = generate_psk(1000, M=4)
    elif 'HQAM' in mod: symbols = generate_hqam(1000, alpha=1.0) # Απλοποίηση για το παράδειγμα
    elif 'APSK' in mod: symbols = generate_apsk_by_order(1000, M=int(mod.split('-')[0]))
    else: symbols = generate_qam(1000, M=int(mod.split('-')[0]))

    # 2. Add Impairments
    symbols = normalize_signal(symbols)
    symbols = add_iq_imbalance(symbols, iq_vals['g'], iq_vals['p'])
    symbols = add_phase_noise(symbols, 0.02) # Σταθερό phase noise
    symbols = add_awgn(symbols, snr)

    # 3. Save Image
    file_name = f"sample_{sample_id}.png"
    img = iq_to_constellation_image(symbols)
    img.save(os.path.join(IMAGES_DIR, file_name))

    return {
        "id": sample_id,
        "file": file_name,
        "modulation": mod,
        "snr_db": snr,
        "snr_level": snr_cat,
        "iq_imbalance_level": iq_cat
    }

if __name__ == "__main__":
    generate_and_save_dataset(samples_per_mod=40)