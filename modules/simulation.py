import numpy as np
import matplotlib.pyplot as plt
from generators import generate_qam
from impairments import add_awgn, add_phase_noise
from normalization import normalize_signal

def calculate_sep(M, snr_range, phase_noise_sigma=0.02):
    sep_list = []
    num_symbols = 10000 # Μεγάλο δείγμα για ακρίβεια στατιστικής
    
    # Δημιουργία καθαρού αστερισμού για τον αποδιαμορφωτή
    # (Χρειαζόμαστε τα "πρότυπα" σημεία για να συγκρίνουμε)
    pure_symbols = generate_qam(M, M=M, normalize=True)
    unique_constellation = np.unique(pure_symbols)

    for snr in snr_range:
        # 1. Παραγωγή και αλλοίωση
        tx = generate_qam(num_symbols, M=M, normalize=True)
        rx = add_phase_noise(tx, phase_noise_sigma)
        rx = add_awgn(rx, snr)

        # 2. Αποδιαμόρφωση (ML Detector - βρες το πιο κοντινό σημείο)
        # Υπολογίζουμε την απόσταση κάθε ληφθέντος σημείου από όλα τα σημεία του αστερισμού
        errors = 0
        for i in range(num_symbols):
            distances = np.abs(rx[i] - unique_constellation)
            detected_idx = np.argmin(distances)
            if unique_constellation[detected_idx] != tx[i]:
                errors += 1
        
        sep_list.append(errors / num_symbols)
    
    return sep_list

# Εκτέλεση Προσομοίωσης
snr_range = np.arange(0, 22, 2)
plt.figure(figsize=(10, 6))

for M in [16, 64]:
    print(f"Simulating {M}-QAM...")
    sep = calculate_sep(M, snr_range)
    plt.semilogy(snr_range, sep, 'o-', label=f'{M}-QAM')

plt.grid(True, which='both')
plt.xlabel('SNR (dB)')
plt.ylabel('Symbol Error Probability (SEP)')
plt.title('SEP vs SNR with Phase Noise (sigma=0.02)')
plt.legend()
plt.show()