# ΛΟΓΙΚΗ ΥΛΟΠΟΙΗΣΗ ΓΙΑ  ΤΑ ΣΗΜΑΤΑ

import numpy as np

# Υλοποιηση Συναρτησης που παραγει τυχαια συμβολα ASK
def generate_ask(num_symbols, M=4):

    levels = np.arange(-(M-1), M, 2)

    indices = np.random.randint(0, M, num_symbols)
    symbols = levels[indices]

    return symbols.astype(complex)

# Υλοποιηση Συναρτησης που παραγει τυχαια συμβολα ΒPSK-QPSK
def generate_psk(num_symbols, M=2, phase_offset=True):

    #αν phase offset=FALSE  δεν περιστρεφεται ο αστερισμςο κατα π/Μ  και τα συμβολα ειναι 0,2π/M,4π/M,...,2π(M-1)/M
    offset = np.pi / M if phase_offset else 0

    phases = 2 * np.pi * np.arange(M) / M + offset

    indices = np.random.randint(0, M, num_symbols)

    symbols = np.exp(1j * phases[indices])

    return symbols
# Yλοποιηση Συναρτησης που παραγει τυχαια συμβολα M-HQAM
def generate_hqam(num_symbols, alpha=1.0, normalize=True):
    levels = np.array([
        -(1 + 2*alpha),
        -1,
         1,
         (1 + 2*alpha)
    ])

    I, Q = np.meshgrid(levels, levels)
    constellation = I.flatten() + 1j * Q.flatten()

    indices = np.random.randint(0, 16, num_symbols)
    symbols = constellation[indices]

    if normalize:
        symbols = symbols / np.sqrt(np.mean(np.abs(constellation)**2))

    return symbols
# Yλοποιηση Συναρτησης που παραγει τυχαια συμβολα M-QAM
def generate_qam(num_symbols, M=16, normalize=True):
    """
    Παραγωγή συμβόλων QAM. 
    Υποστηρίζει M = 4, 16, 32, 64, 128, 256.
    """
    if M == 4: # QPSK
        return generate_psk(num_symbols, M=4)
    
    # Για τετραγωνικές QAM (16, 64, 256)
    if M in [16, 64, 256]:
        side = int(np.sqrt(M))
        levels = np.arange(-(side-1), side, 2)
        # Δημιουργία πλέγματος
        x, y = np.meshgrid(levels, levels)
        constellation = x.flatten() + 1j * y.flatten()
    
    # Για Cross QAM (32, 128) - Αν τις χρειάζεσαι
    elif M in [32, 128]:
        # Η λογική παραμένει ως έχει στον κώδικά σου 
        # (συνήθως αφαιρούμε τις γωνίες από το επόμενο μεγαλύτερο τετράγωνο)
        side = int(np.sqrt(M * 2)) 
        levels = np.arange(-(side-1), side, 2)
        x, y = np.meshgrid(levels, levels)
        constellation = x.flatten() + 1j * y.flatten()
        # Κρατάμε μόνο τα M σημεία με τη μικρότερη ενέργεια
        dist = np.abs(constellation)**2
        thresh = sorted(dist)[M-1]
        constellation = constellation[dist <= thresh]

    # Επιλογή τυχαίων συμβόλων
    indices = np.random.randint(0, M, num_symbols)
    symbols = constellation[indices]

    return symbols

# Υλοποιηση Συναρτησης που παραγει τυχαια συμβολα M-APSK


def generate_apsk(num_symbols, rings_symbols=[4, 12], rings_radii=[1.0, 2.6], offsets=None):
    if offsets is None:
        offsets = [np.pi / n for n in rings_symbols]
        
    constellation = []
    for n, r, off in zip(rings_symbols, rings_radii, offsets):
        # Χρήση του off που έρχεται από το config
        phases = 2 * np.pi * np.arange(n) / n + off
        constellation.extend(r * np.exp(1j * phases))
    
    constellation = np.array(constellation)
    indices = np.random.randint(0, len(constellation), num_symbols)
    return constellation[indices]

def generate_apsk_by_order(num_symbols, M=16):
    configs = {
        16:  {'n': [4, 12],            'r': [1.0, 2.7],           'offsets': [np.pi/4, 0]},
        32:  {'n': [4, 12, 16],         'r': [1.0, 2.5, 4.3],      'offsets': [np.pi/4, 0, 0]},
        64:  {'n': [4, 12, 20, 28],      'r': [1.0, 2.4, 3.8, 5.2], 'offsets': [np.pi/4, 0, 0, 0]},
        # 128-APSK: Δομή 4 δακτυλίων (16+32+40+40) για καλύτερη οπτική διακριτότητα στο CNN
        128: {'n': [16, 32, 40, 40],    'r': [1.0, 2.2, 3.5, 5.0], 'offsets': [0, 0, 0, 0]}
    }
    
    if M not in configs:
        raise ValueError(f"M={M} APSK is not supported.")
        
    return generate_apsk(
        num_symbols, 
        rings_symbols=configs[M]['n'], 
        rings_radii=configs[M]['r'],
        offsets=configs[M]['offsets']
    )



