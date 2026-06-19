import numpy as np 

def add_awgn(iq_samples, snr_db):
    """
    Προσθήκη λευκού γκαουσιανού θορύβου (AWGN).
    snr_db: ο λόγος σήματος προς θόρυβο σε dB.
    """
    # Υπολογισμός ισχύος σήματος
    sig_power = np.mean(np.abs(iq_samples)**2)
    
    # Μετατροπή SNR από dB σε γραμμική κλίμακα
    snr_linear = 10**(snr_db / 10.0)
    
    # Υπολογισμός ισχύος θορύβου
    noise_power = sig_power / snr_linear
    
    # Δημιουργία σύνθετου θορύβου (I και Q συνιστώσες)
    # Η ισχύς noise_power μοιράζεται στις δύο συνιστώσες, άρα std = sqrt(noise_power/2)
    noise = (np.random.normal(0, np.sqrt(noise_power/2), len(iq_samples)) +
             1j * np.random.normal(0, np.sqrt(noise_power/2), len(iq_samples)))
    
    return iq_samples + noise

def add_phase_noise(iq_samples, sigma):
    """
    Προσθήκη Wrapped Gaussian Phase Noise.
    sigma: η τυπική απόκλιση της φάσης σε ακτίνια.
    """
    # Δημιουργία θορύβου φάσης (ανεξάρτητος για κάθε δείγμα)
    noise_phase = np.random.normal(0, sigma, len(iq_samples))
    
    # Εφαρμογή στα δείγματα μέσω εκθετικού
    return iq_samples * np.exp(1j * noise_phase)

def add_iq_imbalance(iq_samples, gain_imbalance=0.0, phase_imbalance_deg=0.0):
    """
    Προσθήκη IQ imbalance μέσω widely-linear μοντέλου.
    
    gain_imbalance (g): π.χ. 0.1 σημαίνει 10% διαφορά πλάτους.
    phase_imbalance_deg (phi): η γωνιακή απόκλιση σε ΜΟΙΡΕΣ.
    """
    g = gain_imbalance
    # Μετατροπή των μοιρών σε radians για τους τριγωνομετρικούς υπολογισμούς
    phi = np.radians(phase_imbalance_deg)

    # Συντελεστές Widely Linear Model
    # Το μοντέλο αυτό περιγράφει πώς το I και το Q σήμα "ανακατεύονται"
    alpha = np.cos(phi/2) + 1j * g * np.sin(phi/2)
    beta  = g * np.cos(phi/2) - 1j * np.sin(phi/2)

    # Το ληφθέν σήμα είναι γραμμικός συνδυασμός του ορθού σήματος και του συζυγούς του
    return alpha * iq_samples + beta * np.conj(iq_samples)