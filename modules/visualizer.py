#ΜΕΤΑΤΡΟΠΗ ΙQ σε ΕΙΚΟΝΕΣ
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

def iq_to_constellation_image(iq_samples, size=(224, 224), dot_size=0.5, color='black'):
    """
    Μετατρέπει IQ δείγματα σε εικόνα αστερισμού σταθερού μεγέθους.
    """
    # Δημιουργία figure χωρίς άξονες και περιθώρια
    fig = plt.figure(figsize=(4, 4), dpi=size[0]/4)
    ax = fig.add_axes([0, 0, 1, 1]) # Καταλαμβάνει όλο το χώρο
    
    # Σχεδίαση των σημείων (scatter plot)
    ax.scatter(iq_samples.real, iq_samples.imag, s=dot_size, c=color, marker='o')
    
    # Ορισμός ορίων ώστε ο αστερισμός να είναι πάντα κεντραρισμένος
    # Χρησιμοποιούμε σταθερό όριο με βάση τη μέγιστη τιμή για να διατηρηθεί η γεωμετρία
    limit = np.max(np.abs(iq_samples)) * 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # Αφαίρεση αξόνων
    ax.axis('off')
    
    # Μετατροπή του plot σε εικόνα (Buffer)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=size[0]/4)
    plt.close(fig)
    buf.seek(0)
    
    # Άνοιγμα με την PIL και μετατροπή σε RGB (σταθερό μέγεθος)
    img = Image.open(buf).convert('RGB').resize(size)
    
    return img