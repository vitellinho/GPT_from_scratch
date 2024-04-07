import torch
import torch.nn as nn
from torch.nn import functional as F

# txt Datei Shakespear öffnen
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Übersicht der Shakespear txt Datei
#print("length of dataset in characters: ", len(text))
#print(text[:1000])

# Extrahierung aller vorkommenden Zeichen/Buchstaben/Zahlen + Menge dessen aus der txt Datei
chars = sorted(list(set(text)))
vocab_size = len(chars)

#print("".join(chars))
#print(vocab_size)


## - Character Tokenizer - ##

# Zuordnung von Buchstaben (ch, token) zu Zahlen (i, embedding) und durch encode/decode Übersetzer (ch<>i) erschaffen
stoi = { ch:i for i, ch in enumerate(chars) } # stoi: String to Index Wörterbuch (als Dictionary)
itos = { i:ch for i, ch in enumerate(chars) } # itos: Index to String Wörterbuch (als Dictionary)
encode = lambda s: [stoi[c] for c in s] # encoder: übergibt str (s) an stoi, kriegt int (c) zurück
decode = lambda l: "".join([itos[i] for i in l]) # decoder: übergibt int (l) an itos, kriegt str (i) zurück

#print(encode("hii there"))
#print(decode(encode("hii there")))

# Encoding der gesamten txt Datei & speichern dessen in einem torch.Tensor (jeder einzelne Buchstabe wird tokenisiert)
data = torch.tensor(encode(text), dtype=torch.long)

#print(data.shape)
#print(data[:1000]) # Die ersten 1000 Buchstaben der txt Datei werden dem GPT in dieser Form übergeben


## - Vorbereitung für Training des Transformers - ##

# Aufteilung des Datensatzes in train & validation set
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Da man Transformer nicht mit kompletter txt auf einmal trainiert, werden anhand von block_size 'data-chunks' erstellt
block_size = 8 # Die maximale Kontextlänge einer Vorhersage
train_data[:block_size+1]

# Visualisierung, wie innerhalb eines Chunks Verbindungen zwischen Tokens und Folgetokens erstellt wird (Wortnetz)
#x = train_data[:block_size]
#y = train_data[1:block_size+1]
#for t in range(block_size):
#    context = x[:t+1]
#    target = y[t]
#    print(f"when input is {context} the target: {target}")

# Neben der Dimension 'data-chunks', muss auch die 2. Dimension 'data-batches' (ein batch aus chunks) definiert werden
torch.manual_seed(1337) # wird festgelegt, damit torch.randint später nicht immer zufällige verschiedene Werte generiert
batch_size = 4 # Die maximale Anzahl an data-chunks in einem data-batch

# Erstellung Batch
def get_batch(split):
    data = train_data if split == "train" else val_data # data = train_data wenn split = "train", sonst val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Generierung der 4 zufälligen Startindizes für 4 chunks
    x = torch.stack( [data[i:i+block_size] for i in ix]) # Tensor x mit Eingabedaten für Batch wird erstellt
    y = torch.stack( [data[i+1:i+block_size+1] for i in ix]) # Tensor y mit Zieldaten (x + 1) für Batch wird erstellt
    return x, y # x (Eingabedaten) und y (Zieldaten, auch Label genannt) werden durch Funktion als Batch zurückgegeben

# Visualisierung vom input x (xb) und target y (yb) aus Funktion get_batch
xb, yb = get_batch("train")
#print("inputs:")
#print(xb)
#print(xb.shape)
#print("targets:")
#print(yb)
#print(yb.shape)

#print("----")

# Visualisierung von Eingabewerten x und vom Transformer zu vorhersagenden Zielwerten (label) y
#for b in range(batch_size):
#    for t in range(block_size):
#        context = xb[b, :t+1]
#        target = yb[b,t]
#        print(f"when input is {context.tolist()} the target: {target}")

torch.manual_seed(1337)


## Neural Network: Bigram Language Model ##

class BigramLanguageModel(nn.Module): # Klasse BigramLanguageModel, welche aus nn.Module erbt

    # nn.Embedding: Erstellung Embeddingmatrix vocab_size*vocab_size
    def __init__(self, vocab_size): # vocab_size: Menge der einzigartigen Zeichen/Buchstaben/Zahlen in txt
        super().__init__() # super(): Aufruf der init Methode aus der parent class nn.Module
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # logits: Input wird übergeben & verweist auf Zeilen aus Emb.matrix, aus diesen Zeilen wird logits-Matrix erstellt
    def forward(self, idx, targets=None): # targets = None: so bleibt targets optional
        logits = self.token_embedding_table(idx)

        # loss soll nur berechnet werden, wenn auch targets übergeben werden
        if targets is None:
            loss = None
        else:
            # reshapen der logits- und targets-Matrizen, da diese sonst nicht mit Loss Function fitten
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)  # Loss Function, welche Vorhersage (logits) und targets gegenüberstellt

        return logits, loss # für Eingabetext relevante Zeilen aus Embeddingmatrix & loss werden zurückgegeben

    # Funktion des Modells: Generierung weiterer strings basierend auf input (idx)
    def generate(self, idx, max_new_tokens): # max_new_tokens: max. Anzahl an zu generierenden strings
        for _ in range(max_new_tokens):
            logits, loss = self(idx) # erhalten der predictions (logits) & loss zum gegebenen Input
            logits = logits[:, -1, :] # Fokus wird auf letzten Token gesetzt da auf diesen nächster Token generiert wird
            probs = F.softmax(logits, dim=-1) # softmax: erzeugt durch Vorhersage die Wahrscheinlichkeit für next-Token
            idx_next = torch.multinomial(probs, num_samples=1) # multinomal: wählt basierend auf Wahrsch. next-Token aus
            idx = torch.cat((idx, idx_next), dim=1) # cat: Input (idx) wird um weiteren String (idx_next) erweitert
        return idx # Input + darauf basierend generierte strings werden zurück gegeben

# Erstellung Objekt & übergabe input (xb) & target (yb) & Visualisierung
m = BigramLanguageModel(vocab_size)
out = m(xb, yb)
#print(out.shape)
logits, loss = m(xb, yb)
#print(logits.shape)
#print(loss)

# print der vom Modell decodeten und generierten strings basierend auf input
#print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


## Model-Training ##

# Erstellung eines PyTorch Optimizers (Art: AdamW)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# Erhöhung der batch_size für Training von 4 auf 32
batch_size = 32

# Trainingsschleife
for steps in range(10000):

    xb, yb = get_batch("train") # Generierung eines neuen zufälligen batches (bestehend aus xb/yb) aus den trainingsdata

    logits, loss = m(xb, yb) # Generierung Vorhersage logits, gegenüberstellung dieser mit targets yb & Berechnung loss
    optimizer.zero_grad(set_to_none=True) # Gradienten = 0, sodass im next-step Gradienten neu berechnet werden können
    loss.backward() # Berechnung der Gradienten, sprich die Änderungen des Loss-Werts bei Änderung der Parameter
    optimizer.step() # Optimierung der Parameter basierend auf dem berechneten Gradienten

# .item() gibt zur Überwachung des Trainings den skalaren Wert des Loss zurück
#print(loss.item())

# erneuter print der vom Modell decodeten und generierten strings basierend auf input, diesmal nach dem Training
#print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


