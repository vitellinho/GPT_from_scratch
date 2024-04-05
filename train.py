import torch

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
encode = lambda s: [stoi[c] for c in s] # encoder: übergibt str an stoi, kriegt int zurück
decode = lambda l: "".join([itos[i] for i in l]) # decoder: übergibt int an itos, kriegt str zurück

#print(encode("hii there"))
#print(decode(encode("hii there")))

# Encoding der gesamten txt Datei & speichern dessen in einem torch.Tensor (jeder einzelne Buchstabe wird tokenisiert)
data = torch.tensor(encode(text), dtype=torch.long)

#print(data.shape)
#print(data[:1000]) # Die ersten 1000 Buchstaben der txt Datei werden dem GPT in dieser Form übergeben

## - Training des Transformers - ##

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
torch.manual_seed(1337) # wird festgelegt, damit torch.randint später nicht immer zufällige verschiedene Werte generiert ## kann gelöscht werden ?!
batch_size = 4 # Die maximale Anzahl an data-chunks in einem data-batch


def get_batch(split):
    data = train_data if split == "train" else val_data # data = train_data wenn split = "train", sonst val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Generierung der 4 zufälligen Startindizes für 4 chunks
    x = torch.stack( [data[i:i+block_size] for i in ix]) # Tensor x mit Eingabedaten für Batch wird erstellt
    y = torch.stack( [data[i+1:i+block_size+1] for i in ix]) # Tensor y mit Zieldaten (x + 1) für Batch wird erstellt
    return x, y # x (Eingabedaten) und y (Zieldaten, auch Label genannt) werden durch Funktion zurückgegeben

# Visualisierung vom input x (xb) und target y (yb) aus Funktion get_batch
xb, yb = get_batch("train")
print("inputs:")
print(xb)
print(xb.shape)
print("targets:")
print(yb)
print(yb.shape)

print("----")

# Visualisierung von Eingabewerten x und vom Transformer zu vorhersagenden Zielwerten (label) y
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")