import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # Wie viele data_chunks werden parallel verarbeitet
block_size = 256 # Wie viele vorhergehende Token werden für die Vorhersage des nächsten Tokens verwendet
max_iters = 5000 # max. Anzahl an Trainings-Iterationen
eval_interval = 500 # Intervall (zu max_iters), zu dem das Modell evaluiert wird
learning_rate = 3e-4 # bestimmt die Größe der Schritte, die der optimizer während des Trainings unternimmt
device = 'cuda' if torch.cuda.is_available() else 'cpu' # soll mit GPU ('cuda') oder CPU ('cpu') trainiert werden
eval_iters = 200 # ähnlich wie eval_interval, jedoch spezifiziert eval_iters eine feste Anzahl von Iterationen
n_embd = 384 # Größe der Einbettungsdimensionen im Modell
n_head = 6 # Anzahl der Heads für Klasse "Block"
n_layer = 6 # Anzahl der Layer des Transformers
dropout = 0.2 # Technik um während Training overfitting zu reduzieren

torch.manual_seed(1337) # wird festgelegt, damit torch.randint später nicht immer zufällige verschiedene Werte generiert

# txt Datei Shakespear öffnen
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Extrahierung aller vorkommenden Zeichen/Buchstaben/Zahlen + Menge dessen aus der txt Datei
chars = sorted(list(set(text)))
vocab_size = len(chars)


## - Character Tokenizer - ##

# Zuordnung von Buchstaben (ch, token) zu Zahlen (i, embedding) und durch encode/decode Übersetzer (ch<>i) erschaffen
stoi = { ch:i for i, ch in enumerate(chars) } # stoi: String to Index Wörterbuch (als Dictionary)
itos = { i:ch for i, ch in enumerate(chars) } # itos: Index to String Wörterbuch (als Dictionary)
encode = lambda s: [stoi[c] for c in s] # encoder: übergibt str (s) an stoi, kriegt int (c) zurück
decode = lambda l: "".join([itos[i] for i in l]) # decoder: übergibt int (l) an itos, kriegt str (i) zurück

# Encoding der gesamten txt Datei & speichern dessen in einem torch.Tensor (jeder einzelne Buchstabe wird tokenisiert)
data = torch.tensor(encode(text), dtype=torch.long)


## - Vorbereitung für Training des Transformers - ##

# Aufteilung des Datensatzes in train & validation set
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Beispiel Visualisierung, wie in einem Chunk Verbindungen zwischen Tokens und Folgetokens erstellt werden (Wortnetz)
def example_visualization():
    x = train_data[:block_size]
    y = train_data[1:block_size + 1]

    for t in range(10):
        context = x[:t+1]
        target = y[t]
        print(f"when input is {context} the target: {target}")

#example_visualization()

# Erstellung Batch
def get_batch(split):
    data = train_data if split == "train" else val_data # data = train_data wenn split = "train", sonst val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Generierung der 4 zufälligen Startindizes für 4 chunks
    x = torch.stack( [data[i:i+block_size] for i in ix]) # Tensor x mit Eingabedaten für Batch wird erstellt
    y = torch.stack( [data[i+1:i+block_size+1] for i in ix]) # Tensor y mit Zieldaten (x + 1) für Batch wird erstellt
    x, y = x.to(device), y.to(device) # Übergabe von x und y auf entsprechende Hardware (GPT/CPU)
    return x, y # x (Eingabedaten) und y (Zieldaten, auch Label genannt) werden durch Funktion als Batch zurückgegeben

# estimate_loss: Funktion, die den Mittelwert der Verluste aus n Trainingsiterationen berechnet
@torch.no_grad() # @: Decorator, der verhindert, dass Gradienten berechnet werden, nützlich für die Auswertung
def estimate_loss():
    out = {} # Leeres Dictionary, um die Verluste zu speichern
    model.eval() # setzt das Modell in den Evaluierungsmodus, zB. werden dadurch Dropout Schichten deaktiviert

    for split in ['train', 'val']: # Schleife, um über Trainings- und Validierungsdaten zu iterieren
        losses = torch.zeros(eval_iters) # Erstellt einen Tensor als Platzhalter für die Verluste in jeder Iteration
        for k in range(eval_iters): # Schleife, um eval_iters-mal über die Daten zu iterieren und Losses zu berechnen
            X, Y = get_batch(split) # in jeder Iteration wird zufälliger Batch (X, Y) abgerufen
            logits, loss = model(X, Y) # Berechnung loss anhand Gegenüberstellung logits/Y anhand Eingabe X
            losses[k] = loss.item() # berechneter loss wird der Liste losses hinzugefügt
        out[split] = losses.mean() # nach den Iterationen wird der Mittelwert aller Losses für Datensatz X berechnet
    model.train() # Nach Abschluss Evaluierung wird model von Evaluierungs- zurück in den Trainingsmodus versetzt
    return out # Mittelwert der losses werden als dictionary out zurückgegeben


## Self-Attention-Mechanism ##

# Einzelner Self-Attention Head
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size): # head_size: Größe der Ausgabe eines einzelnen Attention Heads
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # Eingabe (x) in Schlüssel (key) umgewandelt / projiziert
        self.query = nn.Linear(n_embd, head_size, bias=False) # Eingabe (x) in Abfragen (query) umgewandelt / projiziert
        self.value = nn.Linear(n_embd, head_size, bias=False) # Eingabe (x) in Werte (value) umgewandelt / projiziert
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # Erstellung Dreiecksmatrix aus 1en

        self.dropout = nn.Dropout(dropout) # Drop-Out Schicht um Overfitting zu reduzieren

    # Tatsächlicher Self-Attention Mechanism
    def forward(self, x):
        B,T,C = x.shape # Abfrage Dimensionen der Eingabe x
        k = self.key(x)   # Durch x wird key erzeugt
        q = self.query(x) # Durch x wird query erzeugt

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # Berechnung Attention-Matrix
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Einbindung Dreiecksmatrix in Attention-Matrix
        wei = F.softmax(wei, dim=-1) # Normalisierung der Attention-Matrix durch softmax
        wei = self.dropout(wei) # Dropout zur Reduzierung Overfitting

        v = self.value(x) # Durch x wird value erzeugt
        out = wei @ v # Multiplikation der Attention-Matrix und values
        return out # Output Durchschnitt der Wertrepräsentationen, welcher Ausgabe der Attention-Einheit darstellt

# Implementierung mehrerer Self-Attention Heads zur parallelen Ausführung
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size): # num_heads: parallele Anzahl Heads / head_size: Größe Heads
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # Erstellung Liste mit Heads
        self.proj = nn.Linear(head_size * num_heads, n_embd) # Rückumwandlung Ausgaben der Heads in ursprüngliche Dimen.
        self.dropout = nn.Dropout(dropout) # Drop-Out zur Minderung Overfitting

    # Tatsächlicher Multi-Head-Attention Mechanism
    def forward(self, x): # x wird durch jeden Head verarbeitet und gibt output raus
        out = torch.cat([h(x) for h in self.heads], dim=-1) # torch.cat: alle zusammengeführte outputs werden verbunden
        out = self.dropout(self.proj(out)) # Drop-Out zur Minderung Overfitting
        return out # Output in Form eines Tensors, welche kontextualisierte Darstellung wieder gibt

# Multi-Layer Perzeptron zur durchführung einer nichtlinearen Transformation
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(               # Initialisierung des Multi-Layer Perzeptron (neuronales Netz)
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x): # Verarbeitung der Eingabe x durch das neuronale Netz
        return self.net(x) # Output des nn wird zurück gegeben

# Zusammenfügung der Komponenten MultiHeadAttention und FeedFoward in einem Transformerblock
class Block(nn.Module):
    """" Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head # Definition des head_size
        self.sa = MultiHeadAttention(n_head, head_size) # Initialisierung Aufmerksamkeitskomponente (MultiHeadAttention)
        self.ffwd = FeedFoward(n_embd) # Feed-Forward
        self.ln1 = nn.LayerNorm(n_embd) # Instanz zur Layer Normalisierung
        self.ln2 = nn.LayerNorm(n_embd) # Instanz zur Layer Normalisierung

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Aufmerksamkeitskomponente wird auf x angewendet und mit x addiert + normalisiert
        x = x + self.ffwd(self.ln2(x)) # FeedForward wird auf x angewendet und mit x addiert + normalisiert
        return x # verarbeitetes x wird wieder ausgegeben


## GPT (decoder-only und ohne cross-attention) ##

class GPTLanguageModel(nn.Module):

    # nn.Embedding: Erstellung Embeddingmatrix vocab_size*vocab_size
    def __init__(self):
        super().__init__() # super(): Aufruf der init Methode aus der parent class nn.Module
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # vocab_size: Menge einzigartige Zeichen in txt
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # Position jedes einzelnen Tokens
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # Initial. Attention-Blocks
        self.ln_f = nn.LayerNorm(n_embd) # Normalisierungsschicht
        self.lm_head = nn.Linear(n_embd, vocab_size) # Transformationsschicht

        self.apply(self._init_weights) # Erste Initialisierung Gewichte

    # Initalisierungsmethode für Gewichte der Schichten der Transformation (nn.Linear) und Einbettung (nn.Embedding)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear): # Prüfung, ob Transformationsschicht vorliegt
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # Initalisierung Gewichte von nn.Linear
            if module.bias is not None: # überprüft, ob module ein bias (Verschiebung) hat
                torch.nn.init.zeros_(module.bias) # Initialisiert die Verschiebung (Bias) des Moduls mit Nullen.
        elif isinstance(module, nn.Embedding): # Prüfung, ob Einbettungsschicht vorliegt
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # # Initalisierung Gewichte von nn.Embedding

    # logits: Input wird übergeben & verweist auf Zeilen aus Emb.matrix, aus diesen Zeilen wird logits-Matrix erstellt
    def forward(self, idx, targets=None): # targets = None: so bleibt targets optional
        B, T = idx.shape # Abfrage Dimensionen der Eingabe idx

        # idx und targets sind beide (B,T) tensors aus integers
        tok_emb = self.token_embedding_table(idx) # einzigartige Tokens werden in Vektor umgewandelt
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # Token Position in einem Vektor
        x = tok_emb + pos_emb # Addition der Vektoren tok_emb und pos_emb
        x = self.blocks(x) # weitere Verarbeitung der eingaben
        x = self.ln_f(x) # Schicht um eingaben zu Normalisieren
        logits = self.lm_head(x) # (B,T,vocab_size)

        # loss soll nur berechnet werden, wenn auch targets übergeben werden
        if targets is None:
            loss = None
        else:
            # reshapen der logits- und targets-Matrizen, da diese sonst nicht mit Loss Function fitten
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # Loss Function, welche Vorhersage (logits)/targets gegenüberstellt

        return logits, loss # für Eingabetext relevante Zeilen aus Embeddingmatrix & loss werden zurückgegeben

    # Funktion des Modells: Generierung weiterer strings basierend auf input (idx)
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens): # max_new_tokens: max. Anzahl an zu generierenden strings
            idx_cond = idx[:, -block_size:] # trimmen von idx zu den letzten block_size tokens
            logits, loss = self(idx_cond) # erhalten der predictions (logits) & loss zum gegebenen Input
            logits = logits[:, -1, :] # Fokus wird auf letzten Token gesetzt da auf diesen nächster Token generiert wird
            probs = F.softmax(logits, dim=-1) # softmax: erzeugt durch Vorhersage die Wahrscheinlichkeit für next-Token
            idx_next = torch.multinomial(probs, num_samples=1) # multinomal: wählt basierend auf Wahrsch. next-Token aus
            idx = torch.cat((idx, idx_next), dim=1) # cat: Input (idx) wird um weiteren String (idx_next) erweitert
        return idx # Input + darauf basierend generierte strings werden zurück gegeben



# Erstellung Objekt model
model = GPTLanguageModel()
m = model.to(device) # Model wird über entsprechende Hardware trainiert

# print Anzahl an Parameter des Modells
#print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# print der vom Modell decodeten und generierten strings basierend auf input
#print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


## Model-Training ##

# Erstellung eines PyTorch Optimizers (Art: AdamW)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Trainingsschleife
for iter in range(max_iters):

    # Evaluierung des Modells soll alle eval_intervall und bei der letzten Iteration durchgeführt werden
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss() # estimate_loss: Funktion welche Loss-Mittelwert aus n Trainingsiterationen berechnet
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}") # print Loss Mittelwerte

    xb, yb = get_batch('train') # Generierung eines neuen zufälligen batches (bestehend aus xb/yb) aus den trainingsdata

    logits, loss = m(xb, yb)  # Generierung Vorhersage logits, gegenüberstellung dieser mit targets yb & Berechnung loss
    optimizer.zero_grad(set_to_none=True)  # Gradienten = 0, sodass im next-step Gradienten neu berechnet werden können
    loss.backward()  # Berechnung der Gradienten, sprich die Änderungen des Loss-Werts bei Änderung der Parameter
    optimizer.step()  # Optimierung der Parameter basierend auf dem berechneten Gradienten

# print der vom Modell decodeten und generierten strings basierend auf input, diesmal nach dem Training
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Erstellung eines Basis-Tensors bestehend aus Nullen
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))