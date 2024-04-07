import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # Wie viele data_chunks werden parallel verarbeitet?
block_size = 256 # Wie viele vorhergehende Token werden für die Vorhersage des nächsten Tokens verwendet?
max_iters = 5000 # max. Anzahl an Trainings-Iterationen
eval_interval = 500 # Intervall (zu max_iters), zu dem das Modell evaluiert wird
learning_rate = 3e-4 # bestimmt die Größe der Schritte, die der optimizer während des Trainings unternimmt
device = 'cuda' if torch.cuda.is_available() else 'cpu' # soll mit GPU ('cuda') oder CPU ('cpu') trainiert werden?
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


## Self-Attention Blocks ##

class Head(nn.Module): # ?
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module): # ?
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module): # ?
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module): # ?
    """" Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


## Neural Network: Bigram Language Model (GPT) ##

class GPTLanguageModel(nn.Module):

    # nn.Embedding: Erstellung Embeddingmatrix vocab_size*vocab_size
    def __init__(self):
        super().__init__() # super(): Aufruf der init Methode aus der parent class nn.Module
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # vocab_size: Menge einzigartige Zeichen in txt
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # ?
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # ?
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm # ?
        self.lm_head = nn.Linear(n_embd, vocab_size) # ?

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module): # ?
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # logits: Input wird übergeben & verweist auf Zeilen aus Emb.matrix, aus diesen Zeilen wird logits-Matrix erstellt
    def forward(self, idx, targets=None): # targets = None: so bleibt targets optional
        B, T = idx.shape # ?

        # ?
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
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
            idx_cond = idx[:, -block_size:] # ?
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