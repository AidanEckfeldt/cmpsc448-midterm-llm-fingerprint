import re, random, numpy as np, pandas as pd, torch, torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

SEED=448
LABELS=["Claude","GPT","Gemini"]
label_to_id={x:i for i,x in enumerate(LABELS)}
PAD_ID,UNK_ID,SEP_ID=0,1,2
TOKEN_RE=re.compile(r"\w+|[^\w\s]", re.UNICODE)
MAX_LEN=256
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize(text): return TOKEN_RE.findall(str(text).lower())
def make_text(df,mode):
    if mode=="input": return df.llm_input.astype(str)
    if mode=="output": return df.llm_output.astype(str)
    return df.llm_input.astype(str)+" <SEP> "+df.llm_output.astype(str)

def build_vocab(texts,max_vocab=20000,min_freq=2):
    c=Counter()
    for t in texts: c.update(tokenize(t))
    v={"<PAD>":0,"<UNK>":1,"<SEP>":2}
    for tok,n in c.most_common():
        if n<min_freq or len(v)>=max_vocab: break
        if tok not in v: v[tok]=len(v)
    return v

class TextDataset(Dataset):
    def __init__(self,df,texts,vocab,max_len=MAX_LEN):
        self.df=df.reset_index(drop=True); self.texts=list(texts); self.vocab=vocab; self.max_len=max_len
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        ids=[self.vocab.get(t,UNK_ID) for t in tokenize(self.texts[i])[:self.max_len]] or [UNK_ID]
        length=len(ids); ids += [PAD_ID]*(self.max_len-length)
        return torch.tensor(ids), torch.tensor(length), torch.tensor(label_to_id[self.df.loc[i,"llm_family"]])

class TextCNN(nn.Module):
    def __init__(self,vocab_size,emb_dim=128,filters=128,kernels=(3,4,5),dropout=.3):
        super().__init__(); self.embedding=nn.Embedding(vocab_size,emb_dim,padding_idx=PAD_ID)
        self.convs=nn.ModuleList([nn.Conv1d(emb_dim,filters,k) for k in kernels])
        self.dropout=nn.Dropout(dropout); self.fc=nn.Linear(filters*len(kernels),3)
    def forward(self,x,lengths=None):
        e=self.embedding(x).transpose(1,2)
        z=torch.cat([torch.relu(c(e)).max(2).values for c in self.convs],1)
        return self.fc(self.dropout(z))

class TextLSTM(nn.Module):
    def __init__(self,vocab_size,emb_dim=128,hidden=128,dropout=.3):
        super().__init__(); self.embedding=nn.Embedding(vocab_size,emb_dim,padding_idx=PAD_ID)
        self.lstm=nn.LSTM(emb_dim,hidden,batch_first=True,bidirectional=True)
        self.dropout=nn.Dropout(dropout); self.fc=nn.Linear(hidden*2,3)
    def forward(self,x,lengths):
        p=nn.utils.rnn.pack_padded_sequence(self.embedding(x),lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,(h,_)=self.lstm(p); return self.fc(self.dropout(torch.cat([h[-2],h[-1]],1)))

def run_epoch(model,loader,opt=None):
    model.train(opt is not None); crit=nn.CrossEntropyLoss(); ys=[]; ps=[]; losses=[]
    for x,l,y in loader:
        x,l,y=x.to(DEVICE),l.to(DEVICE),y.to(DEVICE)
        if opt: opt.zero_grad()
        logits=model(x,l); loss=crit(logits,y)
        if opt: loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
        losses.append(loss.item()*len(y)); ys+=y.cpu().tolist(); ps+=logits.argmax(1).detach().cpu().tolist()
    return sum(losses)/len(ys), accuracy_score(ys,ps)

def train(arch,vocab_size,tr,va,epochs=8):
    model=(TextCNN(vocab_size) if arch=="CNN" else TextLSTM(vocab_size)).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=.001); best=None; best_acc=-1; wait=0
    for _ in range(epochs):
        run_epoch(model,tr,opt); _,acc=run_epoch(model,va)
        if acc>best_acc:
            best_acc=acc; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=2: break
    model.load_state_dict(best); return model.to(DEVICE)

def evaluate(model,loader):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,l,y in loader:
            logits=model(x.to(DEVICE),l.to(DEVICE)); ys+=y.tolist(); ps+=logits.argmax(1).cpu().tolist()
    p,r,f1,_=precision_recall_fscore_support(ys,ps,average="macro",zero_division=0)
    return accuracy_score(ys,ps),p,r,f1

def main(csv="cmpsc448_llm_fingerprint_dataset.csv"):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    df=pd.read_csv(csv).dropna()
    g1=GroupShuffleSplit(n_splits=1,test_size=.30,random_state=SEED)
    ti,xi=next(g1.split(df,groups=df.question_id)); train_df=df.iloc[ti].reset_index(drop=True); temp=df.iloc[xi].reset_index(drop=True)
    g2=GroupShuffleSplit(n_splits=1,test_size=.50,random_state=SEED)
    vi,si=next(g2.split(temp,groups=temp.question_id)); val_df=temp.iloc[vi].reset_index(drop=True); test_df=temp.iloc[si].reset_index(drop=True)
    out=[]
    for mode in ["input","output","both"]:
        vocab=build_vocab(make_text(train_df,mode))
        loaders=[]
        for frame in [train_df,val_df,test_df]:
            loaders.append(DataLoader(TextDataset(frame,make_text(frame,mode),vocab),batch_size=64,shuffle=(frame is train_df)))
        for arch in ["CNN","LSTM"]:
            m=train(arch,len(vocab),loaders[0],loaders[1]); acc,p,r,f1=evaluate(m,loaders[2])
            out.append([arch,mode,acc,p,r,f1])
    pd.DataFrame(out,columns=["architecture","text_mode","accuracy","precision_macro","recall_macro","f1_macro"]).to_csv("rq1_rq2_results.csv",index=False)

if __name__=="__main__":
    main()
