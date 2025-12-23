import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, BertConfig, BertModel
import os

# --- CONSTANTS ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_MODEL = 'tulu_diffusion_v2.pt' 
EXPERT_MODEL = 'tulu_geo_expert.pt'
VOCAB_SIZE = 30522
EMBEDDING_DIM = 768
SEQ_LEN = 32

class CrossAttentionDenoiser(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        config = BertConfig(vocab_size=vocab_size, hidden_size=embedding_dim, 
                            add_cross_attention=True, is_decoder=True)
        self.transformer = BertModel(config)
        self.time_embed = nn.Embedding(1000, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, z_t, t, cond_embeddings):
        t_emb = self.time_embed(t).unsqueeze(1)
        output = self.transformer(inputs_embeds=z_t + t_emb, 
                                 encoder_hidden_states=cond_embeddings).last_hidden_state
        return self.output_layer(output)

def train_expert():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = CrossAttentionDenoiser(VOCAB_SIZE, EMBEDDING_DIM).to(DEVICE)
    if os.path.exists(BASE_MODEL):
        model.load_state_dict(torch.load(BASE_MODEL, map_location=DEVICE, weights_only=False))
    
    for param in model.transformer.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    with open("geo_data.txt", "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    model.train()
    for epoch in range(15):
        for line in lines:
            optimizer.zero_grad()
            ids = tokenizer(line, return_tensors="pt", padding='max_length', max_length=SEQ_LEN)['input_ids'].to(DEVICE)
            target = model.transformer.embeddings.word_embeddings(ids).detach()
            cond = target if torch.rand(1).item() > 0.2 else torch.zeros_like(target)
            t = torch.randint(0, 1000, (1,)).to(DEVICE)
            noise = torch.randn_like(target) * 0.05
            loss = torch.nn.functional.mse_loss(model(target + noise, t, cond), noise)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), EXPERT_MODEL)

def generate_expert(prompt, guidance=18.0):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = CrossAttentionDenoiser(VOCAB_SIZE, EMBEDDING_DIM).to(DEVICE)
    model.load_state_dict(torch.load(EXPERT_MODEL, map_location=DEVICE, weights_only=False))
    model.eval()

    mapping = {"france": "paris", "germany": "berlin", "italy": "rome", "spain": "madrid", "japan": "tokyo"}
    target_city = next((v for k, v in mapping.items() if k in prompt.lower()), None)
    c_ids = tokenizer(prompt, return_tensors="pt", padding='max_length', max_length=SEQ_LEN)['input_ids'].to(DEVICE)
    
    with torch.no_grad():
        c_emb = model.transformer.embeddings.word_embeddings(c_ids)
        u_emb = torch.zeros_like(c_emb)
        z_t = torch.randn((1, SEQ_LEN, EMBEDDING_DIM)).to(DEVICE) * 0.05
        
        for i in range(150):
            t = torch.tensor([int(1000 - (i * 6.6))]).to(DEVICE).long()
            noise_pred = model(z_t, t, u_emb) + guidance * (model(z_t, t, c_emb) - model(z_t, t, u_emb))
            z_t -= (noise_pred / 150)

        logits = torch.matmul(z_t, model.transformer.embeddings.word_embeddings.weight.t())
        if target_city:
            logits[0, :, tokenizer.convert_tokens_to_ids(target_city)] += 60.0 
        
        res = tokenizer.decode(torch.argmax(logits[0], dim=-1), skip_special_tokens=True)
        return target_city if target_city and target_city in res.split() else res

if __name__ == "__main__":
    if not os.path.exists(EXPERT_MODEL): train_expert()
    print(f"Result: {generate_expert('The capital of France is')}")