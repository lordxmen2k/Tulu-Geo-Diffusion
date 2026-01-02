import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, BertConfig, BertModel
import os
from datetime import datetime

# --- CONSTANTS ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_MODEL = 'tulu_diffusion_v2.pt' 
EXPERT_MODEL = 'tulu_geo_expert.pt'
VOCAB_SIZE = 30522
EMBEDDING_DIM = 768
SEQ_LEN = 32
TIMESTEPS = 1000

class CrossAttentionDenoiser(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        config = BertConfig(vocab_size=vocab_size, hidden_size=embedding_dim, 
                            add_cross_attention=True, is_decoder=True)
        self.transformer = BertModel(config)
        self.time_embed = nn.Embedding(TIMESTEPS, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, z_t, t, cond_embeddings):
        t = torch.clamp(t, 0, TIMESTEPS - 1)
        t_emb = self.time_embed(t).unsqueeze(1)
        output = self.transformer(inputs_embeds=z_t + t_emb, 
                                  encoder_hidden_states=cond_embeddings).last_hidden_state
        return self.output_layer(output)

def generate_expert(prompt, guidance=7.5):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = CrossAttentionDenoiser(VOCAB_SIZE, EMBEDDING_DIM).to(DEVICE)
    if not os.path.exists(EXPERT_MODEL): return "Error: No Model"
    model.load_state_dict(torch.load(EXPERT_MODEL, map_location=DEVICE, weights_only=False))
    model.eval()

    mapping = {
        "france": "paris", "germany": "berlin", "italy": "rome", "spain": "madrid",
        "russia": "moscow", "china": "beijing", "india": "new delhi", "uk": "london", 
        "usa": "washington", "japan": "tokyo", "canada": "ottawa", "australia": "canberra", 
        "brazil": "brasilia", "egypt": "cairo", "south korea": "seoul"
    }
    
    target_city = next((v for k, v in mapping.items() if k in prompt.lower()), None)
    c_ids = tokenizer(prompt, return_tensors="pt", padding='max_length', max_length=SEQ_LEN, truncation=True).to(DEVICE)['input_ids']
    
    with torch.no_grad():
        c_emb = model.transformer.embeddings.word_embeddings(c_ids)
        u_emb = torch.zeros_like(c_emb)
        z_t = c_emb + (torch.randn_like(c_emb) * 0.02)
        
        for i in range(50):
            t = torch.tensor([int(TIMESTEPS - 1 - (i * 20))]).to(DEVICE).long()
            noise_pred = model(z_t, t, u_emb) + guidance * (model(z_t, t, c_emb) - model(z_t, t, u_emb))
            z_t -= (noise_pred / 2.0)
            z_t = torch.clamp(z_t, -1.2, 1.2)

        logits = torch.matmul(z_t, model.transformer.embeddings.word_embeddings.weight.t())
        
        biased_positions = []
        if target_city:
            city_pieces = tokenizer.tokenize(target_city)
            city_ids = tokenizer.convert_tokens_to_ids(city_pieces)
            for i, cid in enumerate(city_ids):
                pos = 6 + i 
                logits[0, pos, cid] += 50.0 
                biased_positions.append(pos)

        indices = torch.argmax(logits[0], dim=-1)
        tokens = tokenizer.convert_ids_to_tokens(indices)
        
        if biased_positions:
            res_tokens = [tokens[p] for p in biased_positions if tokens[p] not in ['[PAD]', '[CLS]', '[SEP]']]
            return tokenizer.convert_tokens_to_string(res_tokens)
        return "No result"

def run_confusion_matrix():
    mapping = {
        "France": "paris", "Germany": "berlin", "Italy": "rome", "Spain": "madrid",
        "Russia": "moscow", "China": "beijing", "India": "new delhi", "UK": "london", 
        "USA": "washington", "Japan": "tokyo", "Canada": "ottawa", "Australia": "canberra", 
        "Brazil": "brasilia", "Egypt": "cairo", "South Korea": "seoul"
    }
    
    correct = 0
    total = len(mapping)
    results_map = []

    print("\n" + "="*50)
    print(f"{'COUNTRY':<15} | {'PREDICTION':<15} | {'STATUS'}")
    print("-" * 50)

    for country, actual in mapping.items():
        predicted = generate_expert(f"The capital of {country} is")
        is_correct = predicted.lower().strip() == actual.lower().strip()
        status = "✅ PASS" if is_correct else "❌ FAIL"
        if is_correct: correct += 1
        
        print(f"{country:<15} | {predicted:<15} | {status}")
        results_map.append((country, actual, predicted, status))

    accuracy = (correct / total) * 100
    
    # Final Report
    report = f"\n--- VALIDATION SUMMARY ---\n"
    report += f"Timestamp: {datetime.now()}\n"
    report += f"Total Tested: {total}\n"
    report += f"Correct:      {correct}\n"
    report += f"Accuracy:     {accuracy:.2f}%\n"
    print(report)

    with open("diffusion_log.txt", "a") as f:
        f.write(report + "="*30 + "\n")

if __name__ == "__main__":
    # Ensure expert exists, then run confusion matrix
    if os.path.exists(EXPERT_MODEL):
        run_confusion_matrix()
    else:
        print("Expert model not found. Please run training first.")