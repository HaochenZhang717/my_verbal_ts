import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPTextConfig

# =========================
# Qwen encoder（无 projector）
# =========================
class ClipTextEncoder(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        clip_config = CLIPTextConfig.from_pretrained("/playpen-shared/haochenz/long_clip")
        clip_config.max_position_embeddings = 248
        self.model = CLIPTextModelWithProjection.from_pretrained(
            "/playpen-shared/haochenz/long_clip",
            config=clip_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained("/playpen-shared/haochenz/long_clip")

        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, text):
        inputs = self.tokenizer(
            text, padding=True, truncation=True,
            max_length=self.max_length,
            return_tensors="pt")["input_ids"]
        inputs = inputs.to(self.device)
        text_emb = self.model(input_ids=inputs).last_hidden_state

        return text_emb


# =========================
# 主逻辑
# =========================
def precompute(
    caps_path,
    save_path,
    split="train",
    batch_size=64,
    device="cuda"
):

    print("Loading captions...")

    caps_dict = {}
    with open(f"{caps_path}/{split}_caps_ready.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            caps_dict[item["id"]] = item["captions"]

    print(f"Loaded {len(caps_dict)} samples")

    encoder = ClipTextEncoder(device).to(device)

    result = {}

    # =========================
    # 遍历每个 sample
    # =========================
    for image_id in tqdm(caps_dict.keys()):

        caps_list = caps_dict[image_id]  # list of dicts

        # flatten这个sample内部
        keys = []
        texts = []

        for d in caps_list:
            for k, v in d.items():
                keys.append(k)     # segX_channelY
                texts.append(v)

        # =========================
        # batch encode
        # =========================
        embeds_all = []

        for i in range(0, len(texts), batch_size):
            batch_text = texts[i:i + batch_size]
            breakpoint()
            embeds = encoder(batch_text)  # (b, L, 1024)
            print(embeds.shape)
            breakpoint()
            embeds_all.append(embeds.cpu())

        embeds_all = torch.cat(embeds_all, dim=0)

        # =========================
        # 存回 dict
        # =========================
        result[image_id] = {}

        for i, k in enumerate(keys):
            result[image_id][k] = embeds_all[i]  # (L, D)

    # =========================
    # 保存
    # =========================
    torch.save(result, save_path)

    print(f"Saved to {save_path}")

    # debug
    example = list(result.keys())[0]
    print("Example:", example)
    print("Keys:", result[example].keys())
    print("Shape:", list(result[example].values())[0].shape)


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--caps_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    precompute(
        caps_path=args.caps_path,
        save_path=args.save_path,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )