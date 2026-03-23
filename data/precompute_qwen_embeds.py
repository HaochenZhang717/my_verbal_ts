import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# =========================
# Qwen encoder（无 projector）
# =========================
class QwenTextEncoder(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B",
            padding_side="left"
        )
        self.model = AutoModel.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B"
        ).to(device).eval()

        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, texts):
        batch = self.tokenizer(
            texts,
            padding="max_length",
            max_length=40,
            return_tensors="pt"
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        outputs = self.model(**batch).last_hidden_state  # (B, L, 1024)
        return outputs


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

    encoder = QwenTextEncoder(device).to(device)

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
            embeds = encoder(batch_text)  # (b, L, 1024)
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