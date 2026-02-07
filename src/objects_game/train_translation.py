import sacrebleu
import torch
import torch.nn as nn
import torch.optim as optim
from src.objects_game.src.translation import Decoder, Encoder
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets import Dataset, load_dataset, load_from_disk

tgt_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tgt_pad_id = tgt_tokenizer.pad_token_id


PAD_ID = 70

def collate(batch):
    src = [torch.tensor(b['message_truncated'], dtype=torch.long) for b in batch]
    src = pad_sequence(
        src,
        batch_first=True,
        padding_value=PAD_ID
    )

    tgt = tgt_tokenizer(
        [b['captions'][0] for b in batch],
        padding=True,
        return_tensors="pt"
    )["input_ids"]

    return src, tgt



def evaluate(encoder, decoder, loader, criterion, device):
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    refs, hyps = [], []

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)

            h, c = encoder(src)
            logits, _, _ = decoder(tgt[:, :-1], h, c)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt[:, 1:].reshape(-1)
            )
            total_loss += loss.item()

            pred = logits.argmax(-1)
            hyps.extend(pred.tolist())
            refs.extend(tgt[:, 1:].tolist())

    bleu = sacrebleu.corpus_bleu(
        [" ".join(map(str, h)) for h in hyps],
        [[" ".join(map(str, r)) for r in refs]]
    ).score

    return total_loss / len(loader), bleu

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(
        vocab_size=70+1,  # +1 for PAD
        emb_dim=256,
        hid_dim=512,
        pad_id=PAD_ID
    ).to(device)

    decoder = Decoder(
        vocab_size=len(tgt_tokenizer.vocab),
        emb_dim=256,
        hid_dim=512,
        pad_id=tgt_pad_id
    ).to(device)

    dataset = load_from_disk("../datasets/coco_train_msg_captions")
    val_dataset = load_from_disk("../datasets/coco_val_msg_captions")

    batch_size = 512
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate
    )


    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_id)
    enc_opt = torch.optim.Adam(encoder.parameters(), lr=1e-2)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=1e-2)

    encoder.train()
    decoder.train()

    num_epochs = 10

    import wandb

    config = {
        "device": str(device),
        "pad_id": PAD_ID,
        "tgt_pad_id": tgt_pad_id,
        "enc_vocab_size": encoder.emb.num_embeddings,
        "dec_vocab_size": decoder.emb.num_embeddings,
        "emb_dim": encoder.emb.embedding_dim,
        "hid_dim": encoder.rnn.hidden_size,
        "enc_num_layers": encoder.rnn.num_layers,
        "dec_num_layers": decoder.rnn.num_layers,
        "encoder_pad_idx": encoder.emb.padding_idx,
        "decoder_pad_idx": decoder.emb.padding_idx,
        "batch_size": getattr(loader, "batch_size", None),
        "lr_enc": enc_opt.param_groups[0]["lr"],
        "lr_dec": dec_opt.param_groups[0]["lr"],
        "optim_enc": type(enc_opt).__name__,
        "optim_dec": type(dec_opt).__name__,
        "optim_enc_betas": enc_opt.param_groups[0]["betas"],
        "optim_dec_betas": dec_opt.param_groups[0]["betas"],
        "criterion": type(criterion).__name__,
        "dataset_len": len(dataset),
        "dataset_features": list(dataset.features.keys()),
        "num_epochs": num_epochs,
        # "sos_id": sos_id,
        # "eos_id": eos_id,
        "tgt_tokenizer": getattr(tgt_tokenizer, "name_or_path", str(tgt_tokenizer)),
    }

    # ensure the subsequent wandb.init call will merge this config into the run
    _orig_wandb_init = wandb.init
    def _wandb_init_with_config(*args, **kwargs):
        run = _orig_wandb_init(*args, **kwargs)
        try:
            wandb.config.update(config)
        except Exception:
            pass
        return run
    wandb.init = _wandb_init_with_config

    print("Prepared wandb config:", config)
    # wandb.init(project=project, id=run_id, name=run_name, **kwargs)
    wandb.init(
        project='EmComm-Caption-Translator',
        name="find_max_batch",
        config={
            "emb_dim": 256,
            "hid_dim": 512,
            # "batch_size": 32,
            "lr": 3e-3,
            "num_epochs": num_epochs,
        }
    )

    patience = 10
    best_val_loss = float("inf")
    patience_ctr = 0

    sched_enc = ReduceLROnPlateau(enc_opt, mode="min", factor=0.9, patience=20)
    sched_dec = ReduceLROnPlateau(dec_opt, mode="min", factor=0.9, patience=20)



    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()

        total_loss = 0.0

        for src, tgt in tqdm(loader, desc=f"epoch {epoch}"):
            src, tgt = src.to(device), tgt.to(device)

            enc_opt.zero_grad()
            dec_opt.zero_grad()

            h, c = encoder(src)
            logits, _, _ = decoder(tgt[:, :-1], h, c)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt[:, 1:].reshape(-1)
            )

            loss.backward()
            enc_opt.step()
            dec_opt.step()
            # sched_enc.step(lo
            # ss)


            total_loss += loss.item()
            wandb.log(
                {"train/batch_loss": loss.item(),
                "lr/encoder": enc_opt.param_groups[0]["lr"],
                "lr/decoder": dec_opt.param_groups[0]["lr"],
            })


        train_loss = total_loss / len(loader)
        # val_loss = evaluate(encoder, decoder, val_loader, criterion, device)
        val_loss, val_bleu = evaluate(
        encoder, decoder, val_loader, criterion, device
        )
        sched_enc.step(val_loss)
        sched_dec.step(val_loss)

        # wandb.log({
        #     "epoch": epoch,
        #     "train/loss": train_loss,
        #     "val/loss": val_loss,
        # })
        wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "val/loss": val_loss,
        "val/bleu": val_bleu,
        })

        print(
            f"epoch {epoch}: "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        # -------- early stopping --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0

            # optional: save best model
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
            }, "best_model.pt")

            wandb.log({"early_stop/best_val_loss": best_val_loss})
        else:
            patience_ctr += 1
            wandb.log({"early_stop/patience": patience_ctr})

            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch}")
                break


if __name__ == "__main__":
    main()