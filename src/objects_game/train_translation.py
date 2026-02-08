import argparse
import sacrebleu
import torch
import torch.nn as nn
from src.objects_game.src.translation import Decoder, Encoder
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets import load_from_disk


def get_tokenizer_and_pad():
    """Load tokenizer and get pad ids."""
    tgt_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tgt_pad_id = tgt_tokenizer.pad_token_id
    return tgt_tokenizer, tgt_pad_id


def collate(batch, pad_id, tokenizer):
    """Collate function that accepts pad_id and tokenizer as parameters."""
    src = [torch.tensor(b['message_truncated'], dtype=torch.long) for b in batch]
    src = pad_sequence(
        src,
        batch_first=True,
        padding_value=pad_id
    )
    tgt = tokenizer(
        [b['captions'][0] for b in batch],
        padding=True,
        return_tensors="pt"
    )["input_ids"]

    return src, tgt

def evaluate(encoder, decoder, loader, criterion, device, tokenizer):
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

            hyps.extend(
                tokenizer.batch_decode(
                    pred,
                    skip_special_tokens=True
                )
            )
            refs.extend(
                tokenizer.batch_decode(
                    tgt[:, 1:],
                    skip_special_tokens=True
                )
            )

    bleu = sacrebleu.corpus_bleu(
        hyps,
        [refs]
    ).score

    return total_loss / len(loader), bleu

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tgt_tokenizer, tgt_pad_id = get_tokenizer_and_pad()
    
    # Initialize models
    encoder = Encoder(
        vocab_size=args.src_vocab_size,
        emb_dim=args.emb_dim,
        hid_dim=args.hid_dim,
        num_layers=args.enc_num_layers,
        pad_id=args.pad_id
    ).to(device)

    decoder = Decoder(
        vocab_size=len(tgt_tokenizer.vocab),
        emb_dim=args.emb_dim,
        hid_dim=args.hid_dim,
        num_layers=args.dec_num_layers,
        pad_id=tgt_pad_id
    ).to(device)

    # Load datasets
    dataset = load_from_disk(args.train_dataset_path)
    val_test_dataset = load_from_disk(args.val_dataset_path)
    splits = val_test_dataset.train_test_split(test_size=0.5, seed=42)
    val_dataset = splits["train"]
    test_dataset = splits["test"]

    # Create dataloaders
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, args.pad_id, tgt_tokenizer)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, args.pad_id, tgt_tokenizer)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, args.pad_id, tgt_tokenizer)
    )

    # Setup loss and optimizers
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_id)
    enc_opt = torch.optim.Adam(encoder.parameters(), lr=args.lr_enc)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=args.lr_dec)

    # Setup schedulers
    sched_enc = ReduceLROnPlateau(enc_opt, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience)
    sched_dec = ReduceLROnPlateau(dec_opt, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience)

    encoder.train()
    decoder.train()

    # Initialize wandb
    import wandb

    config = {
        "device": str(device),
        "pad_id": args.pad_id,
        "tgt_pad_id": tgt_pad_id,
        "enc_vocab_size": encoder.emb.num_embeddings,
        "dec_vocab_size": decoder.emb.num_embeddings,
        "emb_dim": args.emb_dim,
        "hid_dim": args.hid_dim,
        "enc_num_layers": args.enc_num_layers,
        "dec_num_layers": args.dec_num_layers,
        "encoder_pad_idx": encoder.emb.padding_idx,
        "decoder_pad_idx": decoder.emb.padding_idx,
        "batch_size": args.batch_size,
        "lr_enc": args.lr_enc,
        "lr_dec": args.lr_dec,
        "optim_enc": type(enc_opt).__name__,
        "optim_dec": type(dec_opt).__name__,
        "criterion": type(criterion).__name__,
        "dataset_len": len(dataset),
        "dataset_features": list(dataset.features.keys()),
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "tgt_tokenizer": getattr(tgt_tokenizer, "name_or_path", str(tgt_tokenizer)),
    }

    print("Prepared wandb config:", config)
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=config
    )

    best_val_loss = float("inf")
    patience_ctr = 0

    for epoch in range(args.num_epochs):
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

            total_loss += loss.item()
            wandb.log(
                {"train/batch_loss": loss.item(),
                "lr/encoder": enc_opt.param_groups[0]["lr"],
                "lr/decoder": dec_opt.param_groups[0]["lr"],
            })

        train_loss = total_loss / len(loader)
        val_loss, val_bleu = evaluate(
            encoder, decoder, val_loader, criterion, device, tgt_tokenizer
        )
        sched_enc.step(val_loss)
        sched_dec.step(val_loss)

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/bleu": val_bleu,
        })

        print(
            f"epoch {epoch}: "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_bleu={val_bleu:.2f}"
        )

        # -------- early stopping --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0

            # optional: save best model
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
            }, args.checkpoint_path)

            wandb.log({"early_stop/best_val_loss": best_val_loss})
        else:
            patience_ctr += 1
            wandb.log({"early_stop/patience": patience_ctr})

            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation on test set
    test_loss, test_bleu = evaluate(encoder, decoder, test_loader, criterion, device, tgt_tokenizer)
    
    wandb.log({
        "test/loss": test_loss,
        "test/bleu": test_bleu,
    })
    
    print(f"Test Loss: {test_loss:.4f} | Test BLEU: {test_bleu:.2f}")
    
    wandb.finish()
    
    return best_val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train translation model")
    
    # Dataset paths
    parser.add_argument("--train_dataset_path", type=str, default="datasets/coco_train_msg_captions",
                        help="Path to training dataset")
    parser.add_argument("--val_dataset_path", type=str, default="datasets/coco_val_msg_captions",
                        help="Path to validation/test dataset")
    parser.add_argument("--checkpoint_path", type=str, default="best_model.pt",
                        help="Path to save best model checkpoint")
    
    # Model architecture
    parser.add_argument("--src_vocab_size", type=int, default=71,
                        help="Source vocabulary size (messages)")
    parser.add_argument("--emb_dim", type=int, default=256,
                        help="Embedding dimension")
    parser.add_argument("--hid_dim", type=int, default=512,
                        help="Hidden dimension")
    parser.add_argument("--enc_num_layers", type=int, default=2,
                        help="Number of encoder layers")
    parser.add_argument("--dec_num_layers", type=int, default=2,
                        help="Number of decoder layers")
    parser.add_argument("--pad_id", type=int, default=70,
                        help="Padding ID for source sequences")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size")
    parser.add_argument("--lr_enc", type=float, default=1e-3,
                        help="Encoder learning rate")
    parser.add_argument("--lr_dec", type=float, default=1e-3,
                        help="Decoder learning rate")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--scheduler_patience", type=int, default=20,
                        help="LR scheduler patience")
    parser.add_argument("--scheduler_factor", type=float, default=0.9,
                        help="LR scheduler decay factor")
    
    # Logging
    parser.add_argument("--wandb_project", type=str, default="EmComm-Caption-Translator",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_name", type=str, default="translation_baseline",
                        help="Weights & Biases run name")
    
    args = parser.parse_args()
    main(args)