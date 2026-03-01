import csv
import numpy as np
import random
import sacrebleu
import sys
import torch
import torch.nn as nn
from src.objects_game.src.translation import (
    Decoder, Encoder, TransformerDecoder, TransformerEncoder
)
from src.objects_game.helpers.train_translation_argument_handling import get_params
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

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

def evaluate(encoder, decoder, loader, criterion, device, tokenizer, sim_model, model_type="rnn", csv_save_path=None):
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    refs, hyps = [], []

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)

            if model_type == "rnn":
                h, c = encoder(src)
                logits, _, _ = decoder(tgt[:, :-1], h, c)
            else:  # transformer
                encoder_output, _ = encoder(src)
                logits, _, _ = decoder(tgt[:, :-1], encoder_output)

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

    # Compute semantic similarity
    hyp_embeddings = sim_model.encode(hyps, convert_to_tensor=True)
    ref_embeddings = sim_model.encode(refs, convert_to_tensor=True)
    
    # Compute cosine similarity for each pair
    cosine_scores = util.pytorch_cos_sim(hyp_embeddings, ref_embeddings)
    semantic_similarity = torch.diagonal(cosine_scores).mean().item()
    if csv_save_path is not None:
        with open(csv_save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["hypothesis", "reference"])
            writer.writerows(zip(hyps, refs))

    return total_loss / len(loader), bleu, semantic_similarity

def main(params):
    args = get_params(params)
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tgt_tokenizer, tgt_pad_id = get_tokenizer_and_pad()
    
    # Load semantic similarity model
    sim_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    sim_model.to(device)
    
    # Initialize models based on model_type
    if args.model_type == "rnn":
        encoder = Encoder(
            vocab_size=args.src_vocab_size,
            emb_dim=args.emb_dim,
            hid_dim=args.hid_dim,
            num_layers=args.enc_num_layers,
            pad_id=args.pad_id,
            dropout=args.dropout
        ).to(device)

        decoder = Decoder(
            vocab_size=len(tgt_tokenizer.vocab),
            emb_dim=args.emb_dim,
            hid_dim=args.hid_dim,
            num_layers=args.dec_num_layers,
            pad_id=tgt_pad_id,
            dropout=args.dropout
        ).to(device)
    else:  # transformer
        encoder = TransformerEncoder(
            vocab_size=args.src_vocab_size,
            emb_dim=args.emb_dim,
            hid_dim=args.hid_dim,
            num_layers=args.enc_num_layers,
            pad_id=args.pad_id,
            dropout=args.dropout,
            num_heads=args.num_heads
        ).to(device)

        decoder = TransformerDecoder(
            vocab_size=len(tgt_tokenizer.vocab),
            emb_dim=args.emb_dim,
            hid_dim=args.hid_dim,
            num_layers=args.dec_num_layers,
            pad_id=tgt_pad_id,
            dropout=args.dropout,
            num_heads=args.num_heads
        ).to(device)

    # If running in test-only mode we don't need training/validation data
    if args.test_only:
        if args.test_dataset_path is None:
            raise ValueError("--test_dataset_path must be specified when using --test_only")
        test_data = load_from_disk(args.test_dataset_path)
        test_loader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate(batch, args.pad_id, tgt_tokenizer)
        )
    else:
        # Load datasets for training/validation
        dataset = load_from_disk(args.train_dataset_path)
        val_test_dataset = load_from_disk(args.val_dataset_path)
        splits = val_test_dataset.train_test_split(test_size=0.5, seed=args.seed)
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

    # Setup loss and optimizers (only needed if training)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_id)
    config = {
    "device": str(device),
    "seed": args.seed,
    "pad_id": args.pad_id,
    "tgt_pad_id": tgt_pad_id,
    "enc_vocab_size": encoder.emb.num_embeddings,
    "dec_vocab_size": decoder.emb.num_embeddings,
    "emb_dim": args.emb_dim,
    "hid_dim": args.hid_dim,
    "enc_num_layers": args.enc_num_layers,
    "dec_num_layers": args.dec_num_layers,
    "dropout": args.dropout,
    "encoder_pad_idx": encoder.emb.padding_idx,
    "decoder_pad_idx": decoder.emb.padding_idx,
    "batch_size": args.batch_size,
    "lr_enc": args.lr_enc,
    "lr_dec": args.lr_dec,
    "criterion": type(criterion).__name__,
    "num_epochs": args.num_epochs,
    "patience": args.patience,
    "tgt_tokenizer": getattr(tgt_tokenizer, "name_or_path", str(tgt_tokenizer)),
    "semantic_similarity_model": "paraphrase-multilingual-mpnet-base-v2",
    }

    if not args.test_only:
        enc_opt = torch.optim.Adam(encoder.parameters(), lr=args.lr_enc)
        dec_opt = torch.optim.Adam(decoder.parameters(), lr=args.lr_dec)
        config.update({       
            "optim_enc": type(enc_opt).__name__,
            "optim_dec": type(dec_opt).__name__ ,
            "dataset_len": len(dataset),
            "dataset_features": list(dataset.features.keys()),
            })

        # Setup schedulers
        sched_enc = ReduceLROnPlateau(enc_opt, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience)
        sched_dec = ReduceLROnPlateau(dec_opt, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience)

        encoder.train()
        decoder.train()

    # Initialize wandb
    import wandb



    print("Prepared wandb config:", config)
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=config
    )

    # If test-only mode, load checkpoint and evaluate without training
    if args.test_only:
        print(f"Running in test-only mode. Loading checkpoint from {args.checkpoint_path} and evaluating on provided test dataset: {args.test_dataset_path}.")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        encoder.to(device)
        decoder.to(device)
        test_loss, test_bleu, test_semantic_sim = evaluate(
            encoder, 
            decoder, 
            test_loader, 
            criterion, 
            device, 
            tgt_tokenizer, 
            sim_model, 
            model_type=args.model_type,
            csv_save_path="test_predictions.csv"
        )
        print(f"Test Loss: {test_loss:.4f} | Test BLEU: {test_bleu:.2f} | Test Semantic Similarity: {test_semantic_sim:.4f}")
        return None

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

            if args.model_type == "rnn":
                h, c = encoder(src)
                logits, _, _ = decoder(tgt[:, :-1], h, c)
            else:  # transformer
                encoder_output, _ = encoder(src)
                logits, _, _ = decoder(tgt[:, :-1], encoder_output)

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
        val_loss, val_bleu, val_semantic_sim = evaluate(
            encoder, decoder, val_loader, criterion, device, tgt_tokenizer, sim_model, model_type=args.model_type
        )
        sched_enc.step(val_loss)
        sched_dec.step(val_loss)

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/bleu": val_bleu,
            "val/semantic_similarity": val_semantic_sim,
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
    test_loss, test_bleu, test_semantic_sim = evaluate(encoder, decoder, test_loader, criterion, device, tgt_tokenizer, sim_model, model_type=args.model_type)
    
    wandb.log({
        "test/loss": test_loss,
        "test/bleu": test_bleu,
        "test/semantic_similarity": test_semantic_sim,
    })
    
    print(f"Test Loss: {test_loss:.4f} | Test BLEU: {test_bleu:.2f} | Test Semantic Similarity: {test_semantic_sim:.4f}")
    
    wandb.finish()
    
    return best_val_loss

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_translation.py <config.yaml>")
        sys.exit(1)

    # forward the single path as a list so get_params can handle it
    main([sys.argv[1]])