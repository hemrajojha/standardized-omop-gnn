"""
train_rotate.py
===============
Train RotatE KG embeddings on OMOP concept triples using PyKEEN.

Reference: Sun et al. "RotatE: Knowledge Graph Embedding by Relational
Rotation in Complex Space", ICLR 2019. https://arxiv.org/abs/1902.10197

Input:
    kg_triples.csv          head_concept_id, relation, tail_concept_id
    concept_vocab.json      (from build_patient_graphs.py) — concept_id → index
                            Only concepts appearing in patient graphs get embeddings

Output (--out_dir):
    rotate_model.pt         full PyKEEN pipeline result (model weights, metadata)
    entity_embeddings.pt    {concept_id: int → embedding tensor [embed_dim]}
    relation_embeddings.pt  {relation: str → embedding tensor [embed_dim]}
    kg_vocab.json           {concept_id: entity_index} as used by PyKEEN

Usage
-----
python train_rotate.py \\
    --triples_path /path/to/omop_export/kg_triples.csv \\
    --vocab_path   /path/to/processed/concept_vocab.json \\
    --out_dir      /path/to/processed/kg_embeddings \\
    --embed_dim    128 \\
    --num_epochs   200 \\
    --batch_size   4096 \\
    --device       cuda

Notes
-----
- RotatE requires embed_dim to be even (real + imaginary parts)
- Filter triples to concepts that appear in patient data to keep embedding
  table size manageable; full OMOP concept graph can be 10M+ triples
- entity_embeddings.pt maps concept_id → numpy array for model injection
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train RotatE embeddings on OMOP KG triples")
    p.add_argument("--triples_path", required=True,  help="Path to kg_triples.csv")
    p.add_argument("--vocab_path",   required=True,  help="Path to concept_vocab.json")
    p.add_argument("--out_dir",      required=True,  help="Output directory")
    p.add_argument("--embed_dim",    type=int, default=128,
                   help="Embedding dimension (must be even for RotatE, default: 128)")
    p.add_argument("--num_epochs",   type=int, default=200)
    p.add_argument("--batch_size",   type=int, default=4096)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--num_neg_per_pos", type=int, default=64,
                   help="Negative samples per positive triple")
    p.add_argument("--device",       default="cuda",
                   help="'cuda' or 'cpu'")
    p.add_argument("--filter_to_patient_concepts", action="store_true", default=True,
                   help="Only keep triples where both head AND tail appear in patient graphs")
    p.add_argument("--no_filter", action="store_true",
                   help="Train on all KG triples (overrides --filter_to_patient_concepts)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Triple loading and filtering
# ---------------------------------------------------------------------------

def load_triples(triples_path: Path, vocab_path: Path, filter_to_patient: bool):
    """Load kg_triples.csv, optionally filtering to concepts in patient graphs."""
    log.info("Loading kg_triples.csv …")
    triples = pd.read_csv(
        triples_path,
        usecols=["head_concept_id", "relation", "tail_concept_id"],
    )
    triples["head_concept_id"] = triples["head_concept_id"].astype(int)
    triples["tail_concept_id"] = triples["tail_concept_id"].astype(int)
    log.info("  Loaded %d raw triples", len(triples))

    if filter_to_patient:
        log.info("Loading concept_vocab.json for patient-concept filter …")
        with open(vocab_path) as f:
            vocab_data = json.load(f)

        # Union of all concept_ids that appear in patient graphs
        patient_concept_ids: set[int] = set()
        for domain_vocab in vocab_data.values():
            patient_concept_ids.update(int(k) for k in domain_vocab.keys())
        log.info("  %d unique concept IDs in patient graphs", len(patient_concept_ids))

        before = len(triples)
        mask = (
            triples["head_concept_id"].isin(patient_concept_ids) |
            triples["tail_concept_id"].isin(patient_concept_ids)
        )
        triples = triples[mask]
        log.info("  Filtered %d → %d triples (kept triples with ≥1 patient concept)",
                 before, len(triples))

    return triples


# ---------------------------------------------------------------------------
# PyKEEN training
# ---------------------------------------------------------------------------

def train_rotate(triples: pd.DataFrame, args, out_dir: Path):
    """Train RotatE using PyKEEN pipeline and save embeddings."""
    try:
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
    except ImportError:
        log.error("PyKEEN not installed. Run: pip install pykeen")
        sys.exit(1)

    log.info("Building PyKEEN TriplesFactory …")
    # PyKEEN expects (head, relation, tail) as strings
    triples_array = np.stack([
        triples["head_concept_id"].astype(str).values,
        triples["relation"].values,
        triples["tail_concept_id"].astype(str).values,
    ], axis=1)

    tf = TriplesFactory.from_labeled_triples(
        triples=triples_array,
        create_inverse_triples=True,   # RotatE benefits from inverse relations
    )
    log.info("  Entities: %d  |  Relations: %d  |  Triples: %d",
             tf.num_entities, tf.num_relations, tf.num_triples)

    # Split into train/valid/test
    train_tf, valid_tf, test_tf = tf.split([0.90, 0.05, 0.05], random_state=42)
    log.info("  Train: %d  |  Valid: %d  |  Test: %d",
             train_tf.num_triples, valid_tf.num_triples, test_tf.num_triples)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    log.info("Training RotatE on %s  (dim=%d, epochs=%d, batch=%d) …",
             device, args.embed_dim, args.num_epochs, args.batch_size)

    result = pipeline(
        training=train_tf,
        validation=valid_tf,
        testing=test_tf,
        model="RotatE",
        model_kwargs=dict(
            embedding_dim=args.embed_dim,   # RotatE uses embed_dim/2 per part internally
        ),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=args.lr),
        training_loop="sLCWA",             # stochastic local closed-world assumption
        negative_sampler="basic",
        negative_sampler_kwargs=dict(num_negs_per_pos=args.num_neg_per_pos),
        loss="nssa",                        # self-adversarial negative sampling loss
        loss_kwargs=dict(margin=6.0, adversarial_temperature=1.0),
        epochs=args.num_epochs,
        training_kwargs=dict(batch_size=args.batch_size),
        device=device,
        random_seed=42,
        result_tracker=None,
    )

    return result, tf


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(result, tf, out_dir: Path):
    """
    Extract entity embeddings as {concept_id_str → numpy_array} and save.
    Also saves a {concept_id: int → local_entity_index} mapping.
    """
    log.info("Extracting entity embeddings …")
    model = result.model.cpu()

    # Entity embedding matrix: [num_entities, embed_dim]
    entity_repr = model.entity_representations[0]
    # RotatE stores complex embeddings; take real part or full vector
    try:
        emb_matrix = entity_repr(
            indices=torch.arange(tf.num_entities)
        ).detach().numpy()
    except Exception:
        # Fallback: access underlying weight directly
        emb_matrix = entity_repr._embeddings.weight.detach().numpy()

    log.info("  Entity embedding matrix shape: %s", emb_matrix.shape)

    # Map back from entity label (concept_id string) → embedding vector
    entity_to_idx = tf.entity_to_id  # dict: str(concept_id) → int
    concept_id_to_embedding: dict[int, np.ndarray] = {}
    for label, idx in entity_to_idx.items():
        try:
            cid = int(label)
            concept_id_to_embedding[cid] = emb_matrix[idx]
        except ValueError:
            pass  # skip non-integer labels

    log.info("  Extracted %d concept embeddings", len(concept_id_to_embedding))

    # Save full embedding lookup as a dict of tensors
    emb_tensors = {cid: torch.tensor(emb, dtype=torch.float)
                   for cid, emb in concept_id_to_embedding.items()}
    out_entity = out_dir / "entity_embeddings.pt"
    torch.save(emb_tensors, out_entity)
    log.info("Saved %s", out_entity)

    # Save kg_vocab.json: {str(concept_id): entity_index}
    out_kg_vocab = out_dir / "kg_vocab.json"
    with open(out_kg_vocab, "w") as f:
        json.dump({str(k): int(v) for k, v in entity_to_idx.items()}, f)
    log.info("Saved %s", out_kg_vocab)

    # Save embedding matrix as numpy for direct indexing
    out_matrix = out_dir / "entity_embedding_matrix.npy"
    np.save(out_matrix, emb_matrix)
    log.info("Saved %s  shape=%s", out_matrix, emb_matrix.shape)

    return emb_tensors, entity_to_idx


def extract_relation_embeddings(result, tf, out_dir: Path):
    log.info("Extracting relation embeddings …")
    model = result.model.cpu()
    rel_repr = model.relation_representations[0]
    try:
        rel_matrix = rel_repr(
            indices=torch.arange(tf.num_relations)
        ).detach().numpy()
    except Exception:
        rel_matrix = rel_repr._embeddings.weight.detach().numpy()

    rel_to_idx = tf.relation_to_id
    rel_tensors = {rel: torch.tensor(rel_matrix[idx], dtype=torch.float)
                   for rel, idx in rel_to_idx.items()}

    out_rel = out_dir / "relation_embeddings.pt"
    torch.save(rel_tensors, out_rel)
    log.info("Saved %s", out_rel)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.embed_dim % 2 != 0:
        log.error("--embed_dim must be even for RotatE (got %d)", args.embed_dim)
        sys.exit(1)

    filter_flag = args.filter_to_patient_concepts and not args.no_filter

    triples = load_triples(
        triples_path=Path(args.triples_path),
        vocab_path=Path(args.vocab_path),
        filter_to_patient=filter_flag,
    )

    if len(triples) == 0:
        log.error("No triples after filtering. Check --triples_path and --vocab_path.")
        sys.exit(1)

    result, tf = train_rotate(triples, args, out_dir)

    # Save full pipeline result (model + metadata)
    out_model = out_dir / "rotate_pipeline.pt"
    result.save_to_directory(str(out_dir / "rotate_pipeline"))
    log.info("Pipeline saved to %s", out_dir / "rotate_pipeline")

    # Extract and save embeddings in convenient format
    extract_embeddings(result, tf, out_dir)
    extract_relation_embeddings(result, tf, out_dir)

    # Print validation metrics
    log.info("Validation metrics:")
    metrics = result.metric_results.to_dict()
    for k, v in metrics.items():
        if "mean_rank" in k or "hits_at" in k:
            log.info("  %-40s %.4f", k, v)

    log.info("Done. Embeddings in %s", out_dir)


if __name__ == "__main__":
    main()
