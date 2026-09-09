# Pretraining Object Storage and On-Demand Caching

Original video backups, encoded latents, and training metadata are distinct assets. This workflow backs up **validated pretraining latents, text embeddings, snapshots, and text indices** to your configured object storage. Keep a separate backup of original downloads, including the official revision, original files, and integrity records. This guide covers video pretraining assets, not downstream policy fine-tuning or evaluation datasets.

All commands use the standard AWS credential provider chain. Supply credentials through your AWS profile, environment configuration, or role. Set `AWS_ENDPOINT_URL` for a custom compatible object-storage endpoint. The examples do not prescribe a particular bucket, account, or private endpoint.

## 1. Explicitly inventory the files to upload

```bash
export LATENT_BUCKET=YOUR_BUCKET
export LATENT_PREFIX=openwam/latents/v1

python scripts/pretraining/storage/inventory.py \
  --snapshot "$WORK_ROOT/snapshots/phase-001" \
  --text-cache "$WORK_ROOT/text/phase-001" \
  --out "$WORK_ROOT/storage/phase-001-inventory.jsonl"
```

The inventory includes only tensors, CSVs, `snapshot.json`, the text index, the prompt inventory, and the corresponding embeddings referenced by the snapshot. It does not recursively scan unrelated data directories. Each entry records the size, modification time, inode, and resolved path.

## 2. Upload and independently verify

```bash
python scripts/pretraining/storage/upload.py \
  "$WORK_ROOT/storage/phase-001-inventory.jsonl" \
  --name phase-001 --out-root "$WORK_ROOT/storage" \
  --bucket "$LATENT_BUCKET" --prefix "$LATENT_PREFIX" --workers 4
```

Object keys are content-addressed: `<prefix>/objects/<first two SHA256 characters>/<full SHA256>`. Identical content does not create a second object; different content does not overwrite an existing object. Both ordinary uploads and multipart completion use conditional writes.

The uploader checks file identity before upload, after hashing, and after upload. Verification first checks the remote full-object SHA256. If the endpoint does not provide a comparable full-object SHA256, the verifier performs a complete GET and recomputes the hash. **A multipart ETag is not the file SHA256** and cannot replace content verification.

Receipts are written to `storage/receipts/phase-001.jsonl`. Rerunning resumes the upload. Errors or files changing during upload cause the command to fail; those files are not ready for cleanup. Do not modify a file while including it in an immutable snapshot.

## 3. Seal the catalog and recovery metadata

```bash
python scripts/pretraining/storage/seal.py \
  --inventory "$WORK_ROOT/storage/phase-001-inventory.jsonl" \
  --receipts "$WORK_ROOT/storage/receipts/phase-001.jsonl" \
  --out "$WORK_ROOT/storage/sealed/phase-001" \
  --bucket "$LATENT_BUCKET" --prefix "$LATENT_PREFIX"
```

Sealing requires full inventory coverage and a verified upload matching every manifest latent's SHA256 and size. It creates a read-only SQLite `catalog.sqlite`, uploads and verifies that catalog, and also uploads `restore.json`. Preserve the small `restore_spec.json` file: it contains the recovery manifest's bucket, key, size, SHA256, and ETag. An unverified directory URL alone is insufficient for recovery.

To continue pretraining from the same directory layout, generate an explicit config:

```bash
python scripts/pretraining/make_config.py \
  --snapshot "$WORK_ROOT/snapshots/phase-001" \
  --model-assets "$MODEL_ASSETS" --run-root "$WORK_ROOT/runs" \
  --out "$WORK_ROOT/configs/phase-001-remote.yaml" \
  --text-cache "$WORK_ROOT/text/phase-001" \
  --catalog-spec "$WORK_ROOT/storage/sealed/phase-001/catalog_spec.json" \
  --object-cache-dir /path/to/local-cache/openwam-objects \
  --object-cache-gib 200 --object-cache-min-free-gib 8
```

This records `data.artifact_cache` and `backbone.prompt_cache.artifact_cache`, each with the catalog specification, cache directory, byte budget, and free-space reserve. Registered paths use verified catalog objects even when a local file exists. Unregistered paths retain ordinary local loading; no object key is guessed. With neither field configured, local/Hugging Face loading is unchanged. AWS credentials and endpoint selection continue to use the standard SDK environment.

## 4. Restore metadata into a different directory

```bash
python scripts/pretraining/storage/restore.py \
  --restore-spec /path/to/saved/restore_spec.json \
  --out "$WORK_ROOT/restored/phase-001" \
  --cache-dir /path/to/local-cache/openwam-latents --cache-gib 200 \
  --bucket "$LATENT_BUCKET" --prefix "$LATENT_PREFIX"

# Pass --catalog-spec "$WORK_ROOT/restored/phase-001/catalog_spec.json"
# and --text-cache "$WORK_ROOT/restored/phase-001/text_cache" to make_config.py.
```

Restoration downloads and verifies the small snapshot and text metadata files, then creates catalog aliases for the relocated manifest and embedding paths. Latent tensors and text embeddings are fetched only when used. The new alias catalog is uploaded to the specified bucket, so this step requires write access to the destination prefix. Original manifest content hashes are retained as recovery provenance, while relocated manifests receive newly computed hashes. Next, run `make_config.py` against the restored `snapshot/` directory.

## 5. Cache limits and concurrency

The shared bounded object cache is used by the data loader and text cache through `src/open_wam/artifacts/resolver.py`.

- Objects are addressed by SHA256. Download bytes are reserved before transfer, and in-flight objects count toward the budget.
- Every process sharing a cache directory must use the same byte budget. Separate 200 GiB caches per process do not constitute a shared 200 GiB budget.
- A given object is downloaded once. Interprocess file locks protect downloading and reading. Readers hold shared leases; LRU eviction requires an exclusive lease, so it cannot delete an object that is still being read.
- SHA256 and size are checked on downloads and cache hits. Corrupt hits are downloaded again. Only verified temporary files are atomically renamed into their final object paths.
- SQLite tracks ready/downloading states, LRU information, and statistics. Failed downloads return their reserved budget, and restart recovery can reclaim abandoned reservations.
- Insufficient free space, an object larger than the budget, or eviction candidates all protected by read leases produce explicit failures rather than unbounded cache growth.

The 200 GiB limit applies to cache payloads, including the catalog and in-flight objects. It excludes small additional overheads such as SQLite, lock files, retained manifests, and system logs. SHA verification itself consumes disk-read bandwidth. Measure actual pretraining throughput rather than assuming that every cache hit is free.

## 6. When original local files can be removed

These tools do not automatically remove raw datasets or old training directories. Before removing files, establish the following:

1. All target latents have encoded successfully and passed tensor/receipt validation.
2. Original videos have a recoverable official copy or your own verified per-object backup; a same-named directory alone is not evidence of a backup.
3. Latents, text, and metadata have been sealed, and restoration to a new location plus representative sample reads have succeeded.
4. The active pretraining configuration uses the intended snapshot/catalog, and running encoding processes no longer require the original videos.

Once these conditions hold, the corresponding original videos and large local latent copies managed through the catalog can be removed. Retain metadata, source revisions, verification receipts, `restore_spec`, and training configurations. Do not remove archive members still being processed or unpublished latents. LRU eviction reclaims old cache objects automatically; deleting the entire cache is unnecessary for maintaining its budget.
