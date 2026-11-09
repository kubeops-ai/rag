# KubeOps RAG

## Current pipeline

The pipeline currently has five simple steps:

1. Load one document from S3.
2. Clean that document in memory.
3. Split the cleaned Markdown document by headings.
4. Create one embedding vector for each chunk.
5. Store the vectors, content, and metadata in Qdrant.

Before loading a document, the pipeline compares its S3 `ETag` and
`LastModified` values with the values stored in Qdrant. Unchanged documents are
skipped.

Only files ending with `.md` are processed. Other S3 objects are skipped.

Every Qdrant chunk also stores document version metadata:

- `document_version`: S3 Version ID, or ETag when versioning is disabled
- `upload_date`: S3 object timestamp
- `last_modified_date`: S3 object timestamp

Changed sections are collected and sent in one embedding request. Unchanged
sections do not use an embedding request.

Temporary S3, OpenAI, and Qdrant failures are retried up to three times with a
short increasing delay. Permanent failures are raised after the final attempt.

After S3 processing, the pipeline compares the current S3 file names with
Qdrant `source_key` values. If a Markdown file was deleted from S3, all of its
Qdrant points are deleted as well.

The document is not saved to the local disk.

```text
S3 → load one file → clean text → create heading chunks → embeddings → Qdrant
```

## Setup

Install the Python dependency:

```powershell
python -m pip install -r requirements.txt
```

The code uses the AWS CLI profile named `kubeops-rag`:

```powershell
aws configure --profile kubeops-rag
```

## Run the pipeline

```powershell
python .\rag_pipeline.py
```

## How the code works

### `create_s3_client()`

Creates a connection to S3 using the local AWS profile.

### `get_s3_files()`

Gets file names from the S3 bucket. It does not download the files.

### `load_one_file()`

Loads only one S3 file into memory.

### `clean_text()`

Removes unwanted control characters, extra spaces, and repeated blank lines.

### `chunk_document()`

Creates one chunk for each Markdown section. It recognizes headings such as
`#`, `##`, and `###`, and keeps the parent heading path with each chunk.

### `run_pipeline()`

Calls the functions in order. After one file is cleaned, the loop moves to the
next file. Each changed document is embedded and saved to Qdrant.

### `create_openai_client()`

Creates an OpenAI client using `OPENAI_API_KEY` from the environment.

Each embedded chunk receives metadata:

- S3 bucket and object key
- File name
- Chunk number
- Heading and heading path
- Embedding model name

### `save_chunks_to_qdrant()`

Creates or reuses the local `kubeops_documents` collection and stores each
embedding with its chunk content and metadata. The local Qdrant server must be
running at `http://localhost:6333`.

The save step is incremental. It calculates a hash for each section:

- Unchanged sections are skipped.
- New or changed sections receive a new embedding and are upserted.
- Sections removed from the document are deleted from Qdrant.

Section IDs are based on the heading path, so changing text inside one section
does not require re-embedding unchanged sections.


## Future steps

The same `rag_pipeline.py` file will later contain:

1. Retrieval and answer generation



## Never Delete this lines below RAG Ingestion Pipeline process.

I built a RAG ingestion pipeline where Amazon S3 acts as the source of truth for troubleshooting documents. The pipeline lists the files in S3 and processes one document at a time in memory instead of storing copies locally. Each document is cleaned by removing control characters, normalizing Unicode, removing extra spaces, and reducing repeated blank lines. The cleaned Markdown is then split into structured chunks based on headings such as #, ##, and ###, while preserving the heading path and section content. Every chunk receives metadata including the S3 bucket, object key, file name, section ID, content hash, heading, heading path, and embedding model. The section ID identifies the logical section, while the content hash identifies its current content. During incremental processing, the pipeline compares both values with the existing Qdrant records. Unchanged sections are skipped, changed sections are embedded again, new sections are inserted, and deleted sections are removed from Qdrant. For changed or new chunks, the pipeline uses OpenAI’s text-embedding-3-small model to generate 1,536-dimensional vectors. These vectors, along with the original chunk content and metadata, are stored in the local Qdrant collection named kubeops_documents using cosine similarity. This pipeline currently handles ingestion, cleaning, chunking, embedding, and vector storage; retrieval and answer generation will be added in the next RAG stage.
