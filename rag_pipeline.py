"""Simple, step-by-step RAG pipeline.

Currently this file only loads and cleans documents. Each document is read
from S3, cleaned in memory, and released before the next document is read.
"""

import hashlib
import logging
import os
import re
import time
import unicodedata
from uuid import NAMESPACE_URL, uuid5

import boto3
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)


# These values identify the S3 location and the local AWS profile.
AWS_PROFILE = "kubeops-rag"
AWS_REGION = "ap-south-1"
S3_BUCKET = "anand-rag-repository-248385581008-ap-south-1-an"
EMBEDDING_MODEL = "text-embedding-3-small"
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "kubeops_documents"
logger = logging.getLogger("kubeops.rag")
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


def setup_logging():
    """Configure readable logs for each pipeline step."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # Keep Qdrant HTTP details out of the normal learning logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def retry_operation(operation, description):
    """Retry a temporary external-service failure up to three times."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == MAX_RETRIES:
                logger.error("%s failed after %d attempts.", description, attempt)
                raise
            delay = RETRY_DELAY_SECONDS * attempt
            logger.warning(
                "%s failed on attempt %d/%d: %s. Retrying in %d seconds.",
                description,
                attempt,
                MAX_RETRIES,
                error,
                delay,
            )
            time.sleep(delay)


def create_s3_client():
    """Create an S3 client using the configured local AWS profile."""
    logger.info("Step 1: connecting to S3 with profile '%s'.", AWS_PROFILE)
    session = boto3.Session(
        profile_name=AWS_PROFILE,
        region_name=AWS_REGION,
    )
    client = session.client("s3")
    logger.info("S3 client is ready for bucket '%s'.", S3_BUCKET)
    return client


def get_s3_files(s3_client):
    """Return S3 object metadata one file at a time."""
    logger.info("Step 2: listing files in the S3 bucket.")
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET):
        for item in page.get("Contents", []):
            file_name = item["Key"]

            # S3 sometimes stores folder markers. They are not documents.
            if file_name.endswith("/"):
                continue

            # This pipeline currently supports Markdown files only.
            if not file_name.lower().endswith(".md"):
                logger.info("Skipping unsupported file: %s", file_name)
                continue

            # Get the S3 version when bucket versioning is enabled.
            object_details = retry_operation(
                lambda: s3_client.head_object(Bucket=S3_BUCKET, Key=file_name),
                f"Read S3 metadata for {file_name}",
            )
            item["VersionId"] = object_details.get("VersionId")
            item["LastModified"] = object_details.get(
                "LastModified", item.get("LastModified")
            )
            item["ETag"] = object_details.get("ETag", item.get("ETag", ""))
            logger.info("Found supported Markdown file: %s", file_name)
            yield item


def load_one_file(s3_client, file_name):
    """Load one S3 file into memory and return it as text."""
    logger.info("Step 3: loading one file into memory: %s", file_name)
    response = retry_operation(
        lambda: s3_client.get_object(Bucket=S3_BUCKET, Key=file_name),
        f"Load S3 file {file_name}",
    )
    file_bytes = response["Body"].read()
    logger.info("Loaded %d bytes from %s.", len(file_bytes), file_name)
    return file_bytes.decode("utf-8")


def clean_text(text):
    """Remove unwanted symbols and extra spaces from one document."""
    logger.info("Step 4: cleaning the document in memory.")
    # Normalize compatible Unicode characters and replace non-breaking spaces.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")

    cleaned_lines = []
    previous_line_was_blank = False

    for line in text.splitlines():
        # Remove control characters while keeping normal tab characters.
        line = "".join(
            character
            for character in line
            if character == "\t" or not unicodedata.category(character).startswith("C")
        )

        # Replace repeated spaces and remove spaces at line boundaries.
        line = re.sub(r"[ \t]+", " ", line).strip()

        # Keep no more than one empty line between sections.
        if line == "":
            if previous_line_was_blank:
                continue
            previous_line_was_blank = True
        else:
            previous_line_was_blank = False

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines).strip() + "\n"
    logger.info("Cleaning complete. Result contains %d characters.", len(cleaned_text))
    return cleaned_text


def chunk_document(cleaned_text):
    """Split one Markdown document into chunks using its headings."""
    chunks = []
    heading_path = []
    current_heading = "Introduction"
    current_lines = []

    def save_current_chunk():
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append(
                {
                    "heading": current_heading,
                    "heading_path": list(heading_path),
                    "content": content,
                }
            )

    for line in cleaned_text.splitlines():
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)

        if heading_match:
            save_current_chunk()
            current_lines = []
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            # Keep parent headings and remove headings at the same or deeper level.
            heading_path[:] = [
                heading for heading in heading_path if heading[0] < level
            ]
            heading_path.append((level, title))
            current_heading = title
        else:
            current_lines.append(line)

    save_current_chunk()
    return chunks


def create_openai_client():
    """Create an OpenAI client using OPENAI_API_KEY from the environment."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before running the embedding step.")
    logger.info("Step 6: connecting to OpenAI embeddings.")
    return OpenAI(api_key=api_key)


def add_chunk_metadata(file_info, chunks):
    """Add stable metadata and a hash to every chunk."""
    file_name = file_info["Key"]
    etag = file_info.get("ETag", "").strip('"')
    last_modified = str(file_info.get("LastModified", ""))
    document_version = file_info.get("VersionId") or etag
    for number, chunk in enumerate(chunks, start=1):
        section_name = " > ".join(title for level, title in chunk["heading_path"])
        section_id = hashlib.sha256(section_name.encode("utf-8")).hexdigest()
        content_hash = hashlib.sha256(chunk["content"].encode("utf-8")).hexdigest()
        chunk["metadata"] = {
            "source_bucket": S3_BUCKET,
            "source_key": file_name,
            "source_etag": etag,
            "source_last_modified": last_modified,
            "document_version": document_version,
            "upload_date": last_modified,
            "last_modified_date": last_modified,
            "file_name": file_name.split("/")[-1],
            "chunk_number": number,
            "section_id": section_id,
            "content_hash": content_hash,
            "heading": chunk["heading"],
            "heading_path": [title for level, title in chunk["heading_path"]],
            "embedding_model": EMBEDDING_MODEL,
        }
    return chunks


def get_existing_chunks(qdrant, file_name):
    """Read existing Qdrant metadata for one source document."""
    source_filter = Filter(
        must=[FieldCondition(key="source_key", match=MatchValue(value=file_name))]
    )
    records, _ = retry_operation(
        lambda: qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=source_filter,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        ),
        "Read existing Qdrant sections",
    )
    return records


def save_chunks_to_qdrant(openai_client, qdrant, file_info, chunks):
    """Embed changed sections and synchronize them with Qdrant."""
    if not chunks:
        logger.info("There are no chunks to save in Qdrant.")
        return

    file_name = file_info["Key"]
    add_chunk_metadata(file_info, chunks)
    vector_size = 1536

    if not qdrant.collection_exists(QDRANT_COLLECTION):
        logger.info("Creating Qdrant collection '%s'.", QDRANT_COLLECTION)
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    existing_records = get_existing_chunks(qdrant, file_name)
    logger.info(
        "Existing Qdrant sections for %s: %d.",
        file_name,
        len(existing_records),
    )
    existing_by_section = {
        record.payload.get("section_id"): record
        for record in existing_records
        if record.payload
    }
    current_sections = set()
    points = []
    chunks_to_embed = []
    unchanged_count = 0
    changed_count = 0
    new_count = 0

    for chunk in chunks:
        metadata = chunk["metadata"]
        section_id = metadata["section_id"]
        current_sections.add(section_id)
        old_record = existing_by_section.get(section_id)
        if (
            old_record
            and old_record.payload.get("content_hash") == metadata["content_hash"]
        ):
            unchanged_count += 1
            logger.info(
                "Section comparison: UNCHANGED | %s | hash=%s",
                metadata["heading"],
                metadata["content_hash"][:12],
            )
            logger.info("Updating metadata only; vector is reused: %s", metadata["heading"])
            # Keep S3 version metadata current without creating a new vector.
            qdrant.set_payload(
                collection_name=QDRANT_COLLECTION,
                payload=metadata,
                points=[old_record.id],
                wait=True,
            )
            continue

        if old_record:
            changed_count += 1
            comparison = "CHANGED"
        else:
            new_count += 1
            comparison = "NEW"
        logger.info(
            "Section comparison: %s | %s | hash=%s",
            comparison,
            metadata["heading"],
            metadata["content_hash"][:12],
        )
        chunks_to_embed.append(chunk)

    if chunks_to_embed:
        logger.info(
            "Creating one embedding request for %d changed section(s).",
            len(chunks_to_embed),
        )
        response = retry_operation(
            lambda: openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[chunk["content"] for chunk in chunks_to_embed],
            ),
            "Create OpenAI embeddings",
        )

        for chunk, embedding_result in zip(chunks_to_embed, response.data):
            metadata = chunk["metadata"]
            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"{S3_BUCKET}/{file_name}/{metadata['section_id']}",
                )
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding_result.embedding,
                    payload={**metadata, "content": chunk["content"]},
                )
            )

    if points:
        retry_operation(
            lambda: qdrant.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points,
                wait=True,
            ),
            "Write changed sections to Qdrant",
        )
        logger.info("Upserted %d changed section(s).", len(points))
    else:
        logger.info("No changed sections needed new embeddings.")

    stale_ids = [
        record.id
        for section_id, record in existing_by_section.items()
        if section_id not in current_sections
    ]
    if stale_ids:
        qdrant.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=PointIdsList(points=stale_ids),
            wait=True,
        )
        logger.info("Deleted %d removed section(s).", len(stale_ids))

    logger.info(
        "Comparison summary: unchanged=%d, changed=%d, new=%d, removed=%d.",
        unchanged_count,
        changed_count,
        new_count,
        len(stale_ids),
    )


def is_file_unchanged(qdrant, file_info):
    """Check S3 metadata before loading a document."""
    if not qdrant.collection_exists(QDRANT_COLLECTION):
        return False

    records = get_existing_chunks(qdrant, file_info["Key"])
    if not records:
        return False

    current_etag = file_info.get("ETag", "").strip('"')
    current_last_modified = str(file_info.get("LastModified", ""))
    for record in records:
        payload = record.payload or {}
        if (
            payload.get("source_etag") != current_etag
            or payload.get("source_last_modified") != current_last_modified
        ):
            return False
    return True


def remove_deleted_files(qdrant, current_s3_files):
    """Delete Qdrant points for Markdown files removed from S3."""
    if not qdrant.collection_exists(QDRANT_COLLECTION):
        return

    all_records, _ = qdrant.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    old_file_names = {
        record.payload.get("source_key")
        for record in all_records
        if record.payload and record.payload.get("source_key")
    }
    deleted_file_names = old_file_names - current_s3_files

    for file_name in deleted_file_names:
        records = get_existing_chunks(qdrant, file_name)
        point_ids = [record.id for record in records]
        if not point_ids:
            continue
        qdrant.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=PointIdsList(points=point_ids),
            wait=True,
        )
        logger.info(
            "Deleted removed S3 file from Qdrant: %s (%d section(s)).",
            file_name,
            len(point_ids),
        )

def run_pipeline():
    """Run the first three pipeline steps for every S3 document."""
    logger.info("RAG pipeline started.")
    s3_client = create_s3_client()
    openai_client = create_openai_client()
    qdrant = QdrantClient(url=QDRANT_URL)
    processed_count = 0
    skipped_count = 0
    current_s3_files = set()

    for file_info in get_s3_files(s3_client):
        file_name = file_info["Key"]
        current_s3_files.add(file_name)
        if is_file_unchanged(qdrant, file_info):
            logger.info("File check: UNCHANGED in S3 | %s", file_name)
            skipped_count += 1
            continue

        logger.info(
            "File check: NEW or CHANGED in S3; checking sections | %s",
            file_name,
        )
        original_text = load_one_file(s3_client, file_name)
        cleaned_text = clean_text(original_text)
        logger.info("Step 5: creating heading-based chunks for %s.", file_name)
        chunks = chunk_document(cleaned_text)
        logger.info("Created %d chunk(s) for %s.", len(chunks), file_name)
        add_chunk_metadata(file_info, chunks)
        save_chunks_to_qdrant(openai_client, qdrant, file_info, chunks)
        logger.info("Finished current document: %s", file_name)
        processed_count += 1

    remove_deleted_files(qdrant, current_s3_files)

    logger.info(
        "Pipeline finished. Processed %d document(s), skipped %d unchanged document(s).",
        processed_count,
        skipped_count,
    )
    logger.info("No documents were saved on the local disk.")


if __name__ == "__main__":
    setup_logging()
    run_pipeline()
