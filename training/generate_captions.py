"""Generate Russian product cards via LLM API (OpenRouter) for training data.

Usage:
    python training/generate_captions.py \
        --input training/data/metadata.jsonl \
        --output training/data/captions.jsonl \
        --model google/gemini-2.0-flash-001
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a product card generator for a Russian e-commerce marketplace.
You receive English product attributes and must return a JSON object with exactly three fields:
- "title": Russian product title, max 70 characters, marketplace style. \
Do not use words like "лучший", "самый", "уникальный".
- "description": Russian marketing description, 200-400 characters. \
Informative but not watery.
- "caption_ru": Short Russian caption as if describing what is visible in the product photo, \
max 150 characters, factual.

Respond ONLY with valid JSON, no markdown, no extra text.\
"""

MAX_RETRIES = 3


def build_user_prompt(record: dict) -> str:
    """Build a user prompt from metadata fields."""
    parts = [
        f"Product: {record.get('product_display_name', 'N/A')}",
        f"Category: {record.get('master_category', 'N/A')}",
        f"Subcategory: {record.get('sub_category', 'N/A')}",
        f"Article type: {record.get('article_type', 'N/A')}",
        f"Color: {record.get('base_colour', 'N/A')}",
        f"Season: {record.get('season', 'N/A')}",
        f"Usage: {record.get('usage', 'N/A')}",
    ]
    return "\n".join(parts)


async def generate_one(
    client: AsyncOpenAI,
    record: dict,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Generate a Russian card for a single record with retry logic."""
    user_prompt = build_user_prompt(record)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    max_tokens=512,
                )

            content = response.choices[0].message.content
            data = json.loads(content)

            result = {
                "id": record["id"],
                "title": data.get("title", ""),
                "description": data.get("description", ""),
                "caption_ru": data.get("caption_ru", ""),
            }

            # Token usage for cost estimation
            usage = response.usage
            if usage:
                result["_prompt_tokens"] = usage.prompt_tokens
                result["_completion_tokens"] = usage.completion_tokens

            return result

        except (RateLimitError, APIConnectionError) as e:
            wait = 2 ** attempt
            logger.warning("Attempt %d/%d for id=%s: %s. Retrying in %ds...", attempt, MAX_RETRIES, record["id"], e, wait)
            await asyncio.sleep(wait)
        except (APIError, json.JSONDecodeError) as e:
            logger.error("Attempt %d/%d for id=%s failed: %s", attempt, MAX_RETRIES, record["id"], e)
            if attempt == MAX_RETRIES:
                return None
            await asyncio.sleep(2 ** attempt)

    return None


async def main_async(args: argparse.Namespace):
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Read metadata
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), input_path)

    # Read already processed IDs for idempotency
    processed_ids: set = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        processed_ids.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        logger.info("Found %d already processed records, skipping them", len(processed_ids))

    # Filter out already processed
    to_process = [r for r in records if r["id"] not in processed_ids]
    if not to_process:
        logger.info("All records already processed, nothing to do")
        return

    logger.info("Will process %d records (skipping %d)", len(to_process), len(processed_ids))

    # Init OpenRouter client
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable is not set")
        raise SystemExit(1)

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/mixelka75/snapcard",
            "X-Title": "SnapCard Training",
        },
    )

    semaphore = asyncio.Semaphore(5)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    success_count = 0
    fail_count = 0

    # Process with progress bar
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=len(to_process), desc="Generating captions")

    async def process_and_write(record: dict):
        nonlocal total_prompt_tokens, total_completion_tokens, success_count, fail_count

        result = await generate_one(client, record, args.model, semaphore)
        if result:
            # Extract usage stats before writing
            prompt_t = result.pop("_prompt_tokens", 0)
            completion_t = result.pop("_completion_tokens", 0)
            total_prompt_tokens += prompt_t
            total_completion_tokens += completion_t

            # Append to file immediately
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            success_count += 1
        else:
            fail_count += 1
        pbar.update(1)

    # Run with bounded concurrency
    tasks = [process_and_write(r) for r in to_process]
    await asyncio.gather(*tasks)
    pbar.close()

    # Summary
    total_tokens = total_prompt_tokens + total_completion_tokens
    # Rough cost estimate for gemini-flash: ~$0.075/1M input, ~$0.30/1M output
    est_cost = (total_prompt_tokens * 0.075 + total_completion_tokens * 0.30) / 1_000_000

    logger.info("=" * 50)
    logger.info("Generation complete!")
    logger.info("  Processed: %d", success_count)
    logger.info("  Skipped (already done): %d", len(processed_ids))
    logger.info("  Failed: %d", fail_count)
    logger.info("  Total tokens: %d (prompt: %d, completion: %d)", total_tokens, total_prompt_tokens, total_completion_tokens)
    logger.info("  Estimated cost: $%.4f", est_cost)
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Generate Russian captions via OpenRouter LLM API")
    parser.add_argument("--input", type=str, default="training/data/metadata.jsonl", help="Input metadata JSONL")
    parser.add_argument("--output", type=str, default="training/data/captions.jsonl", help="Output captions JSONL")
    parser.add_argument("--model", type=str, default="google/gemini-2.0-flash-001", help="LLM model to use")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
