import logging
from typing import List, Optional
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

def extract_main_content(soup: BeautifulSoup) -> Optional[Tag]:
    """Extracts the BeautifulSoup Tag for the main content area."""
    # Try specific and common content IDs/classes first
    selectors = [
        '#main-content',
        '#content',
        '.main-content',
        '.content',
        'main',
        'article'
    ]
    main_content_tag = None
    for selector in selectors:
        main_content_tag = soup.select_one(selector)
        if main_content_tag:
            logger.debug(f"Found main content using selector: '{selector}'")
            break

    if not main_content_tag:
        logger.warning(f"Could not find main content using common selectors. Falling back to body.")
        # Fallback to body if no specific container found
        main_content_tag = soup.find('body')
        if not main_content_tag:
             logger.error("Could not even find body tag in the HTML.")
             return None # Should not happen with valid HTML

    # Optional: Add logic here to remove known irrelevant sections like nav, headers, footers
    # E.g., for tag in main_content_tag.select('nav, header, footer, .sidebar'): tag.decompose()

    return main_content_tag

def chunk_by_html_headings(main_content_tag: Optional[Tag], max_chars: int = 20000) -> List[str]:
    """Chunks HTML text content based on heading tags (h2, h3, h4) within the main content area.

    Args:
        main_content_tag: The BeautifulSoup Tag object representing the main content area.
        max_chars: The target maximum character length for each chunk.

    Returns:
        A list of text chunks, where each chunk ideally starts with a heading or is a segment
        of a larger block if no suitable headings are found or if elements exceed max_chars.
    """
    if not main_content_tag:
        logger.error("chunk_by_html_headings called with no main_content_tag.")
        return []

    chunks = []
    current_chunk_elements = []
    current_chunk_len = 0
    # Define heading tags that signal a new logical section
    heading_tags = ['h2', 'h3', 'h4', 'h5', 'h6']
    # Define common block-level tags that contain content
    content_tags = ['p', 'div', 'ul', 'ol', 'table', 'section', 'article']

    # Find all direct children elements that are headings or common content containers
    # recursive=False helps process the document structure level by level
    candidate_elements = main_content_tag.find_all(heading_tags + content_tags, recursive=False)

    # Fallback: If no direct children match, try finding any matching tags within the main content
    if not candidate_elements:
         logger.warning("No direct children matching heading/content tags found. Searching deeper (recursive=True).")
         candidate_elements = main_content_tag.find_all(heading_tags + content_tags, recursive=True)

    # Fallback: If still no candidates, treat the whole tag's text as one chunk (or split if too long)
    if not candidate_elements:
        logger.warning(f"No heading or content tags found within {main_content_tag.name}. Treating its text content as a single block.")
        full_text = main_content_tag.get_text(separator='\n', strip=True)
        if not full_text:
            logger.warning("Main content tag contains no text.")
            return []
        if len(full_text) <= max_chars:
            return [full_text]
        else:
            logger.warning(f"Single block text ({len(full_text)} chars) exceeds max_chars ({max_chars}). Splitting text.")
            split_chunks = []
            start = 0
            while start < len(full_text):
                end = start + max_chars
                split_chunks.append(full_text[start:end])
                start = end
            return split_chunks

    # Process the identified candidate elements
    for element in candidate_elements:
        # Extract text content cleanly
        element_text = element.get_text(separator='\n', strip=True)
        element_len = len(element_text)

        # Skip elements that contain no text after stripping
        if not element_text:
            continue

        is_heading = element.name in heading_tags

        # Condition to finalize the *previous* chunk:
        # 1. We are starting a new section (is_heading is True)
        # 2. Adding the current element would exceed the max_chars limit
        # 3. Crucially, there must be elements in the current_chunk_elements to finalize.
        if current_chunk_elements and (is_heading or (current_chunk_len + element_len > max_chars)):
            chunk_text = "\n\n".join(el.get_text(separator='\n', strip=True) for el in current_chunk_elements).strip()
            if chunk_text: # Ensure we don't add empty strings
                chunks.append(chunk_text)
            # Reset for the next chunk
            current_chunk_elements = []
            current_chunk_len = 0

        # Handle elements that *by themselves* exceed the max_chars limit
        if element_len > max_chars:
            logger.warning(f"Single element <{element.name}> text ({element_len} chars) exceeds max_chars ({max_chars}). Splitting element text.")
            # Finalize any pending chunk before processing the large element
            if current_chunk_elements:
                chunk_text = "\n\n".join(el.get_text(separator='\n', strip=True) for el in current_chunk_elements).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk_elements = []
                current_chunk_len = 0

            # Split the text of the large element
            start = 0
            while start < element_len:
                end = start + max_chars
                # Optionally prepend the tag name if it's a heading being split
                prefix = f"<{element.name}> (split part) " if is_heading and start == 0 else ""
                chunks.append(prefix + element_text[start:end])
                start = end
            # This large element has been processed and split; skip adding it to current_chunk_elements
            continue

        # Add the current element to the list for the *next* chunk to be finalized
        # This happens if it's not a heading that triggered finalization, or if it's the first element.
        current_chunk_elements.append(element)
        # Rough estimate of length, adding 2 for potential "\n\n" joiner
        current_chunk_len += element_len + 2

    # Add the very last chunk if any elements remain
    if current_chunk_elements:
        chunk_text = "\n\n".join(el.get_text(separator='\n', strip=True) for el in current_chunk_elements).strip()
        if chunk_text:
            chunks.append(chunk_text)

    # Final filter for safety, although logic above tries to prevent empty chunks
    final_chunks = [chunk for chunk in chunks if chunk]
    logger.info(f"Chunked content into {len(final_chunks)} chunks.")
    return final_chunks 