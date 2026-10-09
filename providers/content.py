"""Shared user-content extraction for both OpenAI wire shapes.

Mirrors tau_ai/content.py's role: one place to split a UserMessage's
content into visible text plus sendable images, so each adapter only has
to build its own wire-specific block shape (image_url vs input_image) from
the result. Scaled down from tau's version - no non-vision-model
downgrade/placeholder text, since this app has no per-model vision-capability
catalog (one configured deployment per environment, not tau's many-vendor
catalog) - see app/openai_responses.py's own module docstring for the
matching scope decision on the Responses API side.
"""

from __future__ import annotations

from brain.messages import ImageContent, TextContent, UserContentBlock


def text_and_images(content: str | list[UserContentBlock]) -> tuple[str, list[ImageContent]]:
    """Return a UserMessage's visible text and any attached images."""
    if isinstance(content, str):
        return content, []
    text = "".join(b.text for b in content if isinstance(b, TextContent))
    images = [b for b in content if isinstance(b, ImageContent)]
    return text, images
