"""Signing in to a model provider, and keeping the resulting credentials.

Today that's one flow: OpenAI's "Sign in with ChatGPT", which lets someone
use their ChatGPT plan in place of an API key (app/auth/chatgpt.py). This
package owns the sign-in, the stored tokens and their refresh; providers/
only receives a fresh access token per request and knows none of this.
"""
