"""Portable, provider-neutral agent core - messages, loop, harness, events.

Mirrors HuggingFace tau's tau_agent package: this has zero knowledge of
FastAPI, Azure, this app's Settings, or any concrete provider vendor. Only
providers/ and app/ depend on brain/ - never the other way around.
"""
