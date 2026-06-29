"""Adapters — the ONLY layer allowed to import third-party / network libraries.

Each subpackage implements one port from :mod:`garden_agent.ports` structurally
(duck-typed, no inheritance), so you can swap any of them in tests by handing a
fake to the application service.
"""
