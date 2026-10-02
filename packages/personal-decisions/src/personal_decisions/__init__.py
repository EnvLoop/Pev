"""MUSE-style personal decision records: real tables -> knowledge base -> shards -> question families -> rendered
Kev-format records. See README.md for the record contract."""


class PersonalDecisionsError(RuntimeError):
    """A user-facing failure (bad input, missing setting); the CLI prints the message and exits non-zero."""
