"""Active-learning utilities.

The first implementation is intentionally a dry run: it exercises the
generate -> validate -> score -> select -> optional fine-tune plumbing without
claiming discovered materials. Real discovery requires replacing the heuristic
score with a validated relaxation/property oracle.
"""

