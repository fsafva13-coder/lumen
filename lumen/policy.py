"""Decides what LUMEN does next.

A policy looks at the challenges run so far and returns the next action:
  "light"   run the light challenge
  "turn"    run the head-turn challenge
  "accept"  verified as a live person
  "reject"  not a live person
  "refer"   could not decide - hand over to a human reviewer

`fixed_policy` is the plain rule-based baseline. The Bedrock agent will be a
second policy with the same inputs and outputs, so the two can be compared
on the same sessions (accuracy and time to decision).

Why two challenges are required: on the pilot clips the light challenge
caught every phone screen but passed a steady printed photo, and the head
turn caught every print and phone but can be replayed from a video. Each
covers the other's blind spot, so "live" needs both.
"""
from __future__ import annotations

MAX_TRIES = 2  # per challenge, so a session is at most 4 challenges


def fixed_policy(history: list[dict]) -> str:
    lights = [h for h in history if h["challenge"] == "light"]
    turns = [h for h in history if h["challenge"] == "turn"]

    if not lights:
        return "light"
    last = lights[-1]["verdict"]
    if last == "fake":
        return "reject"
    if last == "inconclusive":
        return "light" if len(lights) < MAX_TRIES else "refer"

    if not turns:
        return "turn"
    if turns[-1]["verdict"] == "live":
        return "accept"
    return "turn" if len(turns) < MAX_TRIES else "refer"


DECISION_TEXT = {
    "accept": ("LIVE PERSON VERIFIED", "Verified. You are a live person."),
    "reject": ("NOT A LIVE PERSON", "Rejected. This is not a live person."),
    "refer": ("NOT VERIFIED", "Not verified. Referring to a human reviewer."),
}
