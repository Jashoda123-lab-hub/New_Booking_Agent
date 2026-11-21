# Note

> DRAFT: rewrite this in your own words before submitting. They asked for an *honest* note, and you'll be
> asked about it on the walkthrough call.

**What I built.** A FastAPI service with a LangGraph agent (understand → lookup → book / cancel / clarify →
reply) over SQLite. The LLM (Groq, gpt-oss-120b) does only two jobs: it turns the message into structured fields,
and it polishes the reply. Choosing and booking a slot is done by code calling tools, so the model can't book
something the database didn't offer. The "one active booking per slot" rule is a partial unique index
(`UNIQUE (slot_id) WHERE status = 'active'`). A concurrency test sends 8 simultaneous bookings for one slot and
exactly one succeeds. I kept the suggested data model. Cancelling sets `status = 'cancelled'` instead of
deleting the row, so history is kept and the slot frees up automatically.

**What I learned, and how.** *(Fill in your own: e.g. LangGraph's StateGraph, nodes returning partial state,
conditional edges for routing; `with_structured_output` for reliable extraction; partial indexes; how SQLite
handles concurrent writers. Say how you learned it: docs, examples, trying things.)*

**Where the AI tool was wrong, and how I caught it.** The first version of `find_property` used fuzzy string
similarity to forgive typos. It matched **"41 Oak Road" to "14 Oak Road"**, which would have booked a
stranger into the wrong flat, and it failed to match "Oak Road 14" because of the word order. I wrote a test
asserting that a different house number must *not* match, and the test failed. I replaced the matching with
word-based matching where the house number must match exactly, street words may contain small typos, and an
ambiguous query returns "unknown" so the agent asks instead of guessing. I also checked that the double-booking
tests are real: with the unique index removed, both DB-level tests fail.

The LLM itself was also wrong once I ran it live (the unit tests use a fake LLM, so they couldn't show this):
the default Groq model had been retired; "Saturday morning" came back with no time window, so nothing was booked;
and the reply-polishing step rewrote "Which property would you like to view?" into *"we don't have any properties
available"*, which is false. It also dropped "I've booked it" from a real booking. Fixes: "morning / afternoon /
evening" is now an enum the model picks and code maps to hours, and every polished reply is checked against the
code-drafted one (same times, same address, same outcome, questions stay questions). If the check fails, the code
draft is sent. Both failures are now regression tests.

**What I'd improve with more time.**
- Store conversation state (the half-finished request waiting on a clarifying answer) in the DB or a LangGraph
  checkpointer, instead of in process memory, which is lost on restart and not shared across workers.
- Time zones: times are naive local times; I'd store UTC and convert per property.
- An evaluation set of real tenant messages run against the live LLM, to measure extraction accuracy
  (the unit tests use a fake LLM so they're deterministic).
- Authentication: today `sender` is trusted as the contact identity.
- Rescheduling ("can I move it to 11?") as a single, atomic cancel-and-book.
