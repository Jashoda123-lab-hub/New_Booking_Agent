# Note: Viewing Booking Agent

**Part 1** is the short note the brief asks for (about half a page).
**Part 2** explains the design in more detail, for anyone who wants to go deeper.

---

## Part 1: Short note

**What I built.** A FastAPI service with a LangGraph agent over SQLite. A tenant sends a message; the agent
works out what they want, checks the database, books or cancels a slot (or asks a clarifying question), and
replies. The LLM (Groq, `openai/gpt-oss-120b`) does only two jobs: it reads the message into structured
fields, and it polishes the reply. **Code** picks the slot and calls the tools, so the model can't book
something the database didn't offer. Double bookings are blocked **by the database** with a partial unique
index, and each tenant's conversation is remembered by a LangGraph checkpointer.

**What I learned, and how.** LangGraph was new to me. I learned how a state graph works (nodes, edges,
conditional routing, shared state) and how a checkpointer saves state per conversation, by reading the docs
and running small examples with an AI coding assistant. I set up SQLite with DB Browser and tested the
"one active booking per slot" index by hand before writing any Python. The biggest lesson: unit tests with a
fake LLM are not enough. Running the real model exposed bugs the tests couldn't show, so I turned each one into
a regression test.

**Where the AI tool was wrong, and how I caught it.** The first address matcher used fuzzy string
similarity and matched **"41 Oak Road" to "14 Oak Road"**, which would book a stranger into the wrong flat. A
test asserting that a different house number must *not* match caught it; I replaced it with word-based
matching where the house number must match exactly. Live testing found more: the LLM rewrote "Which property
would you like to view?" into the false *"we don't have any properties available"*, and a repeated "book at
11:00" booked **10:00**. I fixed both in code (every reply is checked against the facts; an exact time is never
swapped for another) and added tests that fail on the old code.

**What I'd improve with more time.** Store times in UTC with per-property time zones; build an evaluation set
of real multi-turn conversations run against the live LLM; authenticate tenants instead of trusting `sender`;
and support rescheduling as one atomic cancel-and-book.

---

## Part 2: How it works, and why

### 2.1 The flow of one message

```
POST /message ─► understand ─┬─► lookup ──► book ──┐
                  (LLM)      │   (tools)   (tool)  │
                             ├─► cancel ───────────┼─► reply ─► { reply, action, booking }
                             │   (tools)           │    (LLM wording, code-checked)
                             └─► clarify ──────────┘
```

| Step | What happens |
|---|---|
| **understand** | The LLM fills a strict schema: `intent` (book / cancel / question / other), `property`, `date`, `preferred_time`, `earliest_time`, `latest_time`, `time_of_day`. Today's date is in the prompt, so "tomorrow" and "Saturday" resolve correctly. Code then **checks** the output (see 2.5). |
| **lookup** | `find_property` turns free text into one known property, or "unknown". `get_free_slots` returns slots with no active booking. Code picks the exact time if it's free, otherwise the first free slot in the window the tenant gave ("if not, any time after 4"). |
| **book / cancel** | `book_slot` inserts the booking; the database decides who wins a race. `cancel_booking` only cancels the sender's own active booking. |
| **clarify** | If the property is unknown, no time is given, or a cancel could mean several bookings, the agent asks instead of guessing. |
| **reply** | Code writes a correct reply from the facts; the LLM may reword it, but only if the reworded text keeps the same times, address and outcome. |

### 2.2 Why a fixed graph, not a ReAct loop

In a ReAct agent the LLM decides which tool to call next, in a loop. That is flexible, but here the tools
**change real data**: a wrong `book_slot` or `cancel_booking` call is a real mistake. So the LLM understands and
phrases, and the graph decides which tools run. This also makes every path testable. The agentic part is the
**multi-turn loop**: when information is missing, the agent asks and continues from saved state.

### 2.3 The double-booking rule lives in the database

```sql
CREATE UNIQUE INDEX one_active_booking_per_slot
ON bookings(slot_id) WHERE status = 'active';
```

Checking "is the slot free?" in Python and then inserting is a race: two requests can both see it free. The
partial unique index makes the second `INSERT` fail with `IntegrityError`, which `book_slot` turns into "slot
taken", and the agent offers alternatives. Because only `active` rows count, a cancelled slot can be booked
again, and cancelling keeps the history instead of deleting rows. A test runs 8 threads at the same slot at the
same moment: exactly one wins.

### 2.4 Conversation memory

Each tenant is one LangGraph **thread** (`thread_id = sender:<name>`). A SQLite checkpointer
(`conversations.db`) saves this state after every message, so it survives a server restart:

| State | Makes this work |
|---|---|
| `pending`: the open request | "Is 14 Oak Road free on Tuesday?" → "yes book at 10:00" |
| `history`: last 6 messages, shown to the LLM | "the 11 o'clock one please" |
| `last_booking`: booked in this chat | "actually, cancel it" |
| `last_seen` | After 30 minutes of silence, start fresh |

Bookings (`viewings.db`) and chat memory (`conversations.db`) are separate files, so memory can be reset
without touching bookings.

### 2.5 Not trusting the model blindly

Every live bug came from the model doing something plausible but wrong, so code checks its output:

- An **address only counts if this message names it.** With history in the prompt, the model copied an earlier
  address into "I want to cancel my viewing." and cancelled without asking.
- **"Cancel it" only counts if the message says "it" or "that one".**
- **An exact time beats a time-of-day.** The model tagged "book at 11:00" as "morning" too, and the code fell
  back to 10:00.
- **Never guess an address.** "41 Oak Road" is not "14 Oak Road". When cancelling, a partial name like "the maple
  one" is matched only against the tenant's own bookings.
- **Replies are fact-checked** (`is_faithful`); if the reworded reply changes a fact, the code-written reply is
  sent instead.

### 2.6 Testing approach

- **30 tests** run in under a second with a fresh SQLite file per test, a fixed clock (Sat 21 Nov 2026,
  09:00) and a fake LLM that returns fixed structured output. No API key needed, and the results are
  deterministic.
- They exercise the real graph, tools, database and API through FastAPI's test client.
- **The tests catch real bugs:** with the unique index removed, the database tests fail; the multi-turn tests
  fail on the code from before each fix.
- Bugs found with the live LLM were copied into tests using the model's actual bad output.

| Bug found | Caught by |
|---|---|
| "41 Oak Road" matched "14 Oak Road" | unit test |
| LLM reply said "no properties available" | live run → `test_unfaithful_llm_reply_falls_back_to_template` |
| "Saturday morning" booked nothing | live run → `test_successful_booking` |
| "yes book at 10:00" lost the property | web page → `test_yes_book_it_after_an_availability_answer` |
| Repeat "book at 11:00" booked 10:00 | live run → `test_exact_time_is_never_swapped_for_another_time` |
| Copied address cancelled without asking | live run → `test_address_copied_from_history_is_ignored` |

### 2.7 Known limits

- Times are local and naive (no time zones).
- `sender` is trusted as identity; there is no login.
- Rescheduling ("move it to 4pm") isn't supported yet.
- Extraction accuracy on unusual phrasing isn't measured; that needs an evaluation set against the live model.
