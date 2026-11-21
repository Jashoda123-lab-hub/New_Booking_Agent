# Viewing Booking Agent

A small AI agent for a property company. A tenant sends a message, and the agent works out what they want,
checks availability in the database, books or cancels a viewing slot (or asks a clarifying question), and
drafts a short reply.

**Stack:** Python · FastAPI · LangGraph · SQLAlchemy · SQLite (Postgres works by changing `DATABASE_URL`) · Groq (`openai/gpt-oss-120b`) by default

## How it works

```
POST /message ──► LangGraph agent ───────────────────────────────────────────► reply + action
                   understand ──► lookup ──► book ──┐
                       │            └───────────────┼──► reply
                       ├──► cancel ─────────────────┤
                       └──► clarify ────────────────┘
                                  │ tools
                                  ▼
                 SQLite: properties · viewing_slots · bookings
```

| Node | What it does |
|---|---|
| `understand` | LLM turns the message into structured fields: `intent` (book / cancel / question / other), `property`, `date`, `preferred_time`, `earliest_time`, `latest_time`. Today's date goes in the prompt so "tomorrow" or "Saturday morning" resolve correctly. |
| `lookup` | Tools `find_property` and `get_free_slots` pick a real free slot. If the property is unknown or no time is given, the agent asks a question instead. |
| `book` | Tool `book_slot`. If another request took the slot a moment earlier, the DB rejects the insert and the agent offers alternatives. |
| `cancel` | Tools `active_bookings_for` and `cancel_booking`. If no property is named, the agent asks which viewing to cancel. A tenant can only cancel their own bookings. |
| `reply` | LLM polishes a reply that code drafted from the facts. `is_faithful` checks the result (same times, address and outcome, questions stay questions). If the check fails or the LLM is down, the code-drafted reply is sent. |

The LLM never decides *which slot gets booked*. It only reads the message and words the reply. All booking
decisions are made by code calling tools, so a hallucinating model can't create a booking that doesn't exist.

### The double-booking rule lives in the database

```sql
CREATE UNIQUE INDEX one_active_booking_per_slot
ON bookings(slot_id) WHERE status = 'active';
```

A partial unique index allows at most one **active** booking per slot. When two requests arrive at the same
moment, both may see the slot as free, but only one `INSERT` can succeed. The other gets an `IntegrityError`,
which `book_slot` turns into "slot taken". Cancelling sets `status = 'cancelled'`, which frees the slot without
deleting history.

## Setup

Requires Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your GROQ_API_KEY in .env (free: https://console.groq.com/keys)
```

## Run

```bash
python -m app.reset_db                      # optional: fresh DB, 3 properties, slots for the next 7 days
uvicorn app.main:get_app --factory --reload
```

- **http://127.0.0.1:8000/**: a small web page. Chat as a tenant (switch between alice, bob and carol to test double
  bookings), use the example messages from the brief, watch **Current bookings** update, and click a free slot to request it.
- **http://127.0.0.1:8000/docs**: the interactive API.

## API

### `POST /message`
```json
{ "sender": "alice@example.com", "text": "Hi, can I see the flat at 14 Oak Road on Saturday morning?" }
```
Response:
```json
{
  "reply": "Hi! Saturday 28 Nov at 10:00 is free at 14 Oak Road. I've booked it for you. Reply here if you need to change it.",
  "action": "booked",
  "booking": { "booking_id": 1, "property_id": 1, "address": "14 Oak Road",
               "starts_at": "2026-11-28 10:00", "ends_at": "2026-11-28 10:30", "contact": "alice@example.com" }
}
```
`action` is one of `booked`, `cancelled`, `clarify`, `no_availability`, `answered`, `none`.

### `GET /bookings`
All active bookings with property, time and contact.

### `GET /properties`
Each property with its free slots for the next 7 days (`?days=` to change). Used by the web page.

```bash
curl -s -X POST localhost:8000/message -H 'Content-Type: application/json' \
  -d '{"sender":"alice","text":"Is 22 Elm Street free tomorrow at 5pm? If not, any time after 4 works."}'
curl -s localhost:8000/bookings
```

## Example conversations

These are the code-drafted replies, which the tests check. With a live LLM the wording is polished but the
facts are the same.

| Tenant | Agent | action |
|---|---|---|
| Hi, can I see the flat at 14 Oak Road on Saturday morning? | Hi! Saturday 28 Nov at 10:00 is free at 14 Oak Road. I've booked it for you. Reply here if you need to change it. | `booked` |
| Is 22 Elm Street free tomorrow at 5pm? If not, any time after 4 works. *(5pm already taken)* | Hi! Sunday 22 Nov at 16:00 is free at 22 Elm Street. I've booked it for you. … | `booked` |
| I want to cancel my viewing. | Happy to help. Which property was the viewing for? You have: 14 Oak Road on Saturday 28 Nov at 10:00; … | `clarify` |
| The Oak Road one | Done. Your viewing at 14 Oak Road on Saturday 28 Nov at 10:00 is cancelled. … | `cancelled` |
| Can I see 99 Banana Street tomorrow at 10? | Sorry, I couldn't find "99 Banana Street" in our listings. We currently have viewings at 14 Oak Road, 22 Elm Street, 7 Maple Avenue. Which property did you mean? | `clarify` |
| I'd like to book a viewing | Happy to help. Which property would you like to view? | `clarify` |
| Book 14 Oak Road Saturday at 10am *(already booked)* | Sorry, that time isn't available at 14 Oak Road. The nearest free slots are Saturday 28 Nov at 11:00, … | `no_availability` |

## Tests

```bash
pytest -q
```

The tests use a fresh SQLite file per test, a fixed clock (Sat 21 Nov 2026, 09:00) and a fake LLM that returns
fixed structured output. That makes them deterministic and runnable without an API key, and they exercise the real
graph, tools, database and API.

| Required case | Test(s) |
|---|---|
| Successful booking | `test_successful_booking`, `test_is_it_free_else_any_time_after_4` |
| Blocked double booking | `test_double_booking_is_blocked_through_the_agent`, `test_simultaneous_bookings_only_one_wins` (8 threads, one slot, exactly one wins), `test_database_itself_rejects_a_second_active_booking` (raw SQL, no Python logic) |
| Cancellation | `test_cancellation_frees_the_slot`, `test_cancel_without_property_asks_which_then_cancels`, `test_cannot_cancel_someone_elses_booking` |
| Unknown property | `test_unknown_property_gets_a_clarifying_question`, `test_similar_but_different_address_is_not_guessed` |
| Vague request | `test_vague_request_gets_a_clarifying_question`, `test_property_but_no_time_asks_for_a_time` |

Extra: `test_unfaithful_llm_reply_falls_back_to_template` and `test_is_faithful_checks` use real bad replies seen from the live model.

Sanity check: if you remove the unique index from `app/db.py`, the two DB-level double-booking tests fail.

## Project layout

```
app/
  db.py        models, partial unique index, seed data
  tools.py     find_property, get_free_slots, book_slot, active_bookings_for, cancel_booking, list_bookings
  agent.py     LangGraph graph, LLM prompts, reply templates
  main.py      FastAPI app (POST /message, GET /bookings, GET /properties, GET /health, GET /)
  static/      index.html: the web page (plain HTML + JS, no build step)
  reset_db.py  recreate and seed the database
tests/         pytest suite (fake LLM, temp DB)
```
