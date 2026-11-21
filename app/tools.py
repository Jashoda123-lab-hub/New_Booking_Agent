"""The agent's tools: plain functions over the database.

Each tool opens its own short session and returns plain dicts, so the agent
never holds a transaction open while it waits on the LLM.
"""
import difflib
import re
from datetime import datetime
from typing import List, Optional

from sqlalchemy import and_, exists, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from .db import TIME_FMT, Booking, Property, ViewingSlot

_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "ln": "lane",
    "dr": "drive",
}


def _normalise(text: str) -> str:
    words = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
    return " ".join(_ABBREVIATIONS.get(w, w) for w in words)


def _slot_dict(slot: ViewingSlot, prop: Property) -> dict:
    return {
        "slot_id": slot.id,
        "property_id": prop.id,
        "address": prop.address,
        "starts_at": slot.starts_at,
        "ends_at": slot.ends_at,
    }


class BookingTools:
    def __init__(self, session_factory: sessionmaker):
        self.Session = session_factory

    # 1 · find property --------------------------------------------------
    def find_property(self, query: Optional[str]) -> Optional[dict]:
        """Match free text like '14 oak rd' or 'the Oak Road one' to ONE known property.

        Returns None rather than guessing:
        * a house number in the query must match exactly ('41 Oak Road' is not '14 Oak Road');
        * every street word must be present, allowing small typos ('Oak Raod');
        * if several properties match (e.g. two on the same street, no number given), None.
        """
        if not query or not query.strip():
            return None
        q_words = _normalise(query).split()
        q_numbers = {w for w in q_words if w.isdigit()}
        q_text = [w for w in q_words if not w.isdigit()]

        def word_present(word: str) -> bool:
            return any(
                w == word
                or (len(word) > 3 and sorted(w) == sorted(word))  # swapped letters: "raod"
                or (len(word) > 3 and difflib.SequenceMatcher(None, w, word).ratio() >= 0.8)
                for w in q_text
            )

        with self.Session() as s:
            props = s.scalars(select(Property)).all()
        matches = []
        for p in props:
            a_words = _normalise(p.address).split()
            a_numbers = {w for w in a_words if w.isdigit()}
            street = [w for w in a_words if not w.isdigit()]
            if q_numbers and not (q_numbers & a_numbers):
                continue
            if all(word_present(w) for w in street):
                matches.append(p)
        if len(matches) == 1:
            p = matches[0]
            return {"id": p.id, "address": p.address, "city": p.city}
        return None

    # 2 · get free slots -------------------------------------------------
    def get_free_slots(
        self,
        property_id: int,
        start: datetime,
        end: datetime,
        now: Optional[datetime] = None,
    ) -> List[dict]:
        """Slots for a property that start in [start, end) and have no active booking."""
        lower = max(start, now) if now else start
        active = exists().where(
            and_(Booking.slot_id == ViewingSlot.id, Booking.status == "active")
        )
        stmt = (
            select(ViewingSlot, Property)
            .join(Property, Property.id == ViewingSlot.property_id)
            .where(
                ViewingSlot.property_id == property_id,
                ViewingSlot.starts_at >= lower.strftime(TIME_FMT),
                ViewingSlot.starts_at < end.strftime(TIME_FMT),
                ~active,
            )
            .order_by(ViewingSlot.starts_at)
        )
        with self.Session() as s:
            return [_slot_dict(slot, prop) for slot, prop in s.execute(stmt).all()]

    # 3 · book slot ------------------------------------------------------
    def book_slot(self, slot_id: int, contact: str) -> dict:
        """Insert an active booking. The DB's unique index decides who wins a race."""
        with self.Session() as s:
            row = s.execute(
                select(ViewingSlot, Property)
                .join(Property, Property.id == ViewingSlot.property_id)
                .where(ViewingSlot.id == slot_id)
            ).first()
            if row is None:
                return {"ok": False, "reason": "unknown_slot"}
            slot, prop = row
            booking = Booking(slot_id=slot_id, contact=contact, status="active")
            s.add(booking)
            try:
                s.commit()
            except IntegrityError:
                s.rollback()
                return {"ok": False, "reason": "slot_taken"}
            return {"ok": True, "booking_id": booking.id, "contact": contact, **_slot_dict(slot, prop)}

    # 4 · cancel booking -------------------------------------------------
    def active_bookings_for(self, contact: str, property_id: Optional[int] = None) -> List[dict]:
        stmt = (
            select(Booking, ViewingSlot, Property)
            .join(ViewingSlot, ViewingSlot.id == Booking.slot_id)
            .join(Property, Property.id == ViewingSlot.property_id)
            .where(Booking.contact == contact, Booking.status == "active")
            .order_by(ViewingSlot.starts_at)
        )
        if property_id is not None:
            stmt = stmt.where(Property.id == property_id)
        with self.Session() as s:
            return [
                {"booking_id": b.id, "contact": b.contact, **_slot_dict(slot, prop)}
                for b, slot, prop in s.execute(stmt).all()
            ]

    def cancel_booking(self, booking_id: int, contact: str) -> dict:
        """Cancel one of this contact's active bookings. Frees the slot again."""
        with self.Session() as s:
            result = s.execute(
                update(Booking)
                .where(
                    Booking.id == booking_id,
                    Booking.contact == contact,
                    Booking.status == "active",
                )
                .values(status="cancelled")
            )
            s.commit()
            return {"ok": result.rowcount == 1, "booking_id": booking_id}

    # used by GET /bookings ---------------------------------------------
    def list_bookings(self) -> List[dict]:
        stmt = (
            select(Booking, ViewingSlot, Property)
            .join(ViewingSlot, ViewingSlot.id == Booking.slot_id)
            .join(Property, Property.id == ViewingSlot.property_id)
            .where(Booking.status == "active")
            .order_by(ViewingSlot.starts_at)
        )
        with self.Session() as s:
            return [
                {
                    "booking_id": b.id,
                    "contact": b.contact,
                    "status": b.status,
                    "created_at": b.created_at,
                    **_slot_dict(slot, prop),
                }
                for b, slot, prop in s.execute(stmt).all()
            ]
