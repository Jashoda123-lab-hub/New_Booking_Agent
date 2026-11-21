"""Database models, engine and seed data.

The key rule — a slot can have at most ONE active booking — lives in the
database as a partial unique index (`one_active_booking_per_slot`), so it holds
even when two requests race each other. Python never has to "check then insert".
"""
import os
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    create_engine,
    event,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

# Times are stored as sortable text "YYYY-MM-DD HH:MM" (local time of the agency).
TIME_FMT = "%Y-%m-%d %H:%M"

DEFAULT_DB_URL = "sqlite:///viewings.db"


class Base(DeclarativeBase):
    pass


class Property(Base):
    __tablename__ = "properties"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    address: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    city: Mapped[Optional[str]] = mapped_column(String)

    slots = relationship("ViewingSlot", back_populates="property")


class ViewingSlot(Base):
    __tablename__ = "viewing_slots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    property_id: Mapped[int] = mapped_column(ForeignKey("properties.id"), nullable=False)
    starts_at: Mapped[str] = mapped_column(String, nullable=False)
    ends_at: Mapped[str] = mapped_column(String, nullable=False)

    property = relationship("Property", back_populates="slots")


class Booking(Base):
    __tablename__ = "bookings"
    __table_args__ = (
        CheckConstraint("status IN ('active','cancelled')", name="booking_status"),
        # The double-booking guard. Works on SQLite and Postgres.
        Index(
            "one_active_booking_per_slot",
            "slot_id",
            unique=True,
            sqlite_where=text("status = 'active'"),
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slot_id: Mapped[int] = mapped_column(ForeignKey("viewing_slots.id"), nullable=False)
    contact: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="active")
    created_at: Mapped[Optional[str]] = mapped_column(String, server_default=text("CURRENT_TIMESTAMP"))

    slot = relationship("ViewingSlot")


def make_engine(url: Optional[str] = None) -> Engine:
    url = url or os.getenv("DATABASE_URL", DEFAULT_DB_URL)
    connect_args = {"check_same_thread": False, "timeout": 10} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args)

    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys = ON")
            cur.close()

    return engine


def init_db(engine: Engine, seed_from: Optional[datetime] = None) -> None:
    """Create tables (if missing) and seed demo data if the DB is empty."""
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as s:
        if s.scalar(select(Property.id).limit(1)) is None:
            seed(s, seed_from or datetime.now())
            s.commit()


SEED_PROPERTIES = [
    ("14 Oak Road", "London"),
    ("22 Elm Street", "London"),
    ("7 Maple Avenue", "London"),
]
SEED_TIMES = ["10:00", "11:00", "16:00", "17:00"]


def seed(session, base: datetime) -> None:
    """3 properties, 4 half-hour slots a day for the next 7 days."""
    for address, city in SEED_PROPERTIES:
        prop = Property(address=address, city=city)
        session.add(prop)
        session.flush()
        for day in range(1, 8):
            d = (base + timedelta(days=day)).date()
            for t in SEED_TIMES:
                start = datetime.strptime(f"{d} {t}", TIME_FMT)
                session.add(
                    ViewingSlot(
                        property_id=prop.id,
                        starts_at=start.strftime(TIME_FMT),
                        ends_at=(start + timedelta(minutes=30)).strftime(TIME_FMT),
                    )
                )
